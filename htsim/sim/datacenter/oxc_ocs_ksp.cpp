// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_ocs_ksp.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace std;

static constexpr uint64_t TAG_PATH_RANK = 0x4b53500001ULL;
static constexpr uint64_t TAG_CHOICE = 0x4b53430001ULL;

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static string join_nodes(const vector<uint32_t>& nodes) {
    stringstream ss;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (i) {
            ss << ";";
        }
        ss << nodes[i];
    }
    return ss.str();
}

static bool contains_directed_edge(
        const vector<pair<uint32_t, uint32_t>>& edges,
        uint32_t src,
        uint32_t dst) {
    return find(edges.begin(), edges.end(), make_pair(src, dst)) != edges.end();
}

static bool has_prefix(const vector<uint32_t>& path, const vector<uint32_t>& prefix) {
    if (path.size() < prefix.size()) {
        return false;
    }
    return equal(prefix.begin(), prefix.end(), path.begin());
}

OxcOcsKspRouter::OxcOcsKspRouter(
        const OxcOcsGraph& graph,
        uint32_t groups,
        uint32_t l1_eps_per_plane,
        OxcOcsKspParams params)
    : _graph(graph),
      _groups(groups),
      _l1_eps_per_plane(l1_eps_per_plane),
      _params(params) {
    if (_groups == 0) {
        throw invalid_argument("KSP groups must be positive");
    }
    if (_l1_eps_per_plane == 0) {
        throw invalid_argument("KSP l1_eps_per_plane must be positive");
    }
    if (_graph.nodes != _groups * _l1_eps_per_plane) {
        throw invalid_argument("KSP graph node count must equal groups * l1_eps_per_plane");
    }
    if (_params.k == 0) {
        throw invalid_argument("KSP k must be positive");
    }
    if (_params.max_paths_per_pair == 0) {
        throw invalid_argument("KSP max_paths_per_pair must be positive");
    }

    _effective_max_hops = _params.max_hops == 0 ? (_graph.nodes == 0 ? 0 : _graph.nodes - 1)
                                                : _params.max_hops;

    _paths.resize(static_cast<size_t>(_graph.nodes) * _groups);
    for (uint32_t src_node = 0; src_node < _graph.nodes; src_node++) {
        for (uint32_t dst_group = 0; dst_group < _groups; dst_group++) {
            _paths[index(src_node, dst_group)] = build_paths(src_node, dst_group);
        }
    }
}

size_t OxcOcsKspRouter::index(uint32_t src_node, uint32_t dst_group) const {
    if (src_node >= _graph.nodes || dst_group >= _groups) {
        throw out_of_range("KSP src_node or dst_group out of range");
    }
    return static_cast<size_t>(src_node) * _groups + dst_group;
}

bool OxcOcsKspRouter::is_dst_group(uint32_t template_node, uint32_t dst_group) const {
    if (template_node >= _graph.nodes || dst_group >= _groups) {
        throw out_of_range("KSP node or dst_group out of range");
    }
    return oxc_ocs_decode_template_node(template_node, _l1_eps_per_plane).first == dst_group;
}

const vector<OxcOcsKspPath>& OxcOcsKspRouter::paths(uint32_t src_node, uint32_t dst_group) const {
    return _paths.at(index(src_node, dst_group));
}

vector<OxcOcsKspPath> OxcOcsKspRouter::build_paths(uint32_t src_node, uint32_t dst_group) const {
    if (src_node >= _graph.nodes || dst_group >= _groups) {
        throw out_of_range("KSP src_node or dst_group out of range");
    }
    if (is_dst_group(src_node, dst_group)) {
        return {};
    }

    return yen_paths(src_node, dst_group);
}

vector<uint32_t> OxcOcsKspRouter::shortest_path_to_group(
        uint32_t start_node,
        uint32_t dst_group,
        const vector<uint8_t>& banned_nodes,
        const vector<pair<uint32_t, uint32_t>>& banned_edges) const {
    if (start_node >= _graph.nodes || dst_group >= _groups) {
        throw out_of_range("KSP shortest path start_node or dst_group out of range");
    }
    if (banned_nodes.size() != _graph.nodes) {
        throw invalid_argument("KSP banned_nodes size mismatch");
    }
    if (banned_nodes.at(start_node)) {
        return {};
    }

    vector<int32_t> parent(_graph.nodes, -1);
    vector<int32_t> dist(_graph.nodes, -1);
    queue<uint32_t> q;
    q.push(start_node);
    dist[start_node] = 0;

    while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();

        if (current != start_node && is_dst_group(current, dst_group)) {
            vector<uint32_t> path;
            uint32_t node = current;
            while (node != start_node) {
                path.push_back(node);
                node = static_cast<uint32_t>(parent.at(node));
            }
            path.push_back(start_node);
            reverse(path.begin(), path.end());
            if (path.size() > 1 && path.size() - 1 <= _effective_max_hops) {
                return path;
            }
            return {};
        }

        if (static_cast<uint32_t>(dist[current]) >= _effective_max_hops) {
            continue;
        }

        vector<uint32_t> neighbors = _graph.adjacency.at(current);
        sort(neighbors.begin(), neighbors.end());
        stable_sort(neighbors.begin(), neighbors.end(), [&](uint32_t a, uint32_t b) {
            uint64_t ka = splitmix64(_params.seed ^ TAG_PATH_RANK ^ current ^ a ^ dst_group);
            uint64_t kb = splitmix64(_params.seed ^ TAG_PATH_RANK ^ current ^ b ^ dst_group);
            if (ka == kb) {
                return a < b;
            }
            return ka < kb;
        });

        for (uint32_t next : neighbors) {
            if (banned_nodes.at(next) || dist.at(next) >= 0) {
                continue;
            }
            if (contains_directed_edge(banned_edges, current, next)) {
                continue;
            }
            parent[next] = static_cast<int32_t>(current);
            dist[next] = dist[current] + 1;
            q.push(next);
        }
    }

    return {};
}

vector<OxcOcsKspPath> OxcOcsKspRouter::yen_paths(uint32_t src_node, uint32_t dst_group) const {
    auto decoded = oxc_ocs_decode_template_node(src_node, _l1_eps_per_plane);

    vector<uint8_t> no_banned_nodes(_graph.nodes, 0);
    vector<pair<uint32_t, uint32_t>> no_banned_edges;
    vector<uint32_t> first = shortest_path_to_group(
            src_node, dst_group, no_banned_nodes, no_banned_edges);
    if (first.empty()) {
        return {};
    }

    vector<vector<uint32_t>> accepted;
    vector<vector<uint32_t>> candidates;
    set<vector<uint32_t>> seen_paths;
    accepted.push_back(first);
    seen_paths.insert(first);

    auto path_less = [&](const vector<uint32_t>& a, const vector<uint32_t>& b) {
        uint32_t ahops = a.empty() ? 0 : static_cast<uint32_t>(a.size() - 1);
        uint32_t bhops = b.empty() ? 0 : static_cast<uint32_t>(b.size() - 1);
        if (ahops != bhops) {
            return ahops < bhops;
        }
        uint64_t arank = rank_key(a, src_node, dst_group);
        uint64_t brank = rank_key(b, src_node, dst_group);
        if (arank != brank) {
            return arank < brank;
        }
        return a < b;
    };

    while (accepted.size() < _params.k) {
        const vector<uint32_t>& prev = accepted.back();
        for (size_t spur_index = 0; spur_index + 1 < prev.size(); spur_index++) {
            uint32_t spur_node = prev[spur_index];
            vector<uint32_t> root_path(prev.begin(), prev.begin() + spur_index + 1);

            vector<uint8_t> banned_nodes(_graph.nodes, 0);
            for (size_t i = 0; i + 1 < root_path.size(); i++) {
                banned_nodes.at(root_path[i]) = 1;
            }

            vector<pair<uint32_t, uint32_t>> banned_edges;
            for (const auto& path : accepted) {
                if (path.size() > spur_index + 1 && has_prefix(path, root_path)) {
                    banned_edges.emplace_back(path[spur_index], path[spur_index + 1]);
                }
            }

            vector<uint32_t> spur_path = shortest_path_to_group(
                    spur_node, dst_group, banned_nodes, banned_edges);
            if (spur_path.empty()) {
                continue;
            }

            vector<uint32_t> total_path = root_path;
            total_path.insert(total_path.end(), spur_path.begin() + 1, spur_path.end());
            if (total_path.size() <= 1 || total_path.size() - 1 > _effective_max_hops) {
                continue;
            }

            set<uint32_t> unique_nodes(total_path.begin(), total_path.end());
            if (unique_nodes.size() != total_path.size()) {
                continue;
            }
            if (seen_paths.insert(total_path).second) {
                candidates.push_back(total_path);
                if (candidates.size() >= _params.max_paths_per_pair) {
                    break;
                }
            }
        }

        if (candidates.empty()) {
            break;
        }

        auto best_it = min_element(candidates.begin(), candidates.end(), path_less);
        accepted.push_back(*best_it);
        candidates.erase(best_it);
    }

    vector<OxcOcsKspPath> result;
    result.reserve(accepted.size());
    for (uint32_t i = 0; i < accepted.size(); i++) {
        OxcOcsKspPath path;
        path.src_node = src_node;
        path.src_group = decoded.first;
        path.src_eps = decoded.second;
        path.dst_group = dst_group;
        path.path_id = i;
        path.nodes = accepted[i];
        result.push_back(path);
    }
    return result;
}

uint64_t OxcOcsKspRouter::rank_key(
        const vector<uint32_t>& path,
        uint32_t src_node,
        uint32_t dst_group) const {
    uint64_t h = splitmix64(_params.seed);
    h = splitmix64(h ^ TAG_PATH_RANK);
    h = splitmix64(h ^ src_node);
    h = splitmix64(h ^ dst_group);
    for (uint32_t node : path) {
        h = splitmix64(h ^ node);
    }
    return h;
}

uint64_t OxcOcsKspRouter::choice_key(
        uint32_t src_node,
        uint32_t dst_group,
        uint32_t flow_id,
        uint32_t path_id,
        uint32_t packet_id) const {
    uint64_t h = splitmix64(_params.seed);
    h = splitmix64(h ^ TAG_CHOICE);
    h = splitmix64(h ^ src_node);
    h = splitmix64(h ^ dst_group);
    h = splitmix64(h ^ flow_id);
    h = splitmix64(h ^ path_id);
    h = splitmix64(h ^ packet_id);
    return h;
}

uint32_t OxcOcsKspRouter::choose_path(
        uint32_t src_node,
        uint32_t dst_group,
        uint32_t flow_id,
        uint32_t path_id,
        uint32_t packet_id,
        uint32_t rr_counter,
        OxcOcsKspChoice choice) const {
    const auto& ps = paths(src_node, dst_group);
    if (ps.empty()) {
        return numeric_limits<uint32_t>::max();
    }

    uint32_t selected = 0;
    if (choice == OxcOcsKspChoice::PACKET_RR) {
        selected = rr_counter % ps.size();
    } else if (choice == OxcOcsKspChoice::PACKET_HASH) {
        selected = static_cast<uint32_t>(choice_key(src_node, dst_group, flow_id, path_id, packet_id)
                                         % ps.size());
    } else {
        selected = static_cast<uint32_t>(choice_key(src_node, dst_group, flow_id, path_id, 0)
                                         % ps.size());
    }
    return ps.at(selected).path_id;
}

uint32_t OxcOcsKspRouter::next_hop(
        uint32_t src_node,
        uint32_t dst_group,
        uint32_t ksp_path_id,
        uint32_t current_node) const {
    if (is_dst_group(current_node, dst_group)) {
        return numeric_limits<uint32_t>::max();
    }

    const auto& ps = paths(src_node, dst_group);
    if (ksp_path_id >= ps.size()) {
        throw out_of_range("KSP path_id out of range");
    }

    const vector<uint32_t>& nodes = ps.at(ksp_path_id).nodes;
    for (size_t i = 0; i + 1 < nodes.size(); i++) {
        if (nodes[i] == current_node) {
            return nodes[i + 1];
        }
    }
    throw invalid_argument("KSP current_node is not on the selected path before the destination");
}

uint64_t OxcOcsKspRouter::total_paths() const {
    uint64_t total = 0;
    for (const auto& entry : _paths) {
        total += entry.size();
    }
    return total;
}

uint32_t OxcOcsKspRouter::min_path_count() const {
    uint32_t min_count = numeric_limits<uint32_t>::max();
    bool any = false;
    for (uint32_t src_node = 0; src_node < _graph.nodes; src_node++) {
        uint32_t src_group = oxc_ocs_decode_template_node(src_node, _l1_eps_per_plane).first;
        for (uint32_t dst_group = 0; dst_group < _groups; dst_group++) {
            if (dst_group == src_group) {
                continue;
            }
            any = true;
            min_count = min<uint32_t>(min_count, paths(src_node, dst_group).size());
        }
    }
    return any ? min_count : 0;
}

uint32_t OxcOcsKspRouter::max_path_count() const {
    uint32_t max_count = 0;
    for (uint32_t src_node = 0; src_node < _graph.nodes; src_node++) {
        uint32_t src_group = oxc_ocs_decode_template_node(src_node, _l1_eps_per_plane).first;
        for (uint32_t dst_group = 0; dst_group < _groups; dst_group++) {
            if (dst_group == src_group) {
                continue;
            }
            max_count = max<uint32_t>(max_count, paths(src_node, dst_group).size());
        }
    }
    return max_count;
}

void write_oxc_ocs_ksp_paths_csv(
        const OxcOcsKspRouter& router,
        const string& path) {
    ofstream out(path);
    if (!out.is_open()) {
        throw runtime_error("failed to open Oxc OCS KSP path output: " + path);
    }

    out << "# groups " << router.groups() << "\n";
    out << "# l1_eps_per_plane " << router.l1_eps_per_plane() << "\n";
    out << "# nodes " << router.graph().nodes << "\n";
    out << "# degree " << router.graph().degree << "\n";
    out << "# k " << router.params().k << "\n";
    out << "# max_hops " << router.effective_max_hops() << "\n";
    out << "# ksp_seed " << router.params().seed << "\n";
    out << "src_node,src_group,src_eps,dst_group,path_id,hop_count,path_nodes\n";

    for (uint32_t src_node = 0; src_node < router.graph().nodes; src_node++) {
        for (uint32_t dst_group = 0; dst_group < router.groups(); dst_group++) {
            const auto& paths = router.paths(src_node, dst_group);
            for (const auto& ksp_path : paths) {
                out << ksp_path.src_node << ","
                    << ksp_path.src_group << ","
                    << ksp_path.src_eps << ","
                    << ksp_path.dst_group << ","
                    << ksp_path.path_id << ","
                    << (ksp_path.nodes.empty() ? 0 : ksp_path.nodes.size() - 1) << ","
                    << join_nodes(ksp_path.nodes) << "\n";
            }
        }
    }
}
