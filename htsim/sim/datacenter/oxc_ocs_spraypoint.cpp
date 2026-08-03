// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_ocs_spraypoint.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace std;

static constexpr uint64_t TAG_WP = 0x57500001ULL;
static constexpr uint64_t TAG_PARENT = 0x50410001ULL;
static constexpr uint64_t TAG_PARENT_IR = 0x50490001ULL;
static constexpr uint64_t TAG_PARENT_OR = 0x504f0001ULL;
static constexpr uint64_t TAG_FALLBACK = 0x46420001ULL;
static constexpr uint64_t TAG_CHOICE = 0x43480001ULL;

static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static vector<uint32_t> sorted_unique(vector<uint32_t> items) {
    sort(items.begin(), items.end());
    items.erase(unique(items.begin(), items.end()), items.end());
    return items;
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

OxcOcsSprayPointRouter::OxcOcsSprayPointRouter(
        const OxcOcsGraph& graph,
        uint32_t groups,
        uint32_t l1_eps_per_plane,
        OxcOcsSprayPointParams params)
    : _graph(graph),
      _groups(groups),
      _l1_eps_per_plane(l1_eps_per_plane),
      _params(params) {
    if (_groups == 0) {
        throw invalid_argument("SprayPoint groups must be positive");
    }
    if (_l1_eps_per_plane == 0) {
        throw invalid_argument("SprayPoint l1_eps_per_plane must be positive");
    }
    if (_graph.nodes != _groups * _l1_eps_per_plane) {
        throw invalid_argument("SprayPoint graph node count must equal groups * l1_eps_per_plane");
    }
    if (_params.spray_p == 0) {
        throw invalid_argument("SprayPoint spray_p must be positive");
    }
    if (_params.spray_h == 0) {
        throw invalid_argument("SprayPoint spray_h must be positive");
    }

    _params.spray_p = min(_params.spray_p, max<uint32_t>(1, _graph.degree));
    _params.spray_h = min(_params.spray_h, max<uint32_t>(1, _graph.degree));
    _levels = derive_levels();

    _states.reserve(_groups);
    for (uint32_t group = 0; group < _groups; group++) {
        _states.push_back(build_state(group));
    }
}

uint32_t OxcOcsSprayPointRouter::derive_levels() const {
    if (_params.spray_levels >= 0) {
        return max<uint32_t>(1, static_cast<uint32_t>(_params.spray_levels));
    }
    if (_params.spray_p <= 1 || _graph.degree == 0) {
        return 1;
    }

    double denom = 2.0 * static_cast<double>(_graph.degree) * static_cast<double>(_graph.degree);
    double ratio = static_cast<double>(_graph.nodes) / max(denom, 1.0);
    if (ratio <= 1.0) {
        return 1;
    }
    return max<uint32_t>(1, static_cast<uint32_t>(ceil(log(ratio) / log(static_cast<double>(_params.spray_p)))));
}

const OxcOcsSprayPointDestinationState& OxcOcsSprayPointRouter::state_for_group(uint32_t dst_group) const {
    if (dst_group >= _states.size()) {
        throw out_of_range("SprayPoint dst_group out of range");
    }
    return _states[dst_group];
}

bool OxcOcsSprayPointRouter::is_dst_group(uint32_t template_node, uint32_t dst_group) const {
    if (template_node >= _graph.nodes || dst_group >= _groups) {
        throw out_of_range("SprayPoint node or dst_group out of range");
    }
    return oxc_ocs_decode_template_node(template_node, _l1_eps_per_plane).first == dst_group;
}

vector<uint32_t> OxcOcsSprayPointRouter::source_spray_next_hops(uint32_t current_template_node) const {
    if (current_template_node >= _graph.nodes) {
        throw out_of_range("SprayPoint current_template_node out of range");
    }
    return sorted_unique(_graph.adjacency.at(current_template_node));
}

const vector<uint32_t>& OxcOcsSprayPointRouter::pointing_next_hops(
        uint32_t current_template_node,
        uint32_t dst_group) const {
    if (current_template_node >= _graph.nodes) {
        throw out_of_range("SprayPoint current_template_node out of range");
    }
    return state_for_group(dst_group).parents.at(current_template_node);
}

uint32_t OxcOcsSprayPointRouter::choose_next_hop(
        uint32_t current_template_node,
        uint32_t dst_group,
        bool source_step,
        uint32_t flow_id,
        uint32_t path_id,
        uint32_t rr_counter,
        OxcOcsSprayPointChoice choice) const {
    if (is_dst_group(current_template_node, dst_group)) {
        return numeric_limits<uint32_t>::max();
    }

    vector<uint32_t> owned_candidates;
    const vector<uint32_t>* candidates = nullptr;
    if (source_step) {
        owned_candidates = source_spray_next_hops(current_template_node);
        candidates = &owned_candidates;
    } else {
        candidates = &pointing_next_hops(current_template_node, dst_group);
    }

    if (candidates->empty()) {
        return numeric_limits<uint32_t>::max();
    }

    uint32_t index = 0;
    if (choice == OxcOcsSprayPointChoice::FLOW_HASH) {
        uint64_t h = splitmix64(_params.spray_seed);
        h = splitmix64(h ^ TAG_CHOICE);
        h = splitmix64(h ^ flow_id);
        h = splitmix64(h ^ path_id);
        h = splitmix64(h ^ current_template_node);
        h = splitmix64(h ^ dst_group);
        index = static_cast<uint32_t>(h % candidates->size());
    } else {
        index = rr_counter % candidates->size();
    }
    return candidates->at(index);
}

OxcOcsSprayPointDestinationState OxcOcsSprayPointRouter::build_state(uint32_t dst_group) const {
    OxcOcsSprayPointDestinationState state;
    state.dst_group = dst_group;
    state.parents.assign(_graph.nodes, {});
    state.node_level.assign(_graph.nodes, -1);
    state.is_dst.assign(_graph.nodes, 0);
    state.is_inner_ring.assign(_graph.nodes, 0);
    state.is_outer_ring.assign(_graph.nodes, 0);

    vector<uint8_t> assigned(_graph.nodes, 0);
    for (uint32_t eps = 0; eps < _l1_eps_per_plane; eps++) {
        uint32_t node = oxc_ocs_template_node_id(dst_group, eps, _l1_eps_per_plane);
        state.dst_nodes.push_back(node);
        state.is_dst[node] = 1;
        assigned[node] = 1;
    }

    vector<uint32_t> wp0;
    for (uint32_t dst_node : state.dst_nodes) {
        for (uint32_t neighbor : _graph.adjacency.at(dst_node)) {
            if (!assigned[neighbor]) {
                wp0.push_back(neighbor);
            }
        }
    }
    wp0 = sorted_unique(wp0);
    state.waypoint_levels.push_back(wp0);
    for (uint32_t node : wp0) {
        state.node_level[node] = 0;
        assigned[node] = 1;
    }

    for (uint32_t level = 1; level <= _levels; level++) {
        set<uint32_t> cur_set;
        for (uint32_t prev : state.waypoint_levels.at(level - 1)) {
            vector<uint32_t> candidates;
            for (uint32_t neighbor : _graph.adjacency.at(prev)) {
                if (!assigned[neighbor]) {
                    candidates.push_back(neighbor);
                }
            }
            vector<uint32_t> chosen = pick(candidates, _params.spray_p, TAG_WP, dst_group, level, prev);
            cur_set.insert(chosen.begin(), chosen.end());
        }

        vector<uint32_t> cur(cur_set.begin(), cur_set.end());
        state.waypoint_levels.push_back(cur);
        for (uint32_t node : cur) {
            state.node_level[node] = static_cast<int32_t>(level);
            assigned[node] = 1;
        }
    }

    vector<uint32_t> wp_last = state.waypoint_levels.empty() ? vector<uint32_t>{}
                                                            : state.waypoint_levels.back();
    set<uint32_t> inner_set;
    for (uint32_t node : wp_last) {
        for (uint32_t neighbor : _graph.adjacency.at(node)) {
            if (!assigned[neighbor]) {
                inner_set.insert(neighbor);
            }
        }
    }
    state.inner_ring.assign(inner_set.begin(), inner_set.end());
    for (uint32_t node : state.inner_ring) {
        state.is_inner_ring[node] = 1;
    }

    vector<uint8_t> assigned_with_inner = assigned;
    for (uint32_t node : state.inner_ring) {
        assigned_with_inner[node] = 1;
    }
    for (uint32_t node = 0; node < _graph.nodes; node++) {
        if (!assigned_with_inner[node]) {
            state.outer_ring.push_back(node);
            state.is_outer_ring[node] = 1;
        }
    }

    vector<int32_t> dist_to_dst = multi_source_distances(state.dst_nodes);
    vector<int32_t> dist_to_inner(_graph.nodes, -1);
    if (!state.inner_ring.empty()) {
        dist_to_inner = multi_source_distances(state.inner_ring);
    }

    for (uint32_t node = 0; node < _graph.nodes; node++) {
        if (state.is_dst[node]) {
            continue;
        }

        if (state.node_level[node] >= 0) {
            uint32_t level = static_cast<uint32_t>(state.node_level[node]);
            vector<uint8_t> target(_graph.nodes, 0);
            if (level == 0) {
                for (uint32_t dst_node : state.dst_nodes) {
                    target[dst_node] = 1;
                }
            } else {
                for (uint32_t target_node : state.waypoint_levels.at(level - 1)) {
                    target[target_node] = 1;
                }
            }

            vector<uint32_t> candidates;
            for (uint32_t neighbor : _graph.adjacency.at(node)) {
                if (target[neighbor]) {
                    candidates.push_back(neighbor);
                }
            }
            state.parents[node] = pick(candidates, _params.spray_h, TAG_PARENT, dst_group, level, node);
        } else if (state.is_inner_ring[node]) {
            vector<uint8_t> target(_graph.nodes, 0);
            for (uint32_t target_node : wp_last) {
                target[target_node] = 1;
            }

            vector<uint32_t> candidates;
            for (uint32_t neighbor : _graph.adjacency.at(node)) {
                if (target[neighbor]) {
                    candidates.push_back(neighbor);
                }
            }
            state.parents[node] = pick(candidates, _params.spray_h, TAG_PARENT_IR, dst_group, 0, node);
        } else {
            if (!state.inner_ring.empty() && dist_to_inner[node] >= 0) {
                state.parents[node] = shortest_next_hops(node, dist_to_inner, _params.spray_h, TAG_PARENT_OR, dst_group);
            }
        }

        if (state.parents[node].empty()) {
            state.parents[node] = shortest_next_hops(node, dist_to_dst, _params.spray_h, TAG_FALLBACK, dst_group);
        }
    }

    return state;
}

vector<uint32_t> OxcOcsSprayPointRouter::pick(
        vector<uint32_t> items,
        uint32_t count,
        uint64_t tag,
        uint32_t dst_group,
        uint32_t level,
        uint32_t node) const {
    items = sorted_unique(items);
    stable_sort(items.begin(), items.end(), [&](uint32_t a, uint32_t b) {
        uint64_t ka = rank_key(a, tag, dst_group, level, node);
        uint64_t kb = rank_key(b, tag, dst_group, level, node);
        if (ka == kb) {
            return a < b;
        }
        return ka < kb;
    });
    if (items.size() > count) {
        items.resize(count);
    }
    sort(items.begin(), items.end());
    return items;
}

vector<uint32_t> OxcOcsSprayPointRouter::shortest_next_hops(
        uint32_t node,
        const vector<int32_t>& dist,
        uint32_t count,
        uint64_t tag,
        uint32_t dst_group) const {
    int32_t best = numeric_limits<int32_t>::max();
    vector<uint32_t> candidates;
    for (uint32_t neighbor : _graph.adjacency.at(node)) {
        if (dist.at(neighbor) < 0) {
            continue;
        }
        if (dist.at(neighbor) < best) {
            best = dist.at(neighbor);
            candidates.clear();
        }
        if (dist.at(neighbor) == best) {
            candidates.push_back(neighbor);
        }
    }
    return pick(candidates, count, tag, dst_group, 0, node);
}

vector<int32_t> OxcOcsSprayPointRouter::multi_source_distances(
        const vector<uint32_t>& sources) const {
    vector<int32_t> dist(_graph.nodes, -1);
    queue<uint32_t> q;
    for (uint32_t source : sources) {
        dist.at(source) = 0;
        q.push(source);
    }

    while (!q.empty()) {
        uint32_t cur = q.front();
        q.pop();
        for (uint32_t neighbor : _graph.adjacency.at(cur)) {
            if (dist.at(neighbor) >= 0) {
                continue;
            }
            dist.at(neighbor) = dist.at(cur) + 1;
            q.push(neighbor);
        }
    }
    return dist;
}

uint64_t OxcOcsSprayPointRouter::rank_key(
        uint32_t item,
        uint64_t tag,
        uint32_t dst_group,
        uint32_t level,
        uint32_t node) const {
    uint64_t h = splitmix64(_params.spray_seed);
    h = splitmix64(h ^ tag);
    h = splitmix64(h ^ dst_group);
    h = splitmix64(h ^ level);
    h = splitmix64(h ^ node);
    h = splitmix64(h ^ item);
    return h;
}

string OxcOcsSprayPointRouter::role_for_node(uint32_t template_node, uint32_t dst_group) const {
    const auto& state = state_for_group(dst_group);
    if (state.is_dst.at(template_node)) {
        return "dst";
    }
    if (state.node_level.at(template_node) >= 0) {
        return "wp" + to_string(state.node_level.at(template_node));
    }
    if (state.is_inner_ring.at(template_node)) {
        return "inner";
    }
    if (state.is_outer_ring.at(template_node)) {
        return "outer";
    }
    return "unassigned";
}

void write_oxc_ocs_spraypoint_state_csv(
        const OxcOcsSprayPointRouter& router,
        const string& path) {
    ofstream out(path);
    if (!out.is_open()) {
        throw runtime_error("failed to open Oxc OCS SprayPoint state output: " + path);
    }

    out << "# groups " << router.groups() << "\n";
    out << "# l1_eps_per_plane " << router.l1_eps_per_plane() << "\n";
    out << "# nodes " << router.graph().nodes << "\n";
    out << "# degree " << router.graph().degree << "\n";
    out << "# spray_p " << router.params().spray_p << "\n";
    out << "# spray_h " << router.params().spray_h << "\n";
    out << "# spray_levels " << router.levels() << "\n";
    out << "# spray_seed " << router.params().spray_seed << "\n";
    out << "dst_group,node,node_group,node_eps,role,parent_count,parents,source_neighbor_count,source_neighbors\n";

    for (uint32_t dst_group = 0; dst_group < router.groups(); dst_group++) {
        const auto& state = router.state_for_group(dst_group);
        for (uint32_t node = 0; node < router.graph().nodes; node++) {
            auto decoded = oxc_ocs_decode_template_node(node, router.l1_eps_per_plane());
            vector<uint32_t> source_neighbors = router.source_spray_next_hops(node);
            out << dst_group << ","
                << node << ","
                << decoded.first << ","
                << decoded.second << ","
                << router.role_for_node(node, dst_group) << ","
                << state.parents.at(node).size() << ","
                << join_nodes(state.parents.at(node)) << ","
                << source_neighbors.size() << ","
                << join_nodes(source_neighbors) << "\n";
        }
    }
}
