// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_ocs_graph.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

using namespace std;

vector<uint32_t> OxcOcsGraph::degrees() const {
    vector<uint32_t> result(nodes, 0);
    for (const auto& edge : edges) {
        result.at(edge.first)++;
        result.at(edge.second)++;
    }
    return result;
}

bool OxcOcsGraph::is_connected() const {
    if (nodes == 0) {
        return true;
    }

    vector<bool> seen(nodes, false);
    queue<uint32_t> q;
    seen[0] = true;
    q.push(0);

    while (!q.empty()) {
        uint32_t cur = q.front();
        q.pop();
        for (uint32_t next : adjacency.at(cur)) {
            if (!seen.at(next)) {
                seen[next] = true;
                q.push(next);
            }
        }
    }

    return all_of(seen.begin(), seen.end(), [](bool v) { return v; });
}

static void validate_regular_graph_params(uint32_t nodes, uint32_t degree) {
    if (nodes == 0) {
        throw invalid_argument("OCS expander nodes must be positive");
    }
    if (degree >= nodes) {
        throw invalid_argument("OCS expander degree must be in [0, nodes-1]");
    }
    if ((static_cast<uint64_t>(nodes) * degree) % 2 != 0) {
        throw invalid_argument("OCS expander nodes * degree must be even");
    }
}

static bool oxc_ocs_template_has_no_intra_group(
        const OxcOcsGraph& graph,
        uint32_t l1_eps_per_plane);

uint32_t oxc_ocs_template_node_id(uint32_t group, uint32_t eps, uint32_t l1_eps_per_plane) {
    return group * l1_eps_per_plane + eps;
}

pair<uint32_t, uint32_t> oxc_ocs_decode_template_node(
        uint32_t node,
        uint32_t l1_eps_per_plane) {
    if (l1_eps_per_plane == 0) {
        throw invalid_argument("l1_eps_per_plane must be positive");
    }
    return {node / l1_eps_per_plane, node % l1_eps_per_plane};
}

uint32_t oxc_ocs_coupled_endpoint_id(
        uint32_t group,
        uint32_t l1_plane,
        uint32_t l1_eps,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    if (l1_planes == 0) {
        throw invalid_argument("l1_planes must be positive");
    }
    if (l1_eps_per_l1_plane == 0) {
        throw invalid_argument("l1_eps_per_l1_plane must be positive");
    }
    if (l1_plane >= l1_planes) {
        throw invalid_argument("l1_plane is out of range");
    }
    if (l1_eps >= l1_eps_per_l1_plane) {
        throw invalid_argument("l1_eps is out of range");
    }

    uint64_t id = (static_cast<uint64_t>(group) * l1_planes + l1_plane)
                  * l1_eps_per_l1_plane + l1_eps;
    if (id > numeric_limits<uint32_t>::max()) {
        throw invalid_argument("Oxc OCS coupled endpoint id overflows uint32_t");
    }
    return static_cast<uint32_t>(id);
}

OxcOcsEndpoint oxc_ocs_decode_coupled_endpoint(
        uint32_t node,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    if (l1_planes == 0) {
        throw invalid_argument("l1_planes must be positive");
    }
    if (l1_eps_per_l1_plane == 0) {
        throw invalid_argument("l1_eps_per_l1_plane must be positive");
    }

    OxcOcsEndpoint endpoint;
    uint32_t endpoints_per_group = l1_planes * l1_eps_per_l1_plane;
    endpoint.group = node / endpoints_per_group;
    endpoint.endpoint_index = node % endpoints_per_group;
    endpoint.l1_plane = endpoint.endpoint_index / l1_eps_per_l1_plane;
    endpoint.l1_eps = endpoint.endpoint_index % l1_eps_per_l1_plane;
    endpoint.coupled_pair = endpoint.l1_eps / 2;
    endpoint.coupled_member = endpoint.l1_eps % 2;
    return endpoint;
}

uint32_t oxc_ocs_coupled_logical_nodes_per_group(
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    if (l1_planes == 0) {
        throw invalid_argument("l1_planes must be positive");
    }
    if (l1_eps_per_l1_plane == 0) {
        throw invalid_argument("l1_eps_per_l1_plane must be positive");
    }
    if (l1_eps_per_l1_plane % 2 != 0) {
        throw invalid_argument("l1_eps_per_l1_plane must be even");
    }
    return l1_planes * (l1_eps_per_l1_plane / 2);
}

uint32_t oxc_ocs_coupled_logical_node_id(
        uint32_t group,
        uint32_t l1_plane,
        uint32_t coupled_pair,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    uint32_t pairs_per_plane = l1_eps_per_l1_plane / 2;
    if (l1_plane >= l1_planes) {
        throw invalid_argument("l1_plane is out of range");
    }
    if (coupled_pair >= pairs_per_plane) {
        throw invalid_argument("coupled_pair is out of range");
    }

    uint64_t id = static_cast<uint64_t>(group)
                  * oxc_ocs_coupled_logical_nodes_per_group(l1_planes, l1_eps_per_l1_plane)
                  + l1_plane * pairs_per_plane
                  + coupled_pair;
    if (id > numeric_limits<uint32_t>::max()) {
        throw invalid_argument("Oxc OCS coupled logical node id overflows uint32_t");
    }
    return static_cast<uint32_t>(id);
}

OxcOcsLogicalNode oxc_ocs_decode_coupled_logical_node(
        uint32_t node,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    uint32_t pairs_per_plane = l1_eps_per_l1_plane / 2;
    uint32_t logical_per_group = oxc_ocs_coupled_logical_nodes_per_group(
            l1_planes, l1_eps_per_l1_plane);

    OxcOcsLogicalNode logical;
    logical.group = node / logical_per_group;
    logical.local_index = node % logical_per_group;
    logical.l1_plane = logical.local_index / pairs_per_plane;
    logical.coupled_pair = logical.local_index % pairs_per_plane;
    logical.l1_eps_member0 = logical.coupled_pair * 2;
    logical.l1_eps_member1 = logical.coupled_pair * 2 + 1;
    return logical;
}

static void validate_coupled_graph_params(
        uint32_t groups,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane,
        uint32_t degree) {
    if (groups < 2) {
        throw invalid_argument("Oxc coupled OCS groups must be at least 2");
    }
    if (l1_planes == 0) {
        throw invalid_argument("Oxc coupled OCS l1_planes must be positive");
    }
    if (l1_eps_per_l1_plane == 0) {
        throw invalid_argument("Oxc coupled OCS l1_eps_per_l1_plane must be positive");
    }
    if (l1_eps_per_l1_plane % 2 != 0) {
        throw invalid_argument("Oxc coupled OCS l1_eps_per_l1_plane must be even");
    }

    uint32_t logical_per_group = oxc_ocs_coupled_logical_nodes_per_group(
            l1_planes, l1_eps_per_l1_plane);
    uint64_t nodes = static_cast<uint64_t>(groups) * logical_per_group;
    if (nodes > numeric_limits<uint32_t>::max()) {
        throw invalid_argument("Oxc coupled OCS logical node count overflows uint32_t");
    }

    uint64_t max_cross_degree = static_cast<uint64_t>(groups - 1) * logical_per_group;
    if (degree > max_cross_degree) {
        throw invalid_argument("Oxc coupled OCS degree exceeds available cross-group logical peers");
    }
    if ((nodes * degree) % 2 != 0) {
        throw invalid_argument("Oxc coupled OCS logical nodes * degree must be even");
    }
}

static bool random_group_matching(
        vector<pair<uint32_t, uint32_t>>& matching,
        const vector<uint32_t>& nodes,
        uint32_t nodes_per_group,
        const set<pair<uint32_t, uint32_t>>& used_edges,
        mt19937& rng) {
    vector<uint32_t> shuffled = nodes;
    shuffle(shuffled.begin(), shuffled.end(), rng);

    vector<pair<uint32_t, uint32_t>> candidate;
    candidate.reserve(shuffled.size() / 2);
    for (size_t i = 0; i < shuffled.size(); i += 2) {
        uint32_t u = shuffled[i];
        uint32_t v = shuffled[i + 1];
        if (oxc_ocs_decode_template_node(u, nodes_per_group).first
            == oxc_ocs_decode_template_node(v, nodes_per_group).first) {
            return false;
        }
        if (u > v) {
            swap(u, v);
        }
        if (used_edges.count({u, v})) {
            return false;
        }
        candidate.emplace_back(u, v);
    }

    matching.swap(candidate);
    return true;
}

static vector<pair<uint32_t, uint32_t>> fallback_group_matching(
        uint32_t groups,
        uint32_t nodes_per_group,
        uint32_t ocs) {
    if (groups % 2 != 0) {
        throw runtime_error("Oxc OCS fallback matching requires an even number of groups");
    }

    vector<uint32_t> group_order(groups);
    for (uint32_t g = 0; g < groups; g++) {
        group_order[g] = g;
    }

    uint32_t group_rounds = groups - 1;
    uint32_t group_round = group_rounds == 0 ? 0 : ocs % group_rounds;
    for (uint32_t r = 0; r < group_round; r++) {
        uint32_t tail = group_order.back();
        for (uint32_t i = groups - 1; i > 1; i--) {
            group_order[i] = group_order[i - 1];
        }
        group_order[1] = tail;
    }

    uint32_t local_shift = nodes_per_group == 0 ? 0 : (ocs / max<uint32_t>(1, group_rounds)) % nodes_per_group;
    vector<pair<uint32_t, uint32_t>> matching;
    matching.reserve(static_cast<size_t>(groups / 2) * nodes_per_group);
    for (uint32_t i = 0; i < groups / 2; i++) {
        uint32_t g1 = group_order[i];
        uint32_t g2 = group_order[groups - 1 - i];
        for (uint32_t local = 0; local < nodes_per_group; local++) {
            uint32_t u = oxc_ocs_template_node_id(g1, local, nodes_per_group);
            uint32_t v = oxc_ocs_template_node_id(
                    g2, (local + local_shift) % nodes_per_group, nodes_per_group);
            if (u > v) {
                swap(u, v);
            }
            matching.emplace_back(u, v);
        }
    }
    return matching;
}

static void add_ocs_matching(
        OxcOcsGraph& graph,
        uint32_t ocs,
        const vector<pair<uint32_t, uint32_t>>& matching,
        set<pair<uint32_t, uint32_t>>& used_edges) {
    for (auto edge : matching) {
        if (edge.first > edge.second) {
            swap(edge.first, edge.second);
        }
        if (!used_edges.insert(edge).second) {
            throw runtime_error("Oxc OCS matching reused an existing logical edge");
        }
        graph.edges.push_back(edge);
        graph.edge_ocs.push_back(ocs);
        graph.adjacency.at(edge.first).push_back(edge.second);
        graph.adjacency.at(edge.second).push_back(edge.first);
    }
}

static OxcOcsGraph complete_graph(uint32_t nodes, uint32_t seed) {
    OxcOcsGraph graph;
    graph.nodes = nodes;
    graph.degree = nodes - 1;
    graph.seed = seed;
    graph.adjacency.assign(nodes, {});

    for (uint32_t u = 0; u < nodes; u++) {
        for (uint32_t v = u + 1; v < nodes; v++) {
            graph.edges.emplace_back(u, v);
            graph.adjacency[u].push_back(v);
            graph.adjacency[v].push_back(u);
        }
    }
    return graph;
}

OxcOcsGraph build_oxc_ocs_expander(
        uint32_t nodes,
        uint32_t degree,
        uint32_t ocs_expander_seed,
        uint32_t max_attempts) {
    validate_regular_graph_params(nodes, degree);

    OxcOcsGraph graph;
    graph.nodes = nodes;
    graph.degree = degree;
    graph.seed = ocs_expander_seed;
    graph.adjacency.assign(nodes, {});

    if (degree == 0) {
        return graph;
    }
    if (degree == nodes - 1) {
        return complete_graph(nodes, ocs_expander_seed);
    }

    vector<uint32_t> stubs;
    stubs.reserve(static_cast<size_t>(nodes) * degree);
    for (uint32_t node = 0; node < nodes; node++) {
        for (uint32_t d = 0; d < degree; d++) {
            stubs.push_back(node);
        }
    }

    mt19937 rng(ocs_expander_seed);
    for (uint32_t attempt = 0; attempt < max_attempts; attempt++) {
        vector<uint32_t> shuffled = stubs;
        shuffle(shuffled.begin(), shuffled.end(), rng);

        set<pair<uint32_t, uint32_t>> edges;
        bool ok = true;
        for (size_t i = 0; i < shuffled.size(); i += 2) {
            uint32_t u = shuffled[i];
            uint32_t v = shuffled[i + 1];
            if (u == v) {
                ok = false;
                break;
            }
            if (u > v) {
                swap(u, v);
            }
            auto inserted = edges.insert({u, v});
            if (!inserted.second) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        OxcOcsGraph candidate;
        candidate.nodes = nodes;
        candidate.degree = degree;
        candidate.seed = ocs_expander_seed;
        candidate.edges.assign(edges.begin(), edges.end());
        candidate.adjacency.assign(nodes, {});
        for (const auto& edge : candidate.edges) {
            candidate.adjacency[edge.first].push_back(edge.second);
            candidate.adjacency[edge.second].push_back(edge.first);
        }

        vector<uint32_t> degrees = candidate.degrees();
        bool degree_ok = all_of(degrees.begin(), degrees.end(),
                                [degree](uint32_t d) { return d == degree; });
        if (degree_ok) {
            return candidate;
        }
    }

    throw runtime_error("failed to generate OCS random regular graph after "
                        + to_string(max_attempts) + " attempts");
}

OxcOcsGraph build_oxc_ocs_coupled_template(
        uint32_t groups,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane,
        uint32_t degree,
        uint32_t ocs_expander_seed,
        uint32_t max_attempts) {
    validate_coupled_graph_params(groups, l1_planes, l1_eps_per_l1_plane, degree);

    uint32_t logical_per_group = oxc_ocs_coupled_logical_nodes_per_group(
            l1_planes, l1_eps_per_l1_plane);
    uint32_t nodes = groups * logical_per_group;

    OxcOcsGraph graph;
    graph.nodes = nodes;
    graph.degree = degree;
    graph.seed = ocs_expander_seed;
    graph.adjacency.assign(nodes, {});

    if (degree == 0) {
        return graph;
    }

    vector<uint32_t> logical_nodes(nodes);
    for (uint32_t node = 0; node < nodes; node++) {
        logical_nodes[node] = node;
    }

    set<pair<uint32_t, uint32_t>> used_edges;
    mt19937 rng(ocs_expander_seed);
    bool deterministic_fallback = false;
    for (uint32_t ocs = 0; ocs < degree; ocs++) {
        vector<pair<uint32_t, uint32_t>> matching;
        bool ok = false;
        for (uint32_t attempt = 0; attempt < max_attempts; attempt++) {
            if (random_group_matching(
                        matching, logical_nodes, logical_per_group, used_edges, rng)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            deterministic_fallback = true;
            break;
        }
        add_ocs_matching(graph, ocs, matching, used_edges);
    }
    if (deterministic_fallback) {
        graph.edges.clear();
        graph.edge_ocs.clear();
        graph.adjacency.assign(nodes, {});
        used_edges.clear();
        for (uint32_t ocs = 0; ocs < degree; ocs++) {
            vector<pair<uint32_t, uint32_t>> matching = fallback_group_matching(
                    groups, logical_per_group, ocs);
            add_ocs_matching(graph, ocs, matching, used_edges);
        }
    }

    if (!oxc_ocs_coupled_template_has_logical_nodes(
                graph, groups, l1_planes, l1_eps_per_l1_plane)) {
        throw runtime_error("coupled OCS graph failed logical-node sanity check");
    }
    return graph;
}

static bool oxc_ocs_template_has_no_intra_group(
        const OxcOcsGraph& graph,
        uint32_t l1_eps_per_plane) {
    for (const auto& edge : graph.edges) {
        auto src = oxc_ocs_decode_template_node(edge.first, l1_eps_per_plane);
        auto dst = oxc_ocs_decode_template_node(edge.second, l1_eps_per_plane);
        if (src.first == dst.first) {
            return false;
        }
    }
    return true;
}

bool oxc_ocs_coupled_template_has_pair_copy(
        const OxcOcsGraph& graph,
        uint32_t groups,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    return oxc_ocs_coupled_template_has_logical_nodes(
            graph, groups, l1_planes, l1_eps_per_l1_plane);
}

bool oxc_ocs_coupled_template_has_logical_nodes(
        const OxcOcsGraph& graph,
        uint32_t groups,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane) {
    if (groups < 2 || l1_planes == 0 || l1_eps_per_l1_plane == 0
        || l1_eps_per_l1_plane % 2 != 0) {
        return false;
    }

    uint32_t logical_per_group = oxc_ocs_coupled_logical_nodes_per_group(
            l1_planes, l1_eps_per_l1_plane);
    if (graph.nodes != groups * logical_per_group) {
        return false;
    }
    if (!oxc_ocs_template_has_no_intra_group(graph, logical_per_group)) {
        return false;
    }

    vector<uint32_t> degrees = graph.degrees();
    return all_of(degrees.begin(), degrees.end(),
                  [&](uint32_t d) { return d == graph.degree; });
}

void write_oxc_ocs_edges_csv(const OxcOcsGraph& graph, const string& path) {
    ofstream out(path);
    if (!out.is_open()) {
        throw runtime_error("failed to open OCS edge output: " + path);
    }

    out << "# nodes " << graph.nodes << "\n";
    out << "# degree " << graph.degree << "\n";
    out << "# ocs_expander_seed " << graph.seed << "\n";
    out << "src_l1,dst_l1\n";
    for (const auto& edge : graph.edges) {
        out << edge.first << "," << edge.second << "\n";
    }
}

void write_oxc_ocs_coupled_edges_csv(
        const OxcOcsGraph& graph,
        uint32_t groups,
        uint32_t l1_planes,
        uint32_t l1_eps_per_l1_plane,
        const string& path) {
    ofstream out(path);
    if (!out.is_open()) {
        throw runtime_error("failed to open Oxc OCS coupled edge output: " + path);
    }

    out << "# groups " << groups << "\n";
    out << "# l1_planes " << l1_planes << "\n";
    out << "# l1_eps_per_l1_plane " << l1_eps_per_l1_plane << "\n";
    out << "# logical_nodes_per_group "
        << oxc_ocs_coupled_logical_nodes_per_group(l1_planes, l1_eps_per_l1_plane) << "\n";
    out << "# nodes " << graph.nodes << "\n";
    out << "# degree " << graph.degree << "\n";
    out << "# ocs_expander_seed " << graph.seed << "\n";
    out << "edge,ocs,src_node,dst_node,"
        << "src_group,src_l1_plane,src_coupled_pair,src_l1_eps_member0,src_l1_eps_member1,"
        << "dst_group,dst_l1_plane,dst_coupled_pair,dst_l1_eps_member0,dst_l1_eps_member1\n";

    for (size_t i = 0; i < graph.edges.size(); i++) {
        uint32_t ocs = graph.edge_ocs.empty() ? 0 : graph.edge_ocs.at(i);
        auto src = oxc_ocs_decode_coupled_logical_node(
                graph.edges[i].first, l1_planes, l1_eps_per_l1_plane);
        auto dst = oxc_ocs_decode_coupled_logical_node(
                graph.edges[i].second, l1_planes, l1_eps_per_l1_plane);
        out << i << ","
            << ocs << ","
            << graph.edges[i].first << ","
            << graph.edges[i].second << ","
            << src.group << ","
            << src.l1_plane << ","
            << src.coupled_pair << ","
            << src.l1_eps_member0 << ","
            << src.l1_eps_member1 << ","
            << dst.group << ","
            << dst.l1_plane << ","
            << dst.coupled_pair << ","
            << dst.l1_eps_member0 << ","
            << dst.l1_eps_member1 << "\n";
    }
}
