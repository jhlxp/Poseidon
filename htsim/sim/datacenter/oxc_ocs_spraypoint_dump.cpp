// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_ocs_graph.h"
#include "oxc_ocs_spraypoint.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static void usage(const char* prog) {
    cerr << "Usage: " << prog
         << " --coupled --groups G --l1-planes P --l1-eps-per-l1-plane E --degree K"
         << " [--ocs_expander_seed S] [--spray-p P] [--spray-h H]"
         << " [--spray-levels auto|L] [--spray-seed S]"
         << " [--choice packet_rr|flow_hash] [--query-flow-id F]"
         << " [--query-path-id P] [--query-rr-counter R] [--out state.csv]\n"
         << "       Optional query: --query-src-group G --query-src-l1-plane P"
         << " --query-src-l1-eps E --query-dst-group G\n";
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

static OxcOcsSprayPointChoice parse_choice(const string& value) {
    if (value == "packet_rr") {
        return OxcOcsSprayPointChoice::PACKET_RR;
    }
    if (value == "flow_hash") {
        return OxcOcsSprayPointChoice::FLOW_HASH;
    }
    throw invalid_argument("unknown SprayPoint choice: " + value);
}

int main(int argc, char** argv) {
    uint32_t groups = 0;
    uint32_t l1_planes = 0;
    uint32_t l1_eps_per_l1_plane = 0;
    uint32_t degree = 0;
    uint32_t ocs_seed = 42;
    OxcOcsSprayPointParams params;
    string out_path;

    bool has_query_src_group = false;
    bool has_query_src_l1_plane = false;
    bool has_query_src_l1_eps = false;
    bool has_query_dst_group = false;
    uint32_t query_src_group = 0;
    uint32_t query_src_l1_plane = 0;
    uint32_t query_src_l1_eps = 0;
    uint32_t query_dst_group = 0;
    string query_choice_name = "packet_rr";
    OxcOcsSprayPointChoice query_choice = OxcOcsSprayPointChoice::PACKET_RR;
    uint32_t query_flow_id = 1001;
    uint32_t query_path_id = 7;
    uint32_t query_rr_counter = 0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        auto need_value = [&](const string& name) {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << endl;
                usage(argv[0]);
                exit(1);
            }
            return string(argv[++i]);
        };

        if (arg == "--coupled") {
            // Coupled mode is the only supported Oxc OCS SprayPoint dump mode.
        } else if (arg == "--groups" || arg == "-g") {
            groups = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--l1-planes" || arg == "--l1_planes") {
            l1_planes = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--l1-eps-per-l1-plane" || arg == "--l1_eps_per_l1_plane") {
            l1_eps_per_l1_plane = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--degree" || arg == "-d") {
            degree = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--ocs_expander_seed" || arg == "--seed") {
            ocs_seed = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--spray-p") {
            params.spray_p = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--spray-h") {
            params.spray_h = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--spray-levels") {
            string value = need_value(arg);
            params.spray_levels = value == "auto" ? -1 : static_cast<int32_t>(stol(value));
        } else if (arg == "--spray-seed") {
            params.spray_seed = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--choice") {
            query_choice_name = need_value(arg);
            query_choice = parse_choice(query_choice_name);
        } else if (arg == "--query-flow-id") {
            query_flow_id = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-path-id") {
            query_path_id = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-rr-counter") {
            query_rr_counter = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--out" || arg == "-o") {
            out_path = need_value(arg);
        } else if (arg == "--query-src-group") {
            has_query_src_group = true;
            query_src_group = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-src-l1-plane") {
            has_query_src_l1_plane = true;
            query_src_l1_plane = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-src-l1-eps") {
            has_query_src_l1_eps = true;
            query_src_l1_eps = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-dst-group") {
            has_query_dst_group = true;
            query_dst_group = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            usage(argv[0]);
            return 1;
        }
    }

    if (groups == 0 || l1_planes == 0 || l1_eps_per_l1_plane == 0) {
        cerr << "Missing required topology arguments" << endl;
        usage(argv[0]);
        return 1;
    }

    try {
        uint32_t nodes_per_group =
                oxc_ocs_coupled_logical_nodes_per_group(l1_planes, l1_eps_per_l1_plane);
        OxcOcsGraph graph = build_oxc_ocs_coupled_template(
                groups, l1_planes, l1_eps_per_l1_plane, degree, ocs_seed);
        OxcOcsSprayPointRouter router(graph, groups, nodes_per_group, params);

        cout << "Oxc OCS SprayPoint router\n";
        cout << "coupled: true\n";
        cout << "groups: " << groups << "\n";
        cout << "l1_planes: " << l1_planes << "\n";
        cout << "l1_eps_per_l1_plane: " << l1_eps_per_l1_plane << "\n";
        cout << "logical_nodes_per_group: " << nodes_per_group << "\n";
        cout << "logical_graph_ok: "
             << (oxc_ocs_coupled_template_has_logical_nodes(
                         graph, groups, l1_planes, l1_eps_per_l1_plane) ? "true" : "false")
             << "\n";
        cout << "nodes: " << graph.nodes << "\n";
        cout << "degree: " << graph.degree << "\n";
        cout << "ocs_expander_seed: " << graph.seed << "\n";
        cout << "spray_p: " << router.params().spray_p << "\n";
        cout << "spray_h: " << router.params().spray_h << "\n";
        cout << "spray_levels: " << router.levels() << "\n";
        cout << "spray_seed: " << router.params().spray_seed << "\n";

        uint64_t route_pair_count = 0;
        uint64_t source_spray_candidates = 0;
        uint64_t pointing_candidates_at_src = 0;
        uint32_t min_source_spray_candidates = numeric_limits<uint32_t>::max();
        uint32_t max_source_spray_candidates = 0;
        uint32_t min_pointing_candidates_at_src = numeric_limits<uint32_t>::max();
        uint32_t max_pointing_candidates_at_src = 0;
        for (uint32_t src_node = 0; src_node < graph.nodes; src_node++) {
            uint32_t src_group = oxc_ocs_decode_template_node(src_node, nodes_per_group).first;
            uint32_t source_count = router.source_spray_next_hops(src_node).size();
            for (uint32_t dst_group = 0; dst_group < groups; dst_group++) {
                if (dst_group == src_group) {
                    continue;
                }
                route_pair_count++;
                source_spray_candidates += source_count;
                min_source_spray_candidates = min(min_source_spray_candidates, source_count);
                max_source_spray_candidates = max(max_source_spray_candidates, source_count);

                uint32_t pointing_count = router.pointing_next_hops(src_node, dst_group).size();
                pointing_candidates_at_src += pointing_count;
                min_pointing_candidates_at_src = min(min_pointing_candidates_at_src, pointing_count);
                max_pointing_candidates_at_src = max(max_pointing_candidates_at_src, pointing_count);
            }
        }
        if (route_pair_count == 0) {
            min_source_spray_candidates = 0;
            min_pointing_candidates_at_src = 0;
        }
        cout << fixed << setprecision(4);
        cout << "path_count_mode: implicit_next_hop_ecmp_not_enumerated\n";
        cout << "route_pair_count: " << route_pair_count << "\n";
        cout << "source_spray_candidates_min: " << min_source_spray_candidates << "\n";
        cout << "source_spray_candidates_max: " << max_source_spray_candidates << "\n";
        cout << "avg_source_spray_candidates_per_pair: "
             << (route_pair_count ? static_cast<double>(source_spray_candidates) / route_pair_count : 0.0)
             << "\n";
        cout << "avg_unique_source_spray_candidates_per_src: "
             << (graph.nodes ? static_cast<double>(source_spray_candidates)
                                / graph.nodes / max<uint32_t>(1, groups - 1) : 0.0)
             << "\n";
        cout << "pointing_candidates_at_src_min: " << min_pointing_candidates_at_src << "\n";
        cout << "pointing_candidates_at_src_max: " << max_pointing_candidates_at_src << "\n";
        cout << "avg_pointing_candidates_at_src_per_pair: "
             << (route_pair_count ? static_cast<double>(pointing_candidates_at_src) / route_pair_count : 0.0)
             << "\n";

        for (uint32_t dst_group = 0; dst_group < groups; dst_group++) {
            const auto& state = router.state_for_group(dst_group);
            uint32_t parent_nodes = 0;
            uint32_t parent_edges = 0;
            for (const auto& parents : state.parents) {
                if (!parents.empty()) {
                    parent_nodes++;
                    parent_edges += parents.size();
                }
            }

            cout << "dst_group " << dst_group
                 << " dst_nodes " << join_nodes(state.dst_nodes)
                 << " waypoint_levels";
            for (const auto& level : state.waypoint_levels) {
                cout << " " << level.size();
            }
            cout << " inner " << state.inner_ring.size()
                 << " outer " << state.outer_ring.size()
                 << " parent_nodes " << parent_nodes
                 << " parent_edges " << parent_edges
                 << "\n";
        }

        if (has_query_src_group || has_query_src_l1_plane
            || has_query_src_l1_eps || has_query_dst_group) {
            bool has_query = has_query_src_group && has_query_src_l1_plane
                             && has_query_src_l1_eps && has_query_dst_group;
            if (!has_query) {
                cerr << "Query requires --query-src-group, --query-src-l1-plane, --query-src-l1-eps, and --query-dst-group" << endl;
                return 1;
            }
            uint32_t src = oxc_ocs_coupled_logical_node_id(
                    query_src_group, query_src_l1_plane, query_src_l1_eps / 2,
                    l1_planes, l1_eps_per_l1_plane);
            cout << "query src_node: " << src << "\n";
            cout << "query src_group: " << query_src_group << "\n";
            cout << "query src_l1_plane: " << query_src_l1_plane << "\n";
            cout << "query src_l1_eps: " << query_src_l1_eps << "\n";
            cout << "query src_coupled_pair: " << query_src_l1_eps / 2 << "\n";
            cout << "query src_local_logical: "
                 << query_src_l1_plane * (l1_eps_per_l1_plane / 2) + query_src_l1_eps / 2
                 << "\n";
            cout << "query dst_group: " << query_dst_group << "\n";
            cout << "query role: " << router.role_for_node(src, query_dst_group) << "\n";
            cout << "source_spray_next_hops: " << join_nodes(router.source_spray_next_hops(src)) << "\n";
            cout << "pointing_next_hops: " << join_nodes(router.pointing_next_hops(src, query_dst_group)) << "\n";
            cout << "query choice: " << query_choice_name << "\n";
            cout << "query flow_id: " << query_flow_id << "\n";
            cout << "query path_id: " << query_path_id << "\n";
            cout << "query rr_counter: " << query_rr_counter << "\n";
            uint32_t source_next = router.choose_next_hop(
                    src, query_dst_group, true, query_flow_id, query_path_id, query_rr_counter,
                    query_choice);
            cout << "choose_source_" << query_choice_name << ": "
                 << (source_next == numeric_limits<uint32_t>::max() ? string("none") : to_string(source_next))
                 << "\n";
            uint32_t pointing_next = router.choose_next_hop(
                    src, query_dst_group, false, query_flow_id, query_path_id, query_rr_counter,
                    query_choice);
            cout << "choose_pointing_" << query_choice_name << ": "
                 << (pointing_next == numeric_limits<uint32_t>::max() ? string("none") : to_string(pointing_next))
                 << "\n";
        }

        if (!out_path.empty()) {
            write_oxc_ocs_spraypoint_state_csv(router, out_path);
            cout << "spraypoint_state_csv: " << out_path << "\n";
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
