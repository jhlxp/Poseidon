// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_ocs_graph.h"
#include "oxc_ocs_ksp.h"

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
         << " [--ocs_expander_seed S] [--k P] [--max-hops H]"
         << " [--ksp-seed S] [--max-paths-per-pair N]"
         << " [--choice flow_hash|packet_hash|packet_rr] [--query-flow-id F]"
         << " [--query-path-entropy P] [--query-packet-id P]"
         << " [--query-rr-counter R] [--out paths.csv]\n"
         << "       Optional query: --query-src-group G --query-src-l1-plane P"
         << " --query-src-l1-eps E --query-dst-group G"
         << " [--query-path-id P] [--query-current-node N]\n";
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

static OxcOcsKspChoice parse_choice(const string& value) {
    if (value == "flow_hash") {
        return OxcOcsKspChoice::FLOW_HASH;
    }
    if (value == "packet_hash") {
        return OxcOcsKspChoice::PACKET_HASH;
    }
    if (value == "packet_rr") {
        return OxcOcsKspChoice::PACKET_RR;
    }
    throw invalid_argument("unknown KSP choice: " + value);
}

int main(int argc, char** argv) {
    uint32_t groups = 0;
    uint32_t l1_planes = 0;
    uint32_t l1_eps_per_l1_plane = 0;
    uint32_t degree = 0;
    uint32_t ocs_seed = 42;
    OxcOcsKspParams params;
    string out_path;

    bool has_query_src_group = false;
    bool has_query_src_l1_plane = false;
    bool has_query_src_l1_eps = false;
    bool has_query_dst_group = false;
    bool has_query_path_id = false;
    bool has_query_choice = false;
    bool has_query_current_node = false;
    uint32_t query_src_group = 0;
    uint32_t query_src_l1_plane = 0;
    uint32_t query_src_l1_eps = 0;
    uint32_t query_dst_group = 0;
    uint32_t query_path_id = 0;
    uint32_t query_current_node = numeric_limits<uint32_t>::max();
    string query_choice_name = "flow_hash";
    OxcOcsKspChoice query_choice = OxcOcsKspChoice::FLOW_HASH;
    uint32_t query_flow_id = 1001;
    uint32_t query_path_entropy = 7;
    uint32_t query_packet_id = 0;
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
            // Coupled mode is the only supported Oxc OCS KSP dump mode.
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
        } else if (arg == "--k" || arg == "--paths") {
            params.k = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--max-hops") {
            string value = need_value(arg);
            params.max_hops = value == "auto" ? 0 : static_cast<uint32_t>(stoul(value));
        } else if (arg == "--ksp-seed") {
            params.seed = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--max-paths-per-pair") {
            params.max_paths_per_pair = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--choice") {
            has_query_choice = true;
            query_choice_name = need_value(arg);
            query_choice = parse_choice(query_choice_name);
        } else if (arg == "--query-flow-id") {
            query_flow_id = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-path-entropy") {
            query_path_entropy = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-packet-id") {
            query_packet_id = static_cast<uint32_t>(stoul(need_value(arg)));
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
        } else if (arg == "--query-path-id") {
            has_query_path_id = true;
            query_path_id = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--query-current-node") {
            has_query_current_node = true;
            query_current_node = static_cast<uint32_t>(stoul(need_value(arg)));
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
        OxcOcsKspRouter router(graph, groups, nodes_per_group, params);

        cout << "Oxc OCS KSP router\n";
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
        cout << "k: " << router.params().k << "\n";
        cout << "max_hops: " << router.effective_max_hops() << "\n";
        cout << "ksp_seed: " << router.params().seed << "\n";
        cout << "max_paths_per_pair: " << router.params().max_paths_per_pair << "\n";
        cout << "total_paths: " << router.total_paths() << "\n";
        cout << "min_path_count: " << router.min_path_count() << "\n";
        cout << "max_path_count: " << router.max_path_count() << "\n";

        uint64_t route_pair_count = 0;
        uint64_t zero_path_pairs = 0;
        uint64_t nonzero_path_pairs = 0;
        uint64_t counted_paths = 0;
        for (uint32_t src_node = 0; src_node < graph.nodes; src_node++) {
            uint32_t src_group = oxc_ocs_decode_template_node(src_node, nodes_per_group).first;
            for (uint32_t dst_group = 0; dst_group < groups; dst_group++) {
                if (dst_group == src_group) {
                    continue;
                }
                route_pair_count++;
                const auto& ps = router.paths(src_node, dst_group);
                counted_paths += ps.size();
                if (ps.empty()) {
                    zero_path_pairs++;
                } else {
                    nonzero_path_pairs++;
                }
            }
        }
        cout << fixed << setprecision(4);
        cout << "route_pair_count: " << route_pair_count << "\n";
        cout << "nonzero_path_pairs: " << nonzero_path_pairs << "\n";
        cout << "zero_path_pairs: " << zero_path_pairs << "\n";
        cout << "avg_paths_per_pair: "
             << (route_pair_count ? static_cast<double>(counted_paths) / route_pair_count : 0.0)
             << "\n";
        cout << "avg_paths_per_src: "
             << (graph.nodes ? static_cast<double>(counted_paths) / graph.nodes : 0.0)
             << "\n";

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
            const auto& ps = router.paths(src, query_dst_group);
            cout << "query src_node: " << src << "\n";
            cout << "query src_group: " << query_src_group << "\n";
            cout << "query src_l1_plane: " << query_src_l1_plane << "\n";
            cout << "query src_l1_eps: " << query_src_l1_eps << "\n";
            cout << "query src_coupled_pair: " << query_src_l1_eps / 2 << "\n";
            cout << "query src_local_logical: "
                 << query_src_l1_plane * (l1_eps_per_l1_plane / 2) + query_src_l1_eps / 2
                 << "\n";
            cout << "query dst_group: " << query_dst_group << "\n";
            cout << "query path_count: " << ps.size() << "\n";
            for (const auto& path : ps) {
                cout << "path " << path.path_id
                     << " hops " << (path.nodes.empty() ? 0 : path.nodes.size() - 1)
                     << " nodes " << join_nodes(path.nodes)
                     << "\n";
            }

            if (!ps.empty()) {
                uint32_t selected_path_id = 0;
                if (has_query_path_id) {
                    selected_path_id = query_path_id;
                } else if (has_query_choice) {
                    selected_path_id = router.choose_path(
                            src, query_dst_group, query_flow_id, query_path_entropy,
                            query_packet_id, query_rr_counter, query_choice);
                }
                uint32_t current_node = has_query_current_node ? query_current_node : src;
                uint32_t next = router.next_hop(src, query_dst_group, selected_path_id, current_node);
                cout << "query choice: " << query_choice_name << "\n";
                cout << "query flow_id: " << query_flow_id << "\n";
                cout << "query path_entropy: " << query_path_entropy << "\n";
                cout << "query packet_id: " << query_packet_id << "\n";
                cout << "query rr_counter: " << query_rr_counter << "\n";
                cout << "query chosen_path_id: " << selected_path_id << "\n";
                cout << "query selected_path_id: " << selected_path_id << "\n";
                cout << "query current_node: " << current_node << "\n";
                if (next == numeric_limits<uint32_t>::max()) {
                    cout << "query next_hop: none\n";
                } else {
                    cout << "query next_hop: " << next << "\n";
                }
            }
        }

        if (!out_path.empty()) {
            write_oxc_ocs_ksp_paths_csv(router, out_path);
            cout << "ksp_paths_csv: " << out_path << "\n";
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
