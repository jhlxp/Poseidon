// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_ocs_graph.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

using namespace std;

static void usage(const char* prog) {
    cerr << "Usage: " << prog
         << " --nodes N --degree K [--ocs_expander_seed S] [--out edges.csv]\n"
         << "       " << prog
         << " --coupled --groups G --l1-planes P --l1-eps-per-l1-plane E --degree K"
         << " [--ocs_expander_seed S] [--out coupled.csv]\n";
}

static void print_degree_summary(const OxcOcsGraph& graph) {
    auto degrees = graph.degrees();
    uint32_t min_degree = degrees.empty() ? 0 : degrees[0];
    uint32_t max_degree = degrees.empty() ? 0 : degrees[0];
    uint64_t sum_degree = 0;
    for (uint32_t d : degrees) {
        min_degree = min(min_degree, d);
        max_degree = max(max_degree, d);
        sum_degree += d;
    }

    cout << "nodes: " << graph.nodes << "\n";
    cout << "degree: " << graph.degree << "\n";
    cout << "ocs_expander_seed: " << graph.seed << "\n";
    cout << "edges: " << graph.edges.size() << "\n";
    cout << "min_degree: " << min_degree << "\n";
    cout << "max_degree: " << max_degree << "\n";
    cout << "avg_degree: "
         << (graph.nodes == 0 ? 0.0 : static_cast<double>(sum_degree) / graph.nodes)
         << "\n";
    cout << "degree_ok: "
         << (min_degree == graph.degree && max_degree == graph.degree ? "true" : "false")
         << "\n";
    cout << "connected: " << (graph.is_connected() ? "true" : "false") << "\n";
}

int main(int argc, char** argv) {
    uint32_t nodes = 0;
    uint32_t groups = 0;
    uint32_t l1_planes = 0;
    uint32_t l1_eps_per_l1_plane = 0;
    uint32_t degree = 0;
    uint32_t seed = 42;
    bool coupled_mode = false;
    string out_path;

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

        if (arg == "--nodes" || arg == "-n") {
            nodes = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--coupled") {
            coupled_mode = true;
        } else if (arg == "--groups" || arg == "-g") {
            groups = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--l1-planes" || arg == "--l1_planes") {
            l1_planes = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--l1-eps-per-l1-plane" || arg == "--l1_eps_per_l1_plane") {
            l1_eps_per_l1_plane = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--degree" || arg == "-k") {
            degree = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--ocs_expander_seed" || arg == "--seed") {
            seed = static_cast<uint32_t>(stoul(need_value(arg)));
        } else if (arg == "--out" || arg == "-o") {
            out_path = need_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            cerr << "Unknown argument: " << arg << endl;
            usage(argv[0]);
            return 1;
        }
    }

    try {
        if (coupled_mode) {
            if (groups == 0 || l1_planes == 0 || l1_eps_per_l1_plane == 0) {
                cerr << "--coupled requires --groups, --l1-planes, and --l1-eps-per-l1-plane" << endl;
                usage(argv[0]);
                return 1;
            }

            OxcOcsGraph graph = build_oxc_ocs_coupled_template(
                    groups, l1_planes, l1_eps_per_l1_plane, degree, seed);

            cout << "Oxc OCS coupled graph\n";
            cout << "groups: " << groups << "\n";
            cout << "l1_planes: " << l1_planes << "\n";
            cout << "l1_eps_per_l1_plane: " << l1_eps_per_l1_plane << "\n";
            cout << "logical_nodes_per_group: "
                 << oxc_ocs_coupled_logical_nodes_per_group(l1_planes, l1_eps_per_l1_plane)
                 << "\n";
            cout << "logical_node_mapping: node = group * logical_nodes_per_group"
                 << " + l1_plane * (l1_eps_per_l1_plane / 2) + coupled_pair\n";
            cout << "coupled_pair_members: l1_eps = coupled_pair * 2 and coupled_pair * 2 + 1\n";
            print_degree_summary(graph);
            cout << "logical_graph_ok: "
                 << (oxc_ocs_coupled_template_has_logical_nodes(
                             graph, groups, l1_planes, l1_eps_per_l1_plane) ? "true" : "false")
                 << "\n";

            if (!out_path.empty()) {
                write_oxc_ocs_coupled_edges_csv(
                        graph, groups, l1_planes, l1_eps_per_l1_plane, out_path);
                cout << "coupled_edges_csv: " << out_path << "\n";
            }
            return 0;
        }

        if (nodes == 0) {
            cerr << "--nodes must be positive unless --coupled is used" << endl;
            usage(argv[0]);
            return 1;
        }

        OxcOcsGraph graph = build_oxc_ocs_expander(nodes, degree, seed);
        cout << "Oxc OCS generic expander graph\n";
        print_degree_summary(graph);

        if (!out_path.empty()) {
            write_oxc_ocs_edges_csv(graph, out_path);
            cout << "edges_csv: " << out_path << "\n";
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
