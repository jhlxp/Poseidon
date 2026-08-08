// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
//#include "config.h"
#include <cassert>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string.h>

#include <math.h>
#include <unistd.h>
#include "network.h"
#include "pipe.h"
#include "queue.h"
#include "eventlist.h"
#include "logfile.h"
#include "uec_logger.h"
#include "clock.h"
#include "uec_base.h"
#include "uec.h"
#include "uec_mp.h"
#include "uec_pdcses.h"
#include "compositequeue.h"
#include "topology.h"
#include "connection_matrix.h"
#include "pciemodel.h"
#include "oversubscribed_cc.h"


#include "logsim-interface.h"
#include "fat_tree_topology.h"
#include "fat_tree_switch.h"
#include "mprail_switch.h"
#include "mprail_topology.h"
#include "oxc_dag_manager.h"
#include "oxc_topology.h"

#include <algorithm>
#include <limits>
#include <list>
#include <set>
#include <stdexcept>
#include <vector>

// Simulation params

//#define PRINTPATHS 1

#include "main.h"

int DEFAULT_NODES = 128;
uint32_t DEFAULT_TRIMMING_QUEUESIZE_FACTOR = 1;
uint32_t DEFAULT_NONTRIMMING_QUEUESIZE_FACTOR = 5;
// #define DEFAULT_CWND 50

EventList eventlist;

uint32_t parse_positive_u32_arg(const char* option, const char* value) {
    try {
        size_t consumed = 0;
        const unsigned long long parsed = stoull(value, &consumed);
        if (consumed != strlen(value) || parsed == 0
                || parsed > numeric_limits<uint32_t>::max()) {
            throw invalid_argument("outside uint32 range");
        }
        return static_cast<uint32_t>(parsed);
    } catch (const exception&) {
        cerr << option << " requires a positive integer, got " << value << endl;
        exit(1);
    }
}

void exit_error(char* progr) {
    cout << "Usage " << progr << " [-nodes N]\n\t[-cwnd cwnd_size]\n\t[-q queue_size]\n\t[-queue_type composite|random|lossless|lossless_input|]\n\t[-tm traffic_matrix_file]\n\t[-dag task_file]\n\t[-topology mprail]\n\t[-strat route_strategy (single,rand,perm,pull,ecmp,\n\tecmp_host path_count,ecmp_ar,ecmp_rr,\n\tecmp_host_ar ar_thresh)]\n\t[-log log_level]\n\t[-seed random_seed]\n\t[-end end_time_in_usec]\n\t[-mtu MTU]\n\t[-hop_latency x] per hop wire latency in us,default 1\n\t[-target_q_delay x] target_queuing_delay in us, default is 6us \n\t[-switch_latency x] switching latency in us, default 0\n\t[-host_queue_type  swift|prio|fair_prio]\n\t[-logtime dt] sample time for sinklogger, etc\n\t[-conn_reuse] enable connection reuse" << endl;
    cout << "\t[-local_tray_size N] rank group size for direct intra-server routing\n"
         << "\t[-local_linkspeed Mbps] direct intra-server link/injection speed, MpRail default 7200000\n"
         << "\t[-local_latency_ns ns] direct intra-server one-way latency, default 200ns\n"
         << "\t[-mprail_planes N] independent EPS planes, default 8\n"
         << "\t[-mprail_gpus_per_server N] ranks in a FullMesh server, default 8\n"
         << "\t[-mprail_l1_eps_per_plane N] L1 EPS switches per plane, default 1\n"
         << "\t[-mprail_l0_l1_links_per_spine N] parallel links per L0/L1 pair, default 1"
         << endl;
    exit(1);
}

Route* make_direct_local_route(
        int src,
        int dst,
        uint32_t port,
        PacketSink* final_sink,
        linkspeed_bps linkspeed,
        mem_b queuesize,
        simtime_picosec latency,
        const string& direction) {
    Route* route = new Route();

    Queue* queue = new Queue(linkspeed, queuesize, eventlist, nullptr);
    queue->forceName("LOCAL_" + direction + "_Q_SRC_" + ntoa(src)
                     + "_DST_" + ntoa(dst) + "_P_" + ntoa(port));

    Pipe* pipe = new Pipe(latency, eventlist);
    pipe->forceName("LOCAL_" + direction + "_P_SRC_" + ntoa(src)
                    + "_DST_" + ntoa(dst) + "_P_" + ntoa(port));

    route->push_back(queue);
    route->push_back(pipe);
    route->push_back(final_sink);
    return route;
}

enum class OxcOcsDataMode {
    OFF,
    SPRAYPOINT,
    KSP,
};

enum class OxcOcsDataChoice {
    FLOW_HASH,
    PACKET_RR,
};

struct OxcOcsDataConfig {
    OxcOcsDataMode mode = OxcOcsDataMode::OFF;
    OxcOcsDataChoice choice = OxcOcsDataChoice::PACKET_RR;
    uint32_t groups = 0;
    uint32_t ranks_per_group = 0;
    uint32_t ranks_per_tray = 0;
    uint32_t l1_planes = 1;
    uint32_t source_ports = 0;
    uint32_t l1_eps_per_l1_plane = 4;
    uint32_t degree = 0;
    uint32_t ocs_seed = 42;
    uint32_t spray_p = 4;
    uint32_t spray_h = 2;
    int32_t spray_levels = -1;
    uint32_t ksp_k = 8;
    uint32_t ksp_max_hops = 0;
    uint32_t ksp_seed = 42;
    uint32_t max_paths_per_pair = 100000;
    simtime_picosec link_latency = timeFromUs(1.0);
    string route_plan_path;
};

static OxcOcsDataMode parse_oxc_ocs_mode(const string& value) {
    if (value == "off" || value == "none") {
        return OxcOcsDataMode::OFF;
    }
    if (value == "spraypoint") {
        return OxcOcsDataMode::SPRAYPOINT;
    }
    if (value == "ksp") {
        return OxcOcsDataMode::KSP;
    }
    throw invalid_argument("unknown Oxc OCS mode: " + value);
}

static OxcOcsDataChoice parse_oxc_ocs_choice(const string& value) {
    if (value == "flow_hash" || value == "flow") {
        return OxcOcsDataChoice::FLOW_HASH;
    }
    if (value == "packet_rr" || value == "spray" || value == "packet") {
        return OxcOcsDataChoice::PACKET_RR;
    }
    throw invalid_argument("unknown Oxc OCS choice: " + value);
}

static string oxc_ocs_mode_name(OxcOcsDataMode mode) {
    switch (mode) {
    case OxcOcsDataMode::OFF:
        return "off";
    case OxcOcsDataMode::SPRAYPOINT:
        return "spraypoint";
    case OxcOcsDataMode::KSP:
        return "ksp";
    }
    return "unknown";
}

static string oxc_ocs_choice_name(OxcOcsDataChoice choice) {
    switch (choice) {
    case OxcOcsDataChoice::FLOW_HASH:
        return "flow_hash";
    case OxcOcsDataChoice::PACKET_RR:
        return "packet_rr";
    }
    return "unknown";
}

static OxcOcsDataConfig normalize_oxc_ocs_config(
        const OxcOcsDataConfig& cfg,
        uint32_t no_of_nodes) {
    OxcOcsDataConfig normalized = cfg;
    if (normalized.mode == OxcOcsDataMode::OFF) {
        return normalized;
    }
    if (normalized.groups == 0) {
        throw invalid_argument("Oxc OCS mode requires -oxc_ocs_groups");
    }
    if (normalized.l1_planes == 0 || normalized.l1_eps_per_l1_plane == 0) {
        throw invalid_argument("Oxc OCS mode requires positive L1 plane parameters");
    }
    if (normalized.source_ports == 0) {
        normalized.source_ports = normalized.l1_planes;
    }
    if (normalized.source_ports > normalized.l1_planes) {
        throw invalid_argument("Oxc OCS source_ports cannot exceed l1_planes");
    }
    if (normalized.l1_eps_per_l1_plane % 2 != 0) {
        throw invalid_argument("Oxc OCS l1_eps_per_l1_plane must be even for coupled ports");
    }
    if (normalized.ranks_per_group == 0) {
        if (no_of_nodes % normalized.groups != 0) {
            throw invalid_argument("Oxc OCS could not infer ranks_per_group from nodes/groups");
        }
        normalized.ranks_per_group = no_of_nodes / normalized.groups;
    }
    if (normalized.ranks_per_tray == 0) {
        throw invalid_argument("Oxc OCS requires -local_tray_size or -oxc_ranks_per_tray");
    }
    if (normalized.groups * normalized.ranks_per_group < no_of_nodes) {
        throw invalid_argument("Oxc OCS groups * ranks_per_group is smaller than no_of_nodes");
    }
    return normalized;
}

simtime_picosec calculate_rtt(FatTreeTopologyCfg* t_cfg, linkspeed_bps host_linkspeed) { 
    /*
    Using the host linkspeed here is not very accurate, but hopefully good enough for this usecase.
    */
    simtime_picosec rtt = 2 * t_cfg->get_diameter_latency() 
                + (Packet::data_packet_size() * 8 / speedAsGbps(host_linkspeed) * t_cfg->get_diameter() * 1000) 
                + (UecBasePacket::get_ack_size() * 8 / speedAsGbps(host_linkspeed) * t_cfg->get_diameter() * 1000);
    
    return rtt;
};

simtime_picosec calculate_rtt_from_one_way_hops(
        uint32_t one_way_hops,
        simtime_picosec one_way_link_latency,
        linkspeed_bps link_speed) {
    simtime_picosec serialization =
        (Packet::data_packet_size() * 8 / speedAsGbps(link_speed) * one_way_hops * 1000)
        + (UecBasePacket::get_ack_size() * 8 / speedAsGbps(link_speed) * one_way_hops * 1000);
    return 2 * one_way_hops * one_way_link_latency + serialization;
}

uint32_t calculate_bdp_pkt_from_rtt(simtime_picosec rtt, linkspeed_bps host_linkspeed) {
    return ceil((timeAsSec(rtt) * (host_linkspeed / 8)) / (double)Packet::data_packet_size());
}

uint32_t calculate_bdp_pkt(FatTreeTopologyCfg* t_cfg, linkspeed_bps host_linkspeed) {
    simtime_picosec rtt = calculate_rtt(t_cfg, host_linkspeed);
    return calculate_bdp_pkt_from_rtt(rtt, host_linkspeed);
}

uint32_t estimate_expander_hops(uint32_t nodes, uint32_t degree) {
    if (nodes <= 1) {
        return 0;
    }
    if (degree <= 1) {
        return nodes - 1;
    }
    double hops = ceil(log(static_cast<double>(nodes)) / log(static_cast<double>(degree)));
    return max<uint32_t>(1, static_cast<uint32_t>(hops) + 1);
}

uint32_t estimate_oxc_ocs_hops(const OxcOcsDataConfig& cfg) {
    uint32_t logical_nodes_per_group = cfg.l1_planes * cfg.l1_eps_per_l1_plane / 2;
    uint32_t logical_nodes = cfg.groups * logical_nodes_per_group;

    if (cfg.mode == OxcOcsDataMode::KSP && cfg.ksp_max_hops > 0) {
        return cfg.ksp_max_hops;
    }
    if (cfg.mode == OxcOcsDataMode::SPRAYPOINT && cfg.spray_levels > 0) {
        return static_cast<uint32_t>(cfg.spray_levels) + 1;
    }
    return estimate_expander_hops(logical_nodes, cfg.degree);
}

uint32_t oxc_external_one_way_hops(const OxcOcsDataConfig& cfg) {
    // host->L0, L0->L1, cross-group OCS path, L1->L0, L0->host.
    return 4 + max<uint32_t>(1, estimate_oxc_ocs_hops(cfg));
}

simtime_picosec calculate_oxc_network_rtt(
        const OxcOcsDataConfig& cfg,
        linkspeed_bps external_linkspeed) {
    return calculate_rtt_from_one_way_hops(
            oxc_external_one_way_hops(cfg),
            cfg.link_latency,
            external_linkspeed);
}

simtime_picosec calculate_oxc_pair_rtt(
        const OxcOcsDataConfig& cfg,
        uint32_t src,
        uint32_t dst,
        linkspeed_bps external_linkspeed,
        linkspeed_bps local_linkspeed,
        simtime_picosec local_latency) {
    if (src != dst && src / cfg.ranks_per_tray == dst / cfg.ranks_per_tray) {
        return calculate_rtt_from_one_way_hops(1, local_latency, local_linkspeed);
    }
    if (src / cfg.ranks_per_group == dst / cfg.ranks_per_group) {
        return calculate_rtt_from_one_way_hops(4, cfg.link_latency, external_linkspeed);
    }
    return calculate_oxc_network_rtt(cfg, external_linkspeed);
}

simtime_picosec calculate_mprail_rtt(
        uint32_t one_way_links,
        uint32_t one_way_switches,
        simtime_picosec link_latency,
        simtime_picosec switch_latency,
        linkspeed_bps linkspeed) {
    return calculate_rtt_from_one_way_hops(
                   one_way_links, link_latency, linkspeed)
           + 2 * one_way_switches * switch_latency;
}

simtime_picosec calculate_mprail_pair_rtt(
        const MpRailTopologyConfig& cfg,
        uint32_t src,
        uint32_t dst) {
    const uint32_t src_server = src / cfg.gpus_per_server;
    const uint32_t dst_server = dst / cfg.gpus_per_server;
    if (src_server == dst_server) {
        return calculate_rtt_from_one_way_hops(
                1, cfg.local_latency, cfg.local_linkspeed);
    }

    const uint32_t src_rail = src % cfg.gpus_per_server;
    const uint32_t dst_rail = dst % cfg.gpus_per_server;
    if (src_rail == dst_rail) {
        return calculate_mprail_rtt(
                2, 1, cfg.link_latency, cfg.switch_latency,
                cfg.external_linkspeed);
    }
    return calculate_mprail_rtt(
            4, 3, cfg.link_latency, cfg.switch_latency,
            cfg.external_linkspeed);
}

int main(int argc, char **argv) {
    Clock c(timeFromSec(5 / 100.), eventlist);
    bool param_queuesize_set = false;
    uint32_t queuesize_pkt = 0;
    linkspeed_bps linkspeed = speedFromMbps((double)HOST_NIC);
    int packet_size = 4150;
    uint32_t path_entropy_size = 64;
    uint32_t cwnd = 0, no_of_nodes = 0;
    uint32_t tiers = 3; // we support 2 and 3 tier fattrees
    uint32_t planes = 1;  // multi-plane topologies
    uint32_t ports = 1;  // ports per NIC
    uint32_t local_tray_size = 0;
    linkspeed_bps local_linkspeed = 0;
    bool param_local_linkspeed_set = false;
    simtime_picosec local_latency = timeFromNs(200.0);
    uint64_t local_flow_count = 0;
    uint64_t fabric_flow_count = 0;
    OxcOcsDataConfig oxc_ocs_cfg;
    MpRailTopologyConfig mprail_cfg;
    bool mprail_mode = false;
    bool disable_trim = false; // Disable trimming, drop instead
    uint16_t trimsize = 64; // size of a trimmed packet
    simtime_picosec logtime = timeFromMs(0.25); // ms;
    stringstream filename(ios_base::out);
    simtime_picosec hop_latency = timeFromUs((uint32_t)1);
    simtime_picosec switch_latency = timeFromUs((uint32_t)0);
    queue_type qt = COMPOSITE;

    enum LoadBalancing_Algo { BITMAP, REPS, REPS_LEGACY, FREEZING, OBLIVIOUS, MIXED, ECMP};
    LoadBalancing_Algo load_balancing_algo = MIXED;

    bool log_sink = false;
    bool log_nic = false;
    bool log_flow_events = true;

    bool log_tor_downqueue = false;
    bool log_tor_upqueue = false;
    bool log_traffic = false;
    bool log_switches = false;
    bool log_queue_usage = false;
    const double ecn_thresh = 0.5; // default marking threshold for ECN load balancing
    simtime_picosec target_Qdelay = 0;

    bool param_ecn_set = false;
    bool ecn = true;
    uint32_t ecn_low = 0;
    uint32_t ecn_high = 0;
    uint32_t queue_size_bdp_factor = 0;
    uint32_t topo_num_failed = 0;

    bool receiver_driven = false;
    bool sender_driven = true;

    RouteStrategy route_strategy = NOT_SET;
    string strat_param = "default";

    auto route_strategy_name = [](RouteStrategy strategy) -> string {
        switch (strategy) {
        case NOT_SET:
            return "not_set";
        case SINGLE_PATH:
            return "single_path";
        case SCATTER_PERMUTE:
            return "scatter_permute";
        case SCATTER_RANDOM:
            return "scatter_random";
        case PULL_BASED:
            return "pull_based";
        case SCATTER_ECMP:
            return "scatter_ecmp";
        case ECMP_FIB:
            return "ecmp_fib";
        case ECMP_FIB_ECN:
            return "ecmp_fib_ecn";
        case REACTIVE_ECN:
            return "reactive_ecn";
        }
        return "unknown";
    };

    auto load_balancing_name = [](LoadBalancing_Algo algo) -> string {
        switch (algo) {
        case BITMAP:
            return "bitmap";
        case REPS:
            return "reps";
        case REPS_LEGACY:
            return "reps_legacy";
        case FREEZING:
            return "freezing";
        case OBLIVIOUS:
            return "oblivious";
        case MIXED:
            return "mixed";
        case ECMP:
            return "ecmp";
        }
        return "unknown";
    };
    
    int seed = 13;
    int i = 1;
    double pcie_rate = 1.1;

    filename << "logout.dat";
    string goal_filename = "";
    string dag_file = "";
    int end_time = 1000;//in microseconds
    bool force_disable_oversubscribed_cc = false;
    bool enable_accurate_base_rtt = false;

    //unsure how to set this. 
    queue_type snd_type = FAIR_PRIO;

    float ar_sticky_delta = 10;
    FatTreeSwitch::sticky_choices ar_sticky = FatTreeSwitch::PER_PACKET;

    char* tm_file = NULL;
    char* topo_file = NULL;
    int8_t qa_gate = -1;
    bool conn_reuse = false;

    while (i<argc) {
        if (!strcmp(argv[i],"-o")) {
            filename.str(std::string());
            filename << argv[i+1];
            i++;
        } else if (!strcmp(argv[i],"-conn_reuse")){
            conn_reuse = true;
            cout << "Enabling connection reuse" << endl;
        } else if (!strcmp(argv[i],"-end")) {
            end_time = atoi(argv[i+1]);
            cout << "endtime(us) "<< end_time << endl;
            i++;            
        } else if (!strcmp(argv[i],"-nodes")) {
            no_of_nodes = atoi(argv[i+1]);
            cout << "no_of_nodes "<<no_of_nodes << endl;
            i++;
        } else if (!strcmp(argv[i],"-tiers")) {
            tiers = atoi(argv[i+1]);
            cout << "tiers " << tiers << endl;
            assert(tiers == 2 || tiers == 3);
            i++;
        } else if (!strcmp(argv[i], "-goal")) {
            goal_filename = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i],"-planes")) {
            planes = atoi(argv[i+1]);
            ports = planes;
            cout << "planes " << planes << endl;
            cout << "ports per NIC " << ports << endl;
            assert(planes >= 1 && planes <= 8);
            i++;
        } else if (!strcmp(argv[i],"-local_tray_size")) {
            local_tray_size = atoi(argv[i+1]);
            cout << "local_tray_size " << local_tray_size << endl;
            i++;
        } else if (!strcmp(argv[i],"-local_linkspeed")) {
            local_linkspeed = speedFromMbps(atof(argv[i+1]));
            param_local_linkspeed_set = true;
            cout << "local linkspeed " << local_linkspeed/1000000000 << "Gbps" << endl;
            i++;
        } else if (!strcmp(argv[i],"-local_latency_ns")) {
            local_latency = timeFromNs(atof(argv[i+1]));
            cout << "local latency " << timeAsNs(local_latency) << "ns" << endl;
            i++;
        } else if (!strcmp(argv[i],"-mprail_planes")) {
            mprail_cfg.planes = parse_positive_u32_arg(argv[i], argv[i+1]);
            cout << "mprail_planes " << mprail_cfg.planes << endl;
            i++;
        } else if (!strcmp(argv[i],"-mprail_gpus_per_server")) {
            mprail_cfg.gpus_per_server = parse_positive_u32_arg(argv[i], argv[i+1]);
            cout << "mprail_gpus_per_server " << mprail_cfg.gpus_per_server << endl;
            i++;
        } else if (!strcmp(argv[i],"-mprail_l1_eps_per_plane")) {
            mprail_cfg.l1_eps_per_plane = parse_positive_u32_arg(argv[i], argv[i+1]);
            cout << "mprail_l1_eps_per_plane " << mprail_cfg.l1_eps_per_plane << endl;
            i++;
        } else if (!strcmp(argv[i],"-mprail_l0_l1_links_per_spine")) {
            mprail_cfg.l0_l1_links_per_spine =
                    parse_positive_u32_arg(argv[i], argv[i+1]);
            cout << "mprail_l0_l1_links_per_spine "
                 << mprail_cfg.l0_l1_links_per_spine << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ocs_mode")) {
            try {
                oxc_ocs_cfg.mode = parse_oxc_ocs_mode(argv[i+1]);
            } catch (const exception& e) {
                cout << e.what() << endl;
                exit(1);
            }
            cout << "oxc_ocs_mode " << argv[i+1] << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ocs_choice")) {
            try {
                oxc_ocs_cfg.choice = parse_oxc_ocs_choice(argv[i+1]);
            } catch (const exception& e) {
                cout << e.what() << endl;
                exit(1);
            }
            cout << "oxc_ocs_choice " << argv[i+1] << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ocs_groups")) {
            oxc_ocs_cfg.groups = atoi(argv[i+1]);
            cout << "oxc_ocs_groups " << oxc_ocs_cfg.groups << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ranks_per_group")) {
            oxc_ocs_cfg.ranks_per_group = atoi(argv[i+1]);
            cout << "oxc_ranks_per_group " << oxc_ocs_cfg.ranks_per_group << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ranks_per_tray")) {
            oxc_ocs_cfg.ranks_per_tray = atoi(argv[i+1]);
            cout << "oxc_ranks_per_tray " << oxc_ocs_cfg.ranks_per_tray << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_l1_planes")) {
            oxc_ocs_cfg.l1_planes = atoi(argv[i+1]);
            cout << "oxc_l1_planes " << oxc_ocs_cfg.l1_planes << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_source_ports")) {
            oxc_ocs_cfg.source_ports = atoi(argv[i+1]);
            cout << "oxc_source_ports " << oxc_ocs_cfg.source_ports << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_l1_eps_per_l1_plane")) {
            oxc_ocs_cfg.l1_eps_per_l1_plane = atoi(argv[i+1]);
            cout << "oxc_l1_eps_per_l1_plane " << oxc_ocs_cfg.l1_eps_per_l1_plane << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ocs_degree")) {
            oxc_ocs_cfg.degree = atoi(argv[i+1]);
            cout << "oxc_ocs_degree " << oxc_ocs_cfg.degree << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ocs_seed")) {
            oxc_ocs_cfg.ocs_seed = atoi(argv[i+1]);
            cout << "oxc_ocs_seed " << oxc_ocs_cfg.ocs_seed << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_spray_p")) {
            oxc_ocs_cfg.spray_p = atoi(argv[i+1]);
            cout << "oxc_spray_p " << oxc_ocs_cfg.spray_p << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_spray_h")) {
            oxc_ocs_cfg.spray_h = atoi(argv[i+1]);
            cout << "oxc_spray_h " << oxc_ocs_cfg.spray_h << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_spray_levels")) {
            string value = argv[i+1];
            oxc_ocs_cfg.spray_levels = value == "auto" ? -1 : atoi(argv[i+1]);
            cout << "oxc_spray_levels " << value << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ksp_k")) {
            oxc_ocs_cfg.ksp_k = atoi(argv[i+1]);
            cout << "oxc_ksp_k " << oxc_ocs_cfg.ksp_k << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ksp_max_hops")) {
            string value = argv[i+1];
            oxc_ocs_cfg.ksp_max_hops = value == "auto" ? 0 : atoi(argv[i+1]);
            cout << "oxc_ksp_max_hops " << value << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ksp_seed")) {
            oxc_ocs_cfg.ksp_seed = atoi(argv[i+1]);
            cout << "oxc_ksp_seed " << oxc_ocs_cfg.ksp_seed << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ksp_max_paths_per_pair")) {
            oxc_ocs_cfg.max_paths_per_pair = atoi(argv[i+1]);
            cout << "oxc_ksp_max_paths_per_pair " << oxc_ocs_cfg.max_paths_per_pair << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_ocs_latency_ns")) {
            oxc_ocs_cfg.link_latency = timeFromNs(atof(argv[i+1]));
            cout << "oxc_ocs_latency " << timeAsNs(oxc_ocs_cfg.link_latency) << "ns" << endl;
            i++;
        } else if (!strcmp(argv[i],"-oxc_route_plan")) {
            oxc_ocs_cfg.route_plan_path = argv[i+1];
            cout << "oxc_route_plan " << oxc_ocs_cfg.route_plan_path << endl;
            i++;
        } else if (!strcmp(argv[i],"-receiver_cc_only")) {
            UecSrc::_sender_based_cc = false;
            UecSrc::_receiver_based_cc = true;
            UecSink::_oversubscribed_cc = false;
            sender_driven = false;
            receiver_driven = true;
            cout << "receiver based CC enabled ONLY" << endl;
//        } else if (!strcmp(argv[i],"-disable_fd")) {
//            disable_fair_decrease = true;
//            cout << "fair_decrease disabled" << endl;
        } else if (!strcmp(argv[i],"-sender_cc_only")) {
            UecSrc::_sender_based_cc = true;
            UecSrc::_receiver_based_cc = false;
            UecSink::_oversubscribed_cc = false;
            sender_driven = true;
            receiver_driven = false;
            cout << "sender based CC enabled ONLY" << endl;
        } else if (!strcmp(argv[i],"-qa_gate")) {
            qa_gate = atof(argv[i+1]);
            cout << "qa_gate 2^" << qa_gate << endl;
            i++;
        } else if (!strcmp(argv[i],"-target_q_delay")) {
            target_Qdelay = timeFromUs(atof(argv[i+1]));
            cout << "target_q_delay" << atof(argv[i+1]) << " us"<< endl;
            i++;
        } else if (!strcmp(argv[i],"-queue_size_bdp_factor")) {
            queue_size_bdp_factor = atoi(argv[i+1]);
            cout << "Setting queue size to "<< queue_size_bdp_factor << "x BDP." << endl;
            i++;
        } else if (!strcmp(argv[i],"-sender_cc_algo")) {
            UecSrc::_sender_based_cc = true;
            sender_driven = true;
            
            if (!strcmp(argv[i+1],"dctcp")) 
                UecSrc::_sender_cc_algo = UecSrc::DCTCP;
            else if (!strcmp(argv[i+1],"nscc")) 
                UecSrc::_sender_cc_algo = UecSrc::NSCC;
            else if (!strcmp(argv[i+1],"constant")) 
                UecSrc::_sender_cc_algo = UecSrc::CONSTANT;
            else {
                cout << "UNKNOWN CC ALGO " << argv[i+1] << endl;
                exit(1);
            }    
            cout << "sender based algo "<< argv[i+1] << endl;
            i++;
        } else if (!strcmp(argv[i],"-sender_cc")) {
            UecSrc::_sender_based_cc = true;
            UecSink::_oversubscribed_cc = false;
            sender_driven = true;
            cout << "sender based CC enabled " << endl;
        } else if (!strcmp(argv[i],"-receiver_cc")) {
            UecSrc::_receiver_based_cc = true;
            receiver_driven = true;
            cout << "receiver based CC enabled " << endl;
        }
        else if (!strcmp(argv[i],"-load_balancing_algo")){
            if (!strcmp(argv[i+1], "bitmap")) {
                load_balancing_algo = BITMAP;
            } 
            else if (!strcmp(argv[i+1], "reps")) {
                load_balancing_algo = REPS;
            }
            else if (!strcmp(argv[i+1], "reps_legacy")) {
                load_balancing_algo = REPS_LEGACY;
            }
            else if (!strcmp(argv[i+1], "freezing")) {
                load_balancing_algo = FREEZING;
            }
            else if (!strcmp(argv[i+1], "oblivious")) {
                load_balancing_algo = OBLIVIOUS;
            }
            else if (!strcmp(argv[i+1], "mixed")) {
                load_balancing_algo = MIXED;
            }
            else if (!strcmp(argv[i+1], "mixed")) {
                load_balancing_algo = MIXED;
            }
            else if (!strcmp(argv[i+1], "ecmp")) {
                load_balancing_algo = ECMP;
            }
            else {
                cout << "Unknown load balancing algorithm of type " << argv[i+1] << ", expecting bitmap, reps, reps_legacy, freezing, oblivious, mixed, or ecmp" << endl;
                exit_error(argv[0]);
            }
            cout << "Load balancing algorithm set to  "<< argv[i+1] << endl;
            i++;
        }
        else if (!strcmp(argv[i],"-queue_type")) {
            if (!strcmp(argv[i+1], "composite")) {
                qt = COMPOSITE;
            } 
            else if (!strcmp(argv[i+1], "composite_ecn")) {
                qt = COMPOSITE_ECN;
            }
            else if (!strcmp(argv[i+1], "aeolus")){
                qt = AEOLUS;
            }
            else if (!strcmp(argv[i+1], "aeolus_ecn")){
                qt = AEOLUS_ECN;
            }
            else {
                cout << "Unknown queue type " << argv[i+1] << endl;
                exit_error(argv[0]);
            }
            cout << "queue_type "<< qt << endl;
            i++;
        } else if (!strcmp(argv[i],"-debug")) {
            UecSrc::_debug = true;
            UecPdcSes::_debug = true;
        } else if (!strcmp(argv[i],"-host_queue_type")) {
            if (!strcmp(argv[i+1], "swift")) {
                snd_type = SWIFT_SCHEDULER;
            } 
            else if (!strcmp(argv[i+1], "prio")) {
                snd_type = PRIORITY;
            }
            else if (!strcmp(argv[i+1], "fair_prio")) {
                snd_type = FAIR_PRIO;
            }
            else {
                cout << "Unknown host queue type " << argv[i+1] << " expecting one of swift|prio|fair_prio" << endl;
                exit_error(argv[0]);
            }
            cout << "host queue_type "<< snd_type << endl;
            i++;
        } else if (!strcmp(argv[i],"-log")){
            if (!strcmp(argv[i+1], "flow_events")) {
                log_flow_events = true;
            } else if (!strcmp(argv[i+1], "sink")) {
                cout << "logging sinks\n";
                log_sink = true;
            } else if (!strcmp(argv[i+1], "nic")) {
                cout << "logging nics\n";
                log_nic = true;
            } else if (!strcmp(argv[i+1], "tor_downqueue")) {
                cout << "logging tor downqueues\n";
                log_tor_downqueue = true;
            } else if (!strcmp(argv[i+1], "tor_upqueue")) {
                cout << "logging tor upqueues\n";
                log_tor_upqueue = true;
            } else if (!strcmp(argv[i+1], "switch")) {
                cout << "logging total switch queues\n";
                log_switches = true;
            } else if (!strcmp(argv[i+1], "traffic")) {
                cout << "logging traffic\n";
                log_traffic = true;
            } else if (!strcmp(argv[i+1], "queue_usage")) {
                cout << "logging queue usage\n";
                log_queue_usage = true;
            } else {
                exit_error(argv[0]);
            }
            i++;
        } else if (!strcmp(argv[i],"-cwnd")) {
            cwnd = atoi(argv[i+1]);
            cout << "cwnd "<< cwnd << endl;
            i++;
        } else if (!strcmp(argv[i],"-tm")){
            tm_file = argv[i+1];
            cout << "traffic matrix input file: "<< tm_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-dag") || !strcmp(argv[i],"-oxc_dag")){
            dag_file = argv[i+1];
            cout << "DAG input file: " << dag_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-topology")){
            const string topology = argv[i+1];
            if (topology != "mprail") {
                cerr << "Unknown -topology value " << topology
                     << "; expected mprail" << endl;
                exit(1);
            }
            mprail_mode = true;
            cout << "Topology mode: MpRail" << endl;
            i++;
        } else if (!strcmp(argv[i],"-topo")){
            topo_file = argv[i+1];
            cout << "Topology input file: "<< topo_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-q")){
            param_queuesize_set = true;
            queuesize_pkt = atoi(argv[i+1]);
            cout << "Setting queuesize to " << queuesize_pkt << " packets " << endl;
            i++;
        }
        else if (!strcmp(argv[i],"-sack_threshold")){
            UecSink::_bytes_unacked_threshold = atoi(argv[i+1]);
            cout << "Setting receiver SACK bytes threshold to " << UecSink::_bytes_unacked_threshold  << " bytes " << endl;
            i++;            
        }
        else if (!strcmp(argv[i],"-oversubscribed_cc")){
            UecSink::_oversubscribed_cc = true;
            cout << "Using receiver oversubscribed CC " << endl;
        }
        else if (!strcmp(argv[i],"-Ai")){
            OversubscribedCC::_Ai = atof(argv[i+1]);
            cout << "Using Ai "  << OversubscribedCC::_Ai << endl;
            i+=1;
        }
        else if (!strcmp(argv[i],"-Md")){
            OversubscribedCC::_Md = atof(argv[i+1]);
            cout << "Using Md "  << OversubscribedCC::_Md << endl;
            i+=1;
        }
        else if (!strcmp(argv[i],"-alpha")){
            OversubscribedCC::_alpha = atof(argv[i+1]);
            cout << "Using Alpha "  << OversubscribedCC::_alpha << endl;
            i+=1;
        }
        else if (!strcmp(argv[i],"-force_disable_oversubscribed_cc")){
            UecSink::_oversubscribed_cc = false;
            force_disable_oversubscribed_cc = true;
            cout << "Disabling receiver oversubscribed CC even with OS topology" << endl;
        }
        else if (!strcmp(argv[i],"-enable_accurate_base_rtt")){
            enable_accurate_base_rtt = true;
            cout << "Enable accurate base rtt configuration, each flow uses the accurate end-to-end delay for the current sender/receiver pair as rtt upper bound." << endl;
        }
        else if (!strcmp(argv[i],"-disable_base_rtt_update_on_nack")){
            UecSrc::update_base_rtt_on_nack = false;
            cout << "Disables using NACKs to update the base RTT." << endl;
        }
        else if (!strcmp(argv[i],"-sleek")){
            UecSrc::_enable_sleek = true;
            cout << "Using SLEEK, the sender-based fast loss recovery heuristic " << endl;
        }
        else if (!strcmp(argv[i],"-ecn")){
            // fraction of queuesize, between 0 and 1
            param_ecn_set = true;
            ecn = true;
            ecn_low = atoi(argv[i+1]); 
            ecn_high = atoi(argv[i+2]);
            i+=2;
        } else if (!strcmp(argv[i],"-disable_ecn")) {
            ecn = false;
            cout << "ECN disabled" << endl;
        } else if (!strcmp(argv[i],"-disable_trim")) {
            disable_trim = true;
            cout << "Trimming disabled, dropping instead." << endl;
        } else if (!strcmp(argv[i],"-print_stats_flows")) {
            LogSimInterface::print_stats_flows = true;
            cout << "Printing stats for all flows (ONLY when running with LGS/GOAL)." << endl;
        } else if (!strcmp(argv[i],"-trimsize")){
            // size of trimmed packet in bytes
            trimsize = atoi(argv[i+1]);
            cout << "trimmed packet size: " << trimsize << " bytes\n";
            i+=1;
        } else if (!strcmp(argv[i],"-logtime")){
            double log_ms = atof(argv[i+1]);            
            logtime = timeFromMs(log_ms);
            cout << "logtime "<< log_ms << " ms" << endl;
            i++;
        } else if (!strcmp(argv[i],"-logtime_us")){
            double log_us = atof(argv[i+1]);            
            logtime = timeFromUs(log_us);
            cout << "logtime "<< log_us << " us" << endl;
            i++;
        } else if (!strcmp(argv[i],"-failed")){
            // number of failed links (failed to 25% linkspeed)
            topo_num_failed = atoi(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i],"-linkspeed")){
            // linkspeed specified is in Mbps
            linkspeed = speedFromMbps(atof(argv[i+1]));
            i++;
        } else if (!strcmp(argv[i],"-seed")){
            seed = atoi(argv[i+1]);
            cout << "random seed "<< seed << endl;
            i++;
        } else if (!strcmp(argv[i],"-mtu")){
            packet_size = atoi(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i],"-paths")){
            path_entropy_size = atoi(argv[i+1]);
            cout << "no of paths " << path_entropy_size << endl;
            i++;
        } else if (!strcmp(argv[i],"-hop_latency")){
            hop_latency = timeFromUs(atof(argv[i+1]));
            cout << "Hop latency set to " << timeAsUs(hop_latency) << endl;
            i++;
        } else if (!strcmp(argv[i],"-pcie")){
            UecSink::_model_pcie = true;
            pcie_rate = atof(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i],"-switch_latency")){
            switch_latency = timeFromUs(atof(argv[i+1]));
            cout << "Switch latency set to " << timeAsUs(switch_latency) << endl;
            i++;
        } else if (!strcmp(argv[i],"-ar_sticky_delta")){
            ar_sticky_delta = atof(argv[i+1]);
            cout << "Adaptive routing sticky delta " << ar_sticky_delta << "us" << endl;
            i++;
        } else if (!strcmp(argv[i],"-ar_granularity")){
            if (!strcmp(argv[i+1],"packet"))
                ar_sticky = FatTreeSwitch::PER_PACKET;
            else if (!strcmp(argv[i+1],"flow"))
                ar_sticky = FatTreeSwitch::PER_FLOWLET;
            else  {
                cout << "Expecting -ar_granularity packet|flow, found " << argv[i+1] << endl;
                exit(1);
            }   
            i++;
        } else if (!strcmp(argv[i],"-ar_method")){
            if (!strcmp(argv[i+1],"pause")){
                cout << "Adaptive routing based on pause state " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pause;
            }
            else if (!strcmp(argv[i+1],"queue")){
                cout << "Adaptive routing based on queue size " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_queuesize;
            }
            else if (!strcmp(argv[i+1],"bandwidth")){
                cout << "Adaptive routing based on bandwidth utilization " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_bandwidth;
            }
            else if (!strcmp(argv[i+1],"pqb")){
                cout << "Adaptive routing based on pause, queuesize and bandwidth utilization " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pqb;
            }
            else if (!strcmp(argv[i+1],"pq")){
                cout << "Adaptive routing based on pause, queuesize" << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pq;
            }
            else if (!strcmp(argv[i+1],"pb")){
                cout << "Adaptive routing based on pause, bandwidth utilization" << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pb;
            }
            else if (!strcmp(argv[i+1],"qb")){
                cout << "Adaptive routing based on queuesize, bandwidth utilization" << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_qb; 
            }
            else {
                cout << "Unknown AR method expecting one of pause, queue, bandwidth, pqb, pq, pb, qb" << endl;
                exit(1);
            }
            i++;
        } else if (!strcmp(argv[i],"-strat")){
            strat_param = argv[i+1];
            if (!strcmp(argv[i+1], "ecmp_host")) {
                route_strategy = ECMP_FIB;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
                OxcSwitch::set_strategy(OxcSwitch::ECMP);
                MpRailSwitch::set_strategy(MpRailSwitch::ECMP);
            } else if (!strcmp(argv[i+1], "rr_ecmp")) {
                //this is the host route strategy;
                route_strategy = ECMP_FIB_ECN;
                qt = COMPOSITE_ECN_LB;
                //this is the switch route strategy. 
                FatTreeSwitch::set_strategy(FatTreeSwitch::RR_ECMP);
                OxcSwitch::set_strategy(OxcSwitch::RR);
                MpRailSwitch::set_strategy(MpRailSwitch::RR);
            } else if (!strcmp(argv[i+1], "ecmp_host_ecn")) {
                route_strategy = ECMP_FIB_ECN;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
                OxcSwitch::set_strategy(OxcSwitch::ECMP);
                MpRailSwitch::set_strategy(MpRailSwitch::ECMP);
                qt = COMPOSITE_ECN_LB;
            } else if (!strcmp(argv[i+1], "reactive_ecn")) {
                // Jitu's suggestion for something really simple
                // One path at a time, but switch whenever we get a trim or ecn
                //this is the host route strategy;
                route_strategy = REACTIVE_ECN;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
                OxcSwitch::set_strategy(OxcSwitch::ECMP);
                MpRailSwitch::set_strategy(MpRailSwitch::ECMP);
                qt = COMPOSITE_ECN_LB;
            } else if (!strcmp(argv[i+1], "ecmp_ar")) {
                route_strategy = ECMP_FIB;
                path_entropy_size = 1;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ADAPTIVE_ROUTING);
                OxcSwitch::set_strategy(OxcSwitch::ECMP);
                MpRailSwitch::set_strategy(MpRailSwitch::ECMP);
            } else if (!strcmp(argv[i+1], "ecmp_host_ar")) {
                route_strategy = ECMP_FIB;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP_ADAPTIVE);
                OxcSwitch::set_strategy(OxcSwitch::ECMP);
                MpRailSwitch::set_strategy(MpRailSwitch::ECMP);
                //the stuff below obsolete
                //FatTreeSwitch::set_ar_fraction(atoi(argv[i+2]));
                //cout << "AR fraction: " << atoi(argv[i+2]) << endl;
                //i++;
            } else if (!strcmp(argv[i+1], "ecmp_rr")) {
                // switch round robin
                route_strategy = ECMP_FIB;
                path_entropy_size = 1;
                FatTreeSwitch::set_strategy(FatTreeSwitch::RR);
                OxcSwitch::set_strategy(OxcSwitch::RR);
                MpRailSwitch::set_strategy(MpRailSwitch::RR);
            }
            i++;
        } else {
            cout << "Unknown parameter " << argv[i] << endl;
            exit_error(argv[0]);
        }
        i++;
    }

    if (end_time > 0 && logtime >= timeFromUs((uint32_t)end_time)){
        cout << "Logtime set to endtime" << endl;
        logtime = timeFromUs((uint32_t)end_time) - 1;
    }

    assert(trimsize >= 64 && trimsize <= (uint32_t)packet_size);

    cout << "Packet size (MTU) is " << packet_size << endl;
    if (local_linkspeed == 0) {
        local_linkspeed = linkspeed;
    }
    if (oxc_ocs_cfg.ranks_per_tray == 0 && local_tray_size > 0) {
        oxc_ocs_cfg.ranks_per_tray = local_tray_size;
    }

    srand(seed);
    srandom(seed);
    cout << "Parsed args\n";
    Packet::set_packet_size(packet_size);


    UecSrc::_mtu = Packet::data_packet_size();
    UecSrc::_mss = UecSrc::_mtu - UecSrc::_hdr_size;

    if (route_strategy==NOT_SET){
        route_strategy = ECMP_FIB;
        strat_param = "default_ecmp_host";
        FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
        OxcSwitch::set_strategy(OxcSwitch::ECMP);
        MpRailSwitch::set_strategy(MpRailSwitch::ECMP);
    }

    /*
    UecSink::_oversubscribed_congestion_control = oversubscribed_congestion_control;
    */

    FatTreeSwitch::_ar_sticky = ar_sticky;
    FatTreeSwitch::_sticky_delta = timeFromUs(ar_sticky_delta);
    FatTreeSwitch::_ecn_threshold_fraction = ecn_thresh;
    FatTreeSwitch::_disable_trim = disable_trim;
    FatTreeSwitch::_trim_size = trimsize;

    eventlist.setEndtime(timeFromUs((double)end_time));

    switch (route_strategy) {
    case ECMP_FIB_ECN:
    case REACTIVE_ECN:
        if (qt != COMPOSITE_ECN_LB) {
            fprintf(stderr, "Route Strategy is ECMP ECN.  Must use an ECN queue\n");
            exit(1);
        }
        assert(ecn_thresh > 0 && ecn_thresh < 1);
        // no break, fall through
    case ECMP_FIB:
        if (path_entropy_size > 10000) {
            fprintf(stderr, "Route Strategy is ECMP.  Must specify path count using -paths\n");
            exit(1);
        }
        break;
    case NOT_SET:
        fprintf(stderr, "Route Strategy not set.  Use the -strat param.  \nValid values are perm, rand, pull, rg and single\n");
        exit(1);
    default:
        break;
    }

    // prepare the loggers

    cout << "Logging to " << filename.str() << endl;
    //Logfile 
    Logfile logfile(filename.str(), eventlist);

    cout << "Linkspeed set to " << linkspeed/1000000000 << "Gbps" << endl;
    logfile.setStartTime(timeFromSec(0));

    vector<unique_ptr<UecNIC>> nics;
    vector<unique_ptr<UecNIC>> mprail_local_nics;

    UecSinkLoggerSampling* sink_logger = NULL;
    if (log_sink) {
        sink_logger = new UecSinkLoggerSampling(logtime, eventlist);
        logfile.addLogger(*sink_logger);
    }
    NicLoggerSampling* nic_logger = NULL;
    if (log_nic) {
        nic_logger = new NicLoggerSampling(logtime, eventlist);
        logfile.addLogger(*nic_logger);
    }
    TrafficLoggerSimple* traffic_logger = NULL;
    if (log_traffic) {
        traffic_logger = new TrafficLoggerSimple();
        logfile.addLogger(*traffic_logger);
    }
    FlowEventLoggerSimple* event_logger = NULL;
    if (log_flow_events) {
        event_logger = new FlowEventLoggerSimple();
        logfile.addLogger(*event_logger);
    }

    //UecSrc::setMinRTO(50000); //increase RTO to avoid spurious retransmits
    UecSrc* uec_src;
    UecSink* uec_snk;

    //Route* routeout, *routein;

    QueueLoggerFactory *qlf = 0;
    if (log_tor_downqueue || log_tor_upqueue) {
        qlf = new QueueLoggerFactory(&logfile, QueueLoggerFactory::LOGGER_SAMPLING, eventlist);
        qlf->set_sample_period(logtime);
    } else if (log_queue_usage) {
        qlf = new QueueLoggerFactory(&logfile, QueueLoggerFactory::LOGGER_EMPTY, eventlist);
        qlf->set_sample_period(logtime);
    }

    auto conns = std::make_unique<ConnectionMatrix>(no_of_nodes);

    if (tm_file){
        cout << "Loading connection matrix from  " << tm_file << endl;

        if (!conns->load(tm_file)){
            cout << "Failed to load connection matrix " << tm_file << endl;
            exit(-1);
        }
    }
    else if (goal_filename.size() == 0){
        cout << "Loading connection matrix from  standard input" << endl;        
        conns->load(cin);
    }

    if (conns->N != no_of_nodes && no_of_nodes != 0){
        cout << "Connection matrix number of nodes is " << conns->N << " while I am using " << no_of_nodes << endl;
        exit(-1);
    }

    no_of_nodes = conns->N;
    bool oxc_mode = oxc_ocs_cfg.mode != OxcOcsDataMode::OFF;
    if (mprail_mode && oxc_mode) {
        cerr << "MpRail and Oxc topology modes are mutually exclusive" << endl;
        exit(1);
    }
    const bool fabric_mode = mprail_mode || oxc_mode;
    if (!dag_file.empty()) {
        if (!fabric_mode) {
            cerr << "-dag requires -topology mprail or an Oxc topology" << endl;
            exit(1);
        }
        if (conn_reuse) {
            cerr << "-dag does not support -conn_reuse" << endl;
            exit(1);
        }
        if (!goal_filename.empty()) {
            cerr << "-dag cannot be combined with -goal" << endl;
            exit(1);
        }
    }
    if (mprail_mode) {
        if (mprail_cfg.planes == 0 || mprail_cfg.planes > 8
                || mprail_cfg.gpus_per_server == 0
                || mprail_cfg.l1_eps_per_plane == 0
                || mprail_cfg.l0_l1_links_per_spine == 0) {
            cerr << "Invalid MpRail dimensions; counts must be positive and planes must be <= 8"
                 << endl;
            exit(1);
        }
        if (local_tray_size != 0
                && local_tray_size != mprail_cfg.gpus_per_server) {
            cerr << "-local_tray_size conflicts with -mprail_gpus_per_server"
                 << endl;
            exit(1);
        }
        local_tray_size = mprail_cfg.gpus_per_server;
        if (!param_local_linkspeed_set) {
            local_linkspeed = mprail_cfg.local_linkspeed;
        }
        planes = mprail_cfg.planes;
        ports = planes;
        mprail_cfg.packet_spray = load_balancing_algo != ECMP;
        mprail_cfg.nodes = no_of_nodes;
        mprail_cfg.external_linkspeed = linkspeed;
        mprail_cfg.local_linkspeed = local_linkspeed;
        mprail_cfg.link_latency = hop_latency;
        mprail_cfg.local_latency = local_latency;
        mprail_cfg.switch_latency = switch_latency;
    }
    if (oxc_mode) {
        try {
            oxc_ocs_cfg = normalize_oxc_ocs_config(oxc_ocs_cfg, no_of_nodes);
        } catch (const exception& e) {
            cerr << "Failed to initialize Oxc OCS switch-local topology config: "
                 << e.what() << endl;
            exit(1);
        }
        planes = oxc_ocs_cfg.l1_planes;
        ports = oxc_ocs_cfg.source_ports;
    }

    if (local_tray_size > 0) {
        if (!mprail_mode && no_of_nodes % local_tray_size != 0) {
            cerr << "local_tray_size " << local_tray_size
                 << " must divide the node count " << no_of_nodes << endl;
            exit(1);
        }
        cout << "Server-local FullMesh routing enabled: gpus_per_server "
             << local_tray_size
             << ", local linkspeed " << local_linkspeed/1000000000 << "Gbps"
             << ", local one-way latency " << timeAsNs(local_latency) << "ns"
             << endl;
    }

    if (!param_queuesize_set) {
        cout << "Automatic queue sizing enabled ";        
        if (queue_size_bdp_factor==0) {
            if (disable_trim) {
                queue_size_bdp_factor = DEFAULT_NONTRIMMING_QUEUESIZE_FACTOR;
                cout << "non-trimming";
            } else {
                queue_size_bdp_factor = DEFAULT_TRIMMING_QUEUESIZE_FACTOR;
                cout << "trimming";
            }
        }
        cout << " queue-size-to-bdp-factor is " << queue_size_bdp_factor << "xBDP"
             << endl;
    }

    unique_ptr<FatTreeTopologyCfg> topo_cfg;
    simtime_picosec network_max_unloaded_rtt = 0;
    uint32_t bdp_pkt = 0;
    mem_b queuesize = 0;

    if (mprail_mode) {
        if (topo_file) {
            cerr << "MpRail does not use a FatTree topology file" << endl;
            exit(1);
        }
        if (topo_num_failed > 0 || !conns->failures.empty()) {
            cerr << "MpRail does not yet support failed-link declarations" << endl;
            exit(1);
        }

        network_max_unloaded_rtt = calculate_mprail_rtt(
                4, 3, hop_latency, switch_latency, linkspeed);
        bdp_pkt = calculate_bdp_pkt_from_rtt(
                network_max_unloaded_rtt, linkspeed);
        queuesize = param_queuesize_set
                ? memFromPkt(queuesize_pkt)
                : memFromPkt(bdp_pkt * queue_size_bdp_factor);
        mprail_cfg.queue_size = queuesize;

        const uint32_t server_count =
                (no_of_nodes + mprail_cfg.gpus_per_server - 1)
                / mprail_cfg.gpus_per_server;
        const uint32_t rail_count = mprail_cfg.gpus_per_server;
        const uint32_t l0_count = rail_count * mprail_cfg.planes;
        const uint32_t l1_count =
                mprail_cfg.planes * mprail_cfg.l1_eps_per_plane;

        cout << endl;
        cout << "#----------- MPRAIL CONFIG begin ------------" << endl;
        cout << "nodes " << no_of_nodes << endl;
        cout << "servers " << server_count << endl;
        cout << "rails " << rail_count << endl;
        cout << "planes " << mprail_cfg.planes << endl;
        cout << "gpus_per_server " << mprail_cfg.gpus_per_server << endl;
        cout << "rail_mapping gpu_local_index" << endl;
        cout << "l0_switches " << l0_count << endl;
        cout << "l1_switches " << l1_count << endl;
        cout << "l1_eps_per_plane " << mprail_cfg.l1_eps_per_plane << endl;
        cout << "l0_l1_links_per_spine "
             << mprail_cfg.l0_l1_links_per_spine << endl;
        cout << "routing_mode "
             << (mprail_cfg.packet_spray ? "packet_spray_ecmp" : "flow_ecmp")
             << endl;
        cout << "switch_strategy "
             << (MpRailSwitch::strategy() == MpRailSwitch::RR
                     ? "rr" : "ecmp_hash")
             << endl;
        cout << "external_linkspeed_gbps " << linkspeed / 1000000000 << endl;
        cout << "aggregate_rank_bandwidth_gbps "
             << mprail_cfg.planes * (linkspeed / 1000000000) << endl;
        cout << "local_linkspeed_gbps "
             << local_linkspeed / 1000000000 << endl;
        cout << "link_latency_ns " << timeAsNs(hop_latency) << endl;
        cout << "local_latency_ns " << timeAsNs(local_latency) << endl;
        cout << "switch_latency_ns " << timeAsNs(switch_latency) << endl;
        cout << "network_max_unloaded_rtt_us "
             << timeAsUs(network_max_unloaded_rtt) << endl;
        cout << "bdp_pkt " << bdp_pkt << endl;
        cout << "queue_size_bytes " << queuesize << endl;
        cout << "#----------- MPRAIL CONFIG END ------------" << endl;
        cout << endl;
    } else if (oxc_mode) {
        if (topo_file) {
            cout << "Oxc OCS mode ignores topology input file: " << topo_file << endl;
        }
        if (topo_num_failed > 0 || !conns->failures.empty()) {
            cerr << "Oxc OCS mode does not support FatTree-style failed links in the connection matrix" << endl;
            exit(1);
        }

        network_max_unloaded_rtt = calculate_oxc_network_rtt(oxc_ocs_cfg, linkspeed);
        bdp_pkt = calculate_bdp_pkt_from_rtt(network_max_unloaded_rtt, linkspeed);
        if (!param_queuesize_set) {
            queuesize = memFromPkt(bdp_pkt * queue_size_bdp_factor);
        } else {
            queuesize = memFromPkt(queuesize_pkt);
        }

        uint32_t oxc_total_trays =
            (no_of_nodes + oxc_ocs_cfg.ranks_per_tray - 1) / oxc_ocs_cfg.ranks_per_tray;
        uint32_t oxc_trays_per_group =
            (oxc_ocs_cfg.ranks_per_group + oxc_ocs_cfg.ranks_per_tray - 1)
            / oxc_ocs_cfg.ranks_per_tray;
        uint32_t oxc_l0_count = oxc_total_trays * oxc_ocs_cfg.l1_planes;
        uint32_t oxc_l0_per_group = oxc_trays_per_group * oxc_ocs_cfg.l1_planes;
        uint32_t oxc_l1_per_group =
            oxc_ocs_cfg.l1_planes * oxc_ocs_cfg.l1_eps_per_l1_plane;
        uint32_t oxc_l1_count = oxc_ocs_cfg.groups * oxc_l1_per_group;
        uint32_t oxc_logical_nodes_per_group =
            oxc_ocs_coupled_logical_nodes_per_group(
                    oxc_ocs_cfg.l1_planes, oxc_ocs_cfg.l1_eps_per_l1_plane);
        uint32_t oxc_logical_nodes = oxc_ocs_cfg.groups * oxc_logical_nodes_per_group;
        uint32_t oxc_full_cross_group_degree =
            oxc_logical_nodes > oxc_logical_nodes_per_group
                ? oxc_logical_nodes - oxc_logical_nodes_per_group
                : 0;

        cout << endl;
        cout << "#----------- OXC TOPOLOGY begin ------------" << endl;
        cout << "[INPUT]" << endl;
        cout << "nodes " << no_of_nodes << endl;
        cout << "groups " << oxc_ocs_cfg.groups << endl;
        cout << "ranks_per_group " << oxc_ocs_cfg.ranks_per_group << endl;
        cout << "ranks_per_tray " << oxc_ocs_cfg.ranks_per_tray << endl;
        cout << "l1_planes " << oxc_ocs_cfg.l1_planes << endl;
        cout << "source_ports " << ports << endl;
        cout << "l1_eps_per_l1_plane " << oxc_ocs_cfg.l1_eps_per_l1_plane << endl;
        cout << "ocs_degree " << oxc_ocs_cfg.degree << endl;
        cout << "ocs_seed " << oxc_ocs_cfg.ocs_seed << endl;
        cout << "route_plan_path "
             << (oxc_ocs_cfg.route_plan_path.empty() ? "none" : oxc_ocs_cfg.route_plan_path)
             << endl;
        cout << endl;
        cout << "[DERIVED]" << endl;
        cout << "total_trays " << oxc_total_trays << endl;
        cout << "trays_per_group " << oxc_trays_per_group << endl;
        cout << "l0_switches " << oxc_l0_count << endl;
        cout << "l0_switches_per_group " << oxc_l0_per_group << endl;
        cout << "l1_switches " << oxc_l1_count << endl;
        cout << "l1_switches_per_group " << oxc_l1_per_group << endl;
        cout << "ocs_coupled_logical_nodes " << oxc_logical_nodes << endl;
        cout << "ocs_coupled_logical_nodes_per_group " << oxc_logical_nodes_per_group << endl;
        cout << "ocs_full_cross_group_degree " << oxc_full_cross_group_degree << endl;
        cout << "ocs_degree_is_full_cross_group "
             << (oxc_ocs_cfg.degree >= oxc_full_cross_group_degree ? "yes" : "no")
             << endl;
        cout << "#----------- OXC TOPOLOGY END ------------" << endl;
        cout << endl;

        cout << "#----------- OXC ROUTING begin ------------" << endl;
        cout << "[INPUT]" << endl;
        cout << "strat_param " << strat_param << endl;
        cout << "load_balancing_algo " << load_balancing_name(load_balancing_algo) << endl;
        cout << "oxc_ocs_mode " << oxc_ocs_mode_name(oxc_ocs_cfg.mode) << endl;
        cout << "oxc_ocs_choice " << oxc_ocs_choice_name(oxc_ocs_cfg.choice) << endl;
        if (oxc_ocs_cfg.mode == OxcOcsDataMode::KSP) {
            cout << "oxc_ksp_k " << oxc_ocs_cfg.ksp_k << endl;
            cout << "oxc_ksp_max_hops ";
            if (oxc_ocs_cfg.ksp_max_hops == 0) {
                cout << "auto";
            } else {
                cout << oxc_ocs_cfg.ksp_max_hops;
            }
            cout << endl;
            cout << "oxc_ksp_seed " << oxc_ocs_cfg.ksp_seed << endl;
            cout << "oxc_ksp_max_paths_per_pair " << oxc_ocs_cfg.max_paths_per_pair << endl;
        } else if (oxc_ocs_cfg.mode == OxcOcsDataMode::SPRAYPOINT) {
            cout << "oxc_spray_p " << oxc_ocs_cfg.spray_p << endl;
            cout << "oxc_spray_h " << oxc_ocs_cfg.spray_h << endl;
            cout << "oxc_spray_levels ";
            if (oxc_ocs_cfg.spray_levels < 0) {
                cout << "auto";
            } else {
                cout << oxc_ocs_cfg.spray_levels;
            }
            cout << endl;
        }
        cout << endl;
        cout << "[DERIVED]" << endl;
        cout << "host_route_strategy " << route_strategy_name(route_strategy) << endl;
        cout << "source_ports_per_nonlocal_flow " << ports << endl;
        cout << "local_tray_direct_enabled " << (oxc_ocs_cfg.ranks_per_tray > 1 ? "yes" : "no") << endl;
        cout << "switch_local_fib_enabled yes" << endl;
        cout << "estimated_ocs_hops " << estimate_oxc_ocs_hops(oxc_ocs_cfg) << endl;
        cout << "#----------- OXC ROUTING END ------------" << endl;
        cout << endl;

        cout << "#----------- OXC LINK_QUEUE begin ------------" << endl;
        cout << "[INPUT]" << endl;
        cout << "external_linkspeed_gbps " << linkspeed / 1000000000 << endl;
        cout << "local_linkspeed_gbps " << local_linkspeed / 1000000000 << endl;
        cout << "local_latency_ns " << timeAsNs(local_latency) << endl;
        cout << "ocs_latency_ns " << timeAsNs(oxc_ocs_cfg.link_latency) << endl;
        cout << "mtu_bytes " << Packet::data_packet_size() << endl;
        cout << "queue_size_mode " << (param_queuesize_set ? "fixed_packets" : "bdp_factor") << endl;
        if (param_queuesize_set) {
            cout << "queue_size_input_packets " << queuesize_pkt << endl;
        } else {
            cout << "queue_size_bdp_factor " << queue_size_bdp_factor << endl;
        }
        cout << endl;
        cout << "[DERIVED]" << endl;
        cout << "queue_size_bytes " << queuesize << endl;
        cout << "estimated_one_way_hops " << oxc_external_one_way_hops(oxc_ocs_cfg) << endl;
        cout << "network_max_unloaded_rtt_us " << timeAsUs(network_max_unloaded_rtt) << endl;
        cout << "bdp_pkt " << bdp_pkt << endl;
        cout << "oxc_queue_ecn see_ECN_block_below" << endl;
        cout << "#----------- OXC LINK_QUEUE END ------------" << endl;
        cout << endl;

        cout << "#----------- OXC OUTPUT begin ------------" << endl;
        const char* link_sample = getenv("HTSIM_LINK_LOAD_SAMPLE");
        const char* link_sample_us = getenv("HTSIM_LINK_LOAD_SAMPLE_US");
        cout << "[INPUT]" << endl;
        cout << "link_load_sample " << (link_sample ? link_sample : "0") << endl;
        cout << "link_load_sample_us " << (link_sample_us ? link_sample_us : "1000") << endl;
        cout << endl;
        cout << "[DERIVED]" << endl;
        cout << "output_metrics_dir output_metrics" << endl;
        cout << "link_info_file output_metrics/link_info.csv" << endl;
        cout << "link_load_file output_metrics/link_load_1ms.csv" << endl;
        cout << "#----------- OXC OUTPUT END ------------" << endl;
        cout << endl;
    } else {
        if (topo_file) {
            topo_cfg = FatTreeTopologyCfg::load(topo_file, memFromPkt(queuesize_pkt), qt, snd_type);

            if (topo_cfg->no_of_nodes() != no_of_nodes) {
                cerr << "Mismatch between connection matrix (" << no_of_nodes << " nodes) and topology ("
                     << topo_cfg->no_of_nodes() << " nodes)" << endl;
                exit(1);
            }
        } else {
            topo_cfg = make_unique<FatTreeTopologyCfg>(tiers, no_of_nodes, linkspeed, memFromPkt(queuesize_pkt),
                                                       hop_latency, switch_latency,
                                                       qt, snd_type);
        }

        network_max_unloaded_rtt = calculate_rtt(topo_cfg.get(), linkspeed);
        bdp_pkt = calculate_bdp_pkt(topo_cfg.get(), linkspeed);
        if (!param_queuesize_set) {
            queuesize = memFromPkt(bdp_pkt * queue_size_bdp_factor);
        } else {
            queuesize = memFromPkt(queuesize_pkt);
        }
        topo_cfg->set_queue_sizes(queuesize);

        if (topo_num_failed > 0) {
            topo_cfg->set_failed_links(topo_num_failed);
        }

        if (topo_cfg->get_oversubscription_ratio() > 1 && !UecSrc::_sender_based_cc && !force_disable_oversubscribed_cc) {
            UecSink::_oversubscribed_cc = true;
            OversubscribedCC::setOversubscriptionRatio(topo_cfg->get_oversubscription_ratio());
            cout << "Using simple receiver oversubscribed CC. Oversubscription ratio is " << topo_cfg->get_oversubscription_ratio() << endl;
        }
    }

    //2 priority queues; 3 hops for incast
    UecSrc::_min_rto = timeFromUs(15 + queuesize * 6.0 * 8 * 1000000 / linkspeed);
    cout << "Setting min RTO to " << timeAsUs(UecSrc::_min_rto) << endl;

    if (ecn){
        if (!param_ecn_set) {
            ecn_low = memFromPkt(ceil(bdp_pkt * 0.2));
            ecn_high = memFromPkt(ceil(bdp_pkt * 0.8));
        } else {
            ecn_low = memFromPkt(ecn_low);
            ecn_high = memFromPkt(ecn_high);
        }
        cout << "Setting ECN to parameters low " << ecn_low << " high " << ecn_high;
        if (mprail_mode) {
            cout << " (MpRail ECNQueue threshold uses low)";
            mprail_cfg.enable_ecn = true;
            mprail_cfg.ecn_threshold = ecn_low;
        } else if (oxc_mode) {
            cout << " (Oxc ECNQueue threshold uses low)";
        } else {
            cout << " enable on tor downlink " << !receiver_driven;
            topo_cfg->set_ecn_parameters(true, !receiver_driven, ecn_low, ecn_high);
        }
        cout << endl;
        assert(ecn_low <= ecn_high);
        assert(ecn_high <= queuesize);
        if (mprail_mode) {
            cout << "MPRAIL_ECN enabled=yes threshold_bytes=" << ecn_low
                 << " queue_size_bytes=" << queuesize << endl;
        } else if (oxc_mode) {
            cout << "#----------- OXC ECN begin ------------" << endl;
            cout << "enabled yes" << endl;
            cout << "queue_type ecnqueue" << endl;
            cout << "ecn_low_bytes " << ecn_low << endl;
            cout << "ecn_high_bytes " << ecn_high << endl;
            cout << "oxc_ecn_threshold_bytes " << ecn_low << endl;
            cout << "queue_size_bytes " << queuesize << endl;
            cout << "#----------- OXC ECN END ------------" << endl;
        }
    } else if (mprail_mode) {
        mprail_cfg.enable_ecn = false;
        mprail_cfg.ecn_threshold = 0;
        cout << "MPRAIL_ECN enabled=no" << endl;
    } else if (oxc_mode) {
        cout << "#----------- OXC ECN begin ------------" << endl;
        cout << "enabled no" << endl;
        cout << "queue_type drop_tail" << endl;
        cout << "#----------- OXC ECN END ------------" << endl;
    }

    if (!fabric_mode) {
        cout << *topo_cfg << endl;
    }

    vector<unique_ptr<FatTreeTopology>> topo;
    if (!fabric_mode) {
        topo.resize(planes);
        for (uint32_t p = 0; p < planes; p++) {
            topo[p] = make_unique<FatTreeTopology>(topo_cfg.get(), qlf, &eventlist, nullptr);

            if (log_switches) {
                topo[p]->add_switch_loggers(logfile, logtime);
            }
        }
    } else if (log_switches) {
        cout << "Selected fabric topology ignores legacy FatTree switch loggers"
             << endl;
    }
    cout << "network_max_unloaded_rtt " << timeAsUs(network_max_unloaded_rtt) << endl;

    unique_ptr<MpRailTopology> mprail_topology;
    if (mprail_mode) {
        try {
            mprail_topology = make_unique<MpRailTopology>(mprail_cfg, eventlist);
        } catch (const exception& e) {
            cerr << "Failed to initialize MpRail topology: " << e.what() << endl;
            exit(1);
        }
        cout << "MpRail topology built: nodes " << no_of_nodes
             << ", servers " << mprail_topology->server_count()
             << ", rails " << mprail_topology->rail_count()
             << ", planes " << mprail_topology->planes()
             << ", L0 " << mprail_topology->l0_count()
             << ", L1 " << mprail_topology->l1_count()
             << endl;
    }

    unique_ptr<OxcTopology> oxc_topology;
    if (oxc_ocs_cfg.mode != OxcOcsDataMode::OFF) {
        OxcTopologyConfig oxc_topology_cfg;
        oxc_topology_cfg.nodes = no_of_nodes;
        oxc_topology_cfg.groups = oxc_ocs_cfg.groups;
        oxc_topology_cfg.ranks_per_group = oxc_ocs_cfg.ranks_per_group;
        oxc_topology_cfg.ranks_per_tray = oxc_ocs_cfg.ranks_per_tray;
        oxc_topology_cfg.l1_planes = oxc_ocs_cfg.l1_planes;
        oxc_topology_cfg.source_ports = oxc_ocs_cfg.source_ports;
        oxc_topology_cfg.l1_eps_per_l1_plane = oxc_ocs_cfg.l1_eps_per_l1_plane;
        oxc_topology_cfg.ocs_degree = oxc_ocs_cfg.degree;
        oxc_topology_cfg.ocs_seed = oxc_ocs_cfg.ocs_seed;
        oxc_topology_cfg.ocs_mode = oxc_ocs_cfg.mode == OxcOcsDataMode::SPRAYPOINT
                ? OxcOcsMode::SPRAYPOINT
                : OxcOcsMode::KSP;
        oxc_topology_cfg.ocs_choice = oxc_ocs_cfg.choice == OxcOcsDataChoice::FLOW_HASH
                ? OxcOcsChoice::FLOW_HASH
                : OxcOcsChoice::PACKET_RR;
        oxc_topology_cfg.spray_p = oxc_ocs_cfg.spray_p;
        oxc_topology_cfg.spray_h = oxc_ocs_cfg.spray_h;
        oxc_topology_cfg.spray_levels = oxc_ocs_cfg.spray_levels;
        oxc_topology_cfg.ksp_k = oxc_ocs_cfg.ksp_k;
        oxc_topology_cfg.ksp_max_hops = oxc_ocs_cfg.ksp_max_hops;
        oxc_topology_cfg.ksp_seed = oxc_ocs_cfg.ksp_seed;
        oxc_topology_cfg.ksp_max_paths_per_pair = oxc_ocs_cfg.max_paths_per_pair;
        oxc_topology_cfg.external_linkspeed = linkspeed;
        oxc_topology_cfg.local_linkspeed = local_linkspeed;
        oxc_topology_cfg.queue_size = queuesize;
        oxc_topology_cfg.enable_ecn = ecn;
        oxc_topology_cfg.ecn_threshold = ecn ? ecn_low : 0;
        oxc_topology_cfg.link_latency = oxc_ocs_cfg.link_latency;
        oxc_topology_cfg.local_latency = local_latency;
        oxc_topology_cfg.switch_latency = switch_latency;
        oxc_topology_cfg.route_plan_path = oxc_ocs_cfg.route_plan_path;
        try {
            oxc_topology = make_unique<OxcTopology>(oxc_topology_cfg, eventlist);
        } catch (const exception& e) {
            cerr << "Failed to initialize Oxc switch-local topology: "
                 << e.what() << endl;
            exit(1);
        }
        cout << "Oxc switch-local topology built: nodes " << oxc_topology_cfg.nodes
             << ", groups " << oxc_topology_cfg.groups
             << ", ranks_per_group " << oxc_topology_cfg.ranks_per_group
             << ", ranks_per_tray " << oxc_topology_cfg.ranks_per_tray
             << ", l1_planes/ports " << oxc_topology_cfg.l1_planes
             << ", l1_eps_per_l1_plane " << oxc_topology_cfg.l1_eps_per_l1_plane
             << ", ocs_degree " << oxc_topology_cfg.ocs_degree
             << ", external linkspeed " << oxc_topology_cfg.external_linkspeed / 1000000000 << "Gbps"
             << ", local linkspeed " << oxc_topology_cfg.local_linkspeed / 1000000000 << "Gbps"
             << endl;
    }

    if (UecSink::_oversubscribed_cc)
        OversubscribedCC::_base_rtt = network_max_unloaded_rtt;

    
    // Handle FatTree link failures specified in the connection matrix.
    if (!fabric_mode) {
        for (size_t c = 0; c < conns->failures.size(); c++){
            failure* crt = conns->failures.at(c);

            cout << "Adding link failure switch type" << crt->switch_type << " Switch ID " << crt->switch_id << " link ID "  << crt->link_id << endl;
            // xxx we only support failures in plane 0 for now.
            topo[0]->add_failed_link(crt->switch_type,crt->switch_id,crt->link_id);
        }
    }

    // Initialize congestion control algorithms
    if (receiver_driven) {
        // TBD
    }
    if (sender_driven) {
        // UecSrc::parameterScaleToTargetQ();
        bool trimming_enabled = !disable_trim;
        UecSrc::initNsccParams(network_max_unloaded_rtt, linkspeed, target_Qdelay, qa_gate, trimming_enabled);
    }

    vector<unique_ptr<UecPullPacer>> pacers;
    vector<unique_ptr<UecPullPacer>> mprail_local_pacers;
    vector<PCIeModel*> pcie_models;
    vector<OversubscribedCC*> oversubscribed_ccs;

    for (size_t ix = 0; ix < no_of_nodes; ix++){
        auto &pacer = pacers.emplace_back(make_unique<UecPullPacer>(linkspeed, 0.99,
          UecBasePacket::unquantize(UecSink::_credit_per_pull), eventlist, ports));

        if (UecSink::_model_pcie)
            pcie_models.push_back(new PCIeModel(linkspeed * pcie_rate, UecSrc::_mtu, eventlist,
              pacer.get()));

        if (UecSink::_oversubscribed_cc)
            oversubscribed_ccs.push_back(new OversubscribedCC(eventlist, pacer.get()));

        auto &nic = nics.emplace_back(make_unique<UecNIC>(ix, eventlist,
                                                          linkspeed, ports));
        if (log_nic) {
            nic_logger->monitorNic(nic.get());
        }

        if (mprail_mode) {
            mprail_local_pacers.emplace_back(make_unique<UecPullPacer>(
                    local_linkspeed, 0.99,
                    UecBasePacket::unquantize(UecSink::_credit_per_pull),
                    eventlist, 1));
            mprail_local_nics.emplace_back(make_unique<UecNIC>(
                    ix, eventlist, local_linkspeed, 1));
        }
    }
    if (mprail_mode) {
        cout << "MPRAIL_LOCAL_INJECTION independent=yes speed_gbps="
             << local_linkspeed / 1000000000
             << " ports_per_rank=1" << endl;
    }

    // used just to print out stats data at the end
    list <const Route*> routes;

    mem_b cwnd_b = cwnd * Packet::data_packet_size();
    vector<connection*>* all_conns = conns->getAllConnections();
    if (!conn_reuse) {
        set<flowid_t> user_flow_ids;
        for (const connection* item : *all_conns) {
            if (item->flowid && !user_flow_ids.insert(item->flowid).second) {
                cerr << "Duplicate flow ID in connection matrix: "
                     << item->flowid << endl;
                exit(1);
            }
        }
    }
    vector <UecSrc*> uec_srcs;

    struct MpRailFlowHandle {
        UecSrc* source = nullptr;
        UecSink* sink = nullptr;
    };

    auto make_uec_multipath = [&](bool fixed_explicit_path) {
        unique_ptr<UecMultipath> mp;
        if (fixed_explicit_path) {
            mp = make_unique<UecMpEcmp>(1, UecSrc::_debug);
        } else if (load_balancing_algo == BITMAP) {
            mp = make_unique<UecMpBitmap>(path_entropy_size, UecSrc::_debug);
        } else if (load_balancing_algo == REPS
                   || load_balancing_algo == REPS_LEGACY) {
            mp = make_unique<UecMpRepsLegacy>(path_entropy_size, UecSrc::_debug);
        } else if (load_balancing_algo == FREEZING) {
            mp = make_unique<UecMpReps>(
                    path_entropy_size, UecSrc::_debug, !disable_trim);
        } else if (load_balancing_algo == OBLIVIOUS) {
            mp = make_unique<UecMpOblivious>(path_entropy_size, UecSrc::_debug);
        } else if (load_balancing_algo == MIXED) {
            mp = make_unique<UecMpMixed>(path_entropy_size, UecSrc::_debug);
        } else if (load_balancing_algo == ECMP) {
            mp = make_unique<UecMpEcmp>(path_entropy_size, UecSrc::_debug);
        } else {
            throw logic_error("unsupported UEC multipath algorithm");
        }
        return mp;
    };

    auto create_mprail_flow = [&] (
            uint32_t src,
            uint32_t dst,
            uint64_t bytes,
            optional<flowid_t> requested_flow_id,
            simtime_picosec start_time,
            const string& name,
            const MpRailRouteSpec* explicit_route,
            function<void()> completion_callback) {
        if (!mprail_topology) {
            throw logic_error("MpRail route flow requested without MpRail topology");
        }
        const bool local_flow = mprail_topology->rank_server(src)
                == mprail_topology->rank_server(dst);
        UecNIC& source_nic = local_flow
                ? *mprail_local_nics.at(src) : *nics.at(src);
        UecNIC& destination_nic = local_flow
                ? *mprail_local_nics.at(dst) : *nics.at(dst);
        const uint32_t flow_ports = local_flow ? 1 : ports;
        const linkspeed_bps flow_linkspeed = local_flow
                ? local_linkspeed : linkspeed;
        auto* flow_src = new UecSrc(
                traffic_logger, eventlist,
                make_uec_multipath(explicit_route != nullptr),
                source_nic, flow_ports);
        if (requested_flow_id.has_value()) {
            flow_src->setFlowId(requested_flow_id.value());
        }
        flow_src->setFlowsize(bytes);
        flow_src->setSrc(src);
        flow_src->setDst(dst);
        flow_src->setName(name);
        logfile.writeName(*flow_src);

        UecSink* flow_sink = nullptr;
        if (receiver_driven) {
            flow_sink = new UecSink(
                    NULL,
                    local_flow ? mprail_local_pacers.at(dst).get()
                               : pacers.at(dst).get(),
                    destination_nic, flow_ports);
        } else {
            flow_sink = new UecSink(
                    NULL, flow_linkspeed, 1.1,
                    UecBasePacket::unquantize(UecSink::_credit_per_pull),
                    eventlist, destination_nic, flow_ports);
        }
        if (requested_flow_id.has_value()) {
            flow_sink->setFlowId(requested_flow_id.value());
        }
        flow_sink->setSrc(src);
        ((DataReceiver*)flow_sink)->setName(name + "_sink");
        logfile.writeName(*(DataReceiver*)flow_sink);

        const simtime_picosec base_rtt = calculate_mprail_pair_rtt(
                mprail_cfg, src, dst);
        if (receiver_driven) {
            flow_src->initRccc(cwnd_b, enable_accurate_base_rtt
                    ? base_rtt : network_max_unloaded_rtt);
        }
        if (sender_driven) {
            flow_src->initNscc(cwnd_b, enable_accurate_base_rtt
                    ? base_rtt : network_max_unloaded_rtt);
        }
        if (log_flow_events && event_logger) {
            flow_src->logFlowEvents(*event_logger);
        }
        if (UecSink::_model_pcie && !local_flow) {
            flow_sink->setPCIeModel(pcie_models.at(dst));
        }
        if (UecSink::_oversubscribed_cc) {
            flow_sink->setOversubscribedCC(oversubscribed_ccs.at(dst));
        }
        if (completion_callback) {
            flow_src->setCompletionCallback(move(completion_callback));
        }

        mprail_topology->connect_endpoints(
                src, dst, *flow_src, *flow_sink, start_time, explicit_route);
        if (sink_logger) {
            sink_logger->monitorSink(flow_sink);
        }
        if (local_flow) {
            ++local_flow_count;
        } else {
            ++fabric_flow_count;
        }
        uec_srcs.push_back(flow_src);
        return MpRailFlowHandle{flow_src, flow_sink};
    };

    struct ServerForwardPhase {
        string name;
        uint32_t src = 0;
        uint32_t dst = 0;
        MpRailFlowHandle flow;
    };
    struct ServerForwardState {
        flowid_t logical_flow_id = 0;
        uint64_t bytes = 0;
        vector<ServerForwardPhase> phases;
        function<void()> final_callback;
    };

    auto create_server_forward = [&] (
            uint32_t src,
            uint32_t dst,
            uint64_t bytes,
            optional<flowid_t> requested_flow_id,
            simtime_picosec start_time,
            const MpRailRouteSpec& route,
            function<void()> final_callback,
            Trigger* start_trigger,
            Trigger* send_done_trigger,
            Trigger* recv_done_trigger) {
        mprail_topology->validate_route_spec(src, dst, route);

        struct PhaseDescription {
            string name;
            uint32_t src;
            uint32_t dst;
            bool is_fabric;
        };
        vector<PhaseDescription> descriptions;
        if (src != route.src_relay) {
            descriptions.push_back({"src_local", src, route.src_relay, false});
        }
        descriptions.push_back(
                {"fabric", route.src_relay, route.dst_relay, true});
        if (route.dst_relay != dst) {
            descriptions.push_back({"dst_local", route.dst_relay, dst, false});
        }

        auto state = make_shared<ServerForwardState>();
        state->bytes = bytes;
        state->final_callback = move(final_callback);
        size_t fabric_index = 0;
        for (size_t index = 0; index < descriptions.size(); ++index) {
            const PhaseDescription& description = descriptions[index];
            if (description.is_fabric) {
                fabric_index = index;
            }
            optional<flowid_t> phase_flow_id;
            if (description.is_fabric && requested_flow_id.has_value()) {
                phase_flow_id = requested_flow_id.value();
            }
            const simtime_picosec phase_start = index == 0
                    ? start_time : TRIGGER_START;
            MpRailFlowHandle phase_flow = create_mprail_flow(
                    description.src,
                    description.dst,
                    bytes,
                    phase_flow_id,
                    phase_start,
                    "ServerForward_" + description.name + "_"
                            + ntoa(description.src) + "_" + ntoa(description.dst),
                    nullptr,
                    {});
            state->phases.push_back(ServerForwardPhase{
                    description.name, description.src, description.dst, phase_flow});
        }
        state->logical_flow_id = requested_flow_id.has_value()
                ? requested_flow_id.value()
                : state->phases.at(fabric_index).flow.source->flowId();

        cout << "SERVER_FORWARD_BEGIN flow=" << state->logical_flow_id
             << " src=" << src
             << " src_relay=" << route.src_relay
             << " dst_relay=" << route.dst_relay
             << " dst=" << dst
             << " bytes=" << bytes
             << " phases=" << state->phases.size() << endl;

        for (size_t index = 0; index < state->phases.size(); ++index) {
            ServerForwardPhase& phase = state->phases[index];
            phase.flow.source->setStartCallback([state, index]() {
                const ServerForwardPhase& current = state->phases.at(index);
                cout << "SERVER_FORWARD_PHASE_START flow="
                     << state->logical_flow_id
                     << " phase=" << current.name
                     << " src=" << current.src
                     << " dst=" << current.dst
                     << " time_us=" << timeAsUs(eventlist.now()) << endl;
            });
            phase.flow.source->setCompletionCallback([state, index]() {
                const ServerForwardPhase& current = state->phases.at(index);
                cout << "SERVER_FORWARD_PHASE_DONE flow="
                     << state->logical_flow_id
                     << " phase=" << current.name
                     << " time_us=" << timeAsUs(eventlist.now()) << endl;
                if (index + 1 < state->phases.size()) {
                    eventlist.triggerIsPending(
                            *state->phases.at(index + 1).flow.source);
                    return;
                }
                cout << "SERVER_FORWARD_DONE flow=" << state->logical_flow_id
                     << " time_us=" << timeAsUs(eventlist.now()) << endl;
                if (state->final_callback) {
                    state->final_callback();
                }
            });
        }

        ServerForwardPhase& first_phase = state->phases.front();
        ServerForwardPhase& last_phase = state->phases.back();
        if (start_trigger) {
            start_trigger->add_target(*first_phase.flow.source);
        }
        if (send_done_trigger) {
            last_phase.flow.source->setEndTrigger(*send_done_trigger);
        }
        if (recv_done_trigger) {
            last_phase.flow.sink->setEndTrigger(*recv_done_trigger);
        }
        return state;
    };

    map<flowid_t, pair<UecSrc*, UecSink*>> flowmap;
    map<flowid_t, UecPdcSes*> flow_pdc_map;
    unique_ptr<OxcDagManager> oxc_dag_manager;
    if (!dag_file.empty()) {
        if (!all_conns->empty()) {
            cerr << "-dag requires a traffic matrix with zero static connections"
                 << endl;
            exit(1);
        }

        try {
            oxc_dag_manager = make_unique<OxcDagManager>(eventlist, no_of_nodes);
            oxc_dag_manager->load_from_file(dag_file);
            oxc_dag_manager->validate_network_tasks([&](const OxcDagTask& task) {
                if (!task.route.has_value()) {
                    return;
                }
                if (!mprail_topology) {
                    throw invalid_argument(
                            "DAG route fields are supported only by MpRail topology");
                }
                mprail_topology->validate_route_spec(
                        static_cast<uint32_t>(task.src_rank),
                        static_cast<uint32_t>(task.dst_rank),
                        task.route.value());
            });
        } catch (const exception& e) {
            cerr << "Failed to load DAG: " << e.what() << endl;
            exit(1);
        }

        oxc_dag_manager->set_network_launcher([&](const OxcDagTask& task) {
            const uint32_t src = static_cast<uint32_t>(task.src_rank);
            const uint32_t dst = static_cast<uint32_t>(task.dst_rank);
            OxcDagManager* manager = oxc_dag_manager.get();
            if (mprail_topology) {
                if (task.route.has_value()
                        && task.route->mode == MpRailRouteMode::SERVER_FORWARD) {
                    create_server_forward(
                            src,
                            dst,
                            task.transfer_bytes,
                            optional<flowid_t>(task.id),
                            eventlist.now(),
                            task.route.value(),
                            [manager, task_id = task.id]() {
                                manager->notify_network_task_done(task_id);
                            },
                            nullptr,
                            nullptr,
                            nullptr);
                } else {
                    const MpRailRouteSpec* explicit_route = task.route.has_value()
                            ? &task.route.value() : nullptr;
                    create_mprail_flow(
                            src,
                            dst,
                            task.transfer_bytes,
                            optional<flowid_t>(task.id),
                            eventlist.now(),
                            "Dag_" + ntoa(src) + "_" + ntoa(dst)
                                    + "_" + ntoa(task.id),
                            explicit_route,
                            [manager, task_id = task.id]() {
                                manager->notify_network_task_done(task_id);
                            });
                }
                return;
            }
            if (task.route.has_value()) {
                throw invalid_argument(
                        "DAG route fields are supported only by MpRail topology");
            }
            unique_ptr<UecMultipath> mp;
            if (load_balancing_algo == BITMAP) {
                mp = make_unique<UecMpBitmap>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == REPS || load_balancing_algo == REPS_LEGACY) {
                mp = make_unique<UecMpRepsLegacy>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == FREEZING) {
                mp = make_unique<UecMpReps>(path_entropy_size, UecSrc::_debug, !disable_trim);
            } else if (load_balancing_algo == OBLIVIOUS) {
                mp = make_unique<UecMpOblivious>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == MIXED) {
                mp = make_unique<UecMpMixed>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == ECMP) {
                mp = make_unique<UecMpEcmp>(path_entropy_size, UecSrc::_debug);
            } else {
                throw logic_error("unsupported UEC multipath algorithm for DAG");
            }

            auto* dag_src = new UecSrc(traffic_logger, eventlist, move(mp), *nics.at(src), ports);
            dag_src->setFlowId(task.id);
            dag_src->setFlowsize(task.transfer_bytes);
            dag_src->setSrc(src);
            dag_src->setDst(dst);
            dag_src->setName("Dag_" + ntoa(src) + "_" + ntoa(dst)
                             + "_" + ntoa(task.id));
            logfile.writeName(*dag_src);

            UecSink* dag_snk = nullptr;
            if (receiver_driven) {
                dag_snk = new UecSink(
                        NULL, pacers.at(dst).get(), *nics.at(dst), ports);
            } else {
                dag_snk = new UecSink(
                        NULL, linkspeed, 1.1,
                        UecBasePacket::unquantize(UecSink::_credit_per_pull),
                        eventlist, *nics.at(dst), ports);
            }
            dag_snk->setFlowId(task.id);
            dag_snk->setSrc(src);
            ((DataReceiver*)dag_snk)->setName(
                    "Dag_sink_" + ntoa(src) + "_" + ntoa(dst)
                    + "_" + ntoa(task.id));
            logfile.writeName(*(DataReceiver*)dag_snk);

            const simtime_picosec base_rtt = mprail_mode
                    ? calculate_mprail_pair_rtt(mprail_cfg, src, dst)
                    : calculate_oxc_pair_rtt(
                            oxc_ocs_cfg, src, dst, linkspeed,
                            local_linkspeed, local_latency);
            if (receiver_driven) {
                dag_src->initRccc(cwnd_b, enable_accurate_base_rtt
                        ? base_rtt
                        : network_max_unloaded_rtt);
            }
            if (sender_driven) {
                dag_src->initNscc(cwnd_b, enable_accurate_base_rtt
                        ? base_rtt
                        : network_max_unloaded_rtt);
            }
            if (event_logger) {
                dag_src->logFlowEvents(*event_logger);
            }
            if (UecSink::_model_pcie) {
                dag_snk->setPCIeModel(pcie_models.at(dst));
            }
            if (UecSink::_oversubscribed_cc) {
                dag_snk->setOversubscribedCC(oversubscribed_ccs.at(dst));
            }

            dag_src->setCompletionCallback([manager, task_id = task.id]() {
                manager->notify_network_task_done(task_id);
            });
            if (mprail_topology) {
                mprail_topology->connect_endpoints(
                        src, dst, *dag_src, *dag_snk, eventlist.now());
            } else {
                oxc_topology->connect_endpoints(
                        src, dst, *dag_src, *dag_snk, eventlist.now());
            }
            if (sink_logger) {
                sink_logger->monitorSink(dag_snk);
            }
            const uint32_t server_size = mprail_mode
                    ? mprail_cfg.gpus_per_server
                    : oxc_ocs_cfg.ranks_per_tray;
            if (src / server_size == dst / server_size && src != dst) {
                ++local_flow_count;
            } else {
                ++fabric_flow_count;
            }
            uec_srcs.push_back(dag_src);
        });
    }
    if (planes != 1) {
        cout << "Multi-plane mode enabled: " << planes
             << " independent topology planes, one UEC port per plane."
             << endl;
        if (goal_filename.size() > 0) {
            cerr << "GOAL/ATLAHS mode does not support -planes > 1 yet; "
                 << "traffic-matrix mode must be used for multi-plane validation." << endl;
            exit(1);
        }
    }

    // ATLAHS
    LogSimInterface *lgs = NULL;

    if (goal_filename.size() > 0) {
        if (fabric_mode) {
            cerr << "GOAL/ATLAHS mode requires the FatTreeTopology API and is not supported with the selected fabric topology"
                 << endl;
            exit(1);
        }
        AtlahsHtsimApi *api = new AtlahsHtsimApi();
        api->setTopology(topo[0].get());
        api->cwnd_b = cwnd;
        api->setEventList(&eventlist);
        api->setComputeEvent(new ComputeEvent(eventlist));
        api->setNullEvent(new NullEvent(eventlist));
        lgs = new LogSimInterface(NULL, traffic_logger, eventlist, topo[0].get(), nullptr);
        lgs->htsim_api = api;
        api->setLogSimInterface(lgs);
        lgs->set_protocol(UEC_PROTOCOL);
        lgs->htsim_api->linkspeed = linkspeed;
        api->print_stats_flows = LogSimInterface::print_stats_flows;

        // Build a factory for creating per-flow multipath instances
        switch (load_balancing_algo) {
            case BITMAP:
                api->setMultipathFactory([path_entropy_size]() {
                    return std::make_unique<UecMpBitmap>(path_entropy_size, UecSrc::_debug);
                });
                break;
            case REPS:
            case REPS_LEGACY:
                api->setMultipathFactory([path_entropy_size]() {
                    return std::make_unique<UecMpRepsLegacy>(path_entropy_size, UecSrc::_debug);
                });
                break;
            case FREEZING:
                api->setMultipathFactory([path_entropy_size, disable_trim]() {
                    return std::make_unique<UecMpReps>(path_entropy_size, UecSrc::_debug, !disable_trim);
                });
                break;
            case OBLIVIOUS:
                api->setMultipathFactory([path_entropy_size]() {
                    return std::make_unique<UecMpOblivious>(path_entropy_size, UecSrc::_debug);
                });
                break;
            case ECMP:
                api->setMultipathFactory([path_entropy_size]() {
                    return std::make_unique<UecMpEcmp>(path_entropy_size, UecSrc::_debug);
                });
                break;
            case MIXED:
                api->setMultipathFactory([path_entropy_size, disable_trim]() {
                    return std::make_unique<UecMpMixed>(path_entropy_size, UecSrc::_debug);
                });
                break;
            default:
                cout << "ERROR: Failed to set multipath algorithm, abort." << endl;
                abort();
        }

        // Calculate G in cycles
        double linkSpeedBytesPerSec = (linkspeed/1000000000 * 1e9) / 8.0;
        lgs->htsim_api->htsim_G  = 1e9 / linkSpeedBytesPerSec;

        printf("<HTSIM> G %f\n", lgs->htsim_api->htsim_G);

        lgs->htsim_api->total_nodes = no_of_nodes;
        lgs->htsim_api->Setup();
        printf("Started LGS\n");
        
        start_lgs(goal_filename, *lgs);
        printf("Iteration Terminated\n");
    }
    

    if (goal_filename.size() > 0) {
        printf("Finished all\n");
        fflush(stdout);
        return 0;
    }

    for (size_t c = 0; c < all_conns->size(); c++){
        connection* crt = all_conns->at(c);
        int src = crt->src;
        int dest = crt->dst;
        if (crt->route.has_value()) {
            if (!mprail_topology) {
                cerr << "CM route fields are supported only by MpRail topology" << endl;
                exit(1);
            }
            if (conn_reuse || crt->msgid.has_value()) {
                cerr << "CM route fields do not support -conn_reuse or msg" << endl;
                exit(1);
            }
            if (crt->size <= 0) {
                cerr << "CM routed flow requires a positive size" << endl;
                exit(1);
            }
            try {
                mprail_topology->validate_route_spec(
                        static_cast<uint32_t>(src),
                        static_cast<uint32_t>(dest),
                        crt->route.value());
            } catch (const exception& error) {
                cerr << "Invalid CM route: " << error.what() << endl;
                exit(1);
            }

            const optional<flowid_t> requested_flow_id = crt->flowid
                    ? optional<flowid_t>(crt->flowid) : optional<flowid_t>();
            Trigger* start_trigger = crt->trigger
                    ? conns->getTrigger(crt->trigger, eventlist) : nullptr;
            Trigger* send_done_trigger = crt->send_done_trigger
                    ? conns->getTrigger(crt->send_done_trigger, eventlist) : nullptr;
            Trigger* recv_done_trigger = crt->recv_done_trigger
                    ? conns->getTrigger(crt->recv_done_trigger, eventlist) : nullptr;

            if (crt->route->mode == MpRailRouteMode::SERVER_FORWARD) {
                create_server_forward(
                        src,
                        dest,
                        crt->size,
                        requested_flow_id,
                        crt->start,
                        crt->route.value(),
                        {},
                        start_trigger,
                        send_done_trigger,
                        recv_done_trigger);
            } else {
                MpRailFlowHandle routed_flow = create_mprail_flow(
                        src,
                        dest,
                        crt->size,
                        requested_flow_id,
                        crt->start,
                        "Explicit_" + ntoa(src) + "_" + ntoa(dest),
                        &crt->route.value(),
                        {});
                if (start_trigger) {
                    start_trigger->add_target(*routed_flow.source);
                }
                if (send_done_trigger) {
                    routed_flow.source->setEndTrigger(*send_done_trigger);
                }
                if (recv_done_trigger) {
                    routed_flow.sink->setEndTrigger(*recv_done_trigger);
                }
            }
            continue;
        }
        uint32_t route_tray_size = oxc_topology
                ? oxc_ocs_cfg.ranks_per_tray
                : local_tray_size;
        bool local_tray_route = route_tray_size > 0
                                && src != dest
                                && (src / (int)route_tray_size) == (dest / (int)route_tray_size);

        if (!conn_reuse and crt->msgid.has_value()) {
            cout << "msg keyword can only be used when conn_reuse is enabled.\n";
            abort();
        }

        assert(planes > 0);
        simtime_picosec base_rtt_bw_two_points = network_max_unloaded_rtt;
        if (mprail_mode) {
            base_rtt_bw_two_points = calculate_mprail_pair_rtt(
                    mprail_cfg, src, dest);
        } else if (oxc_mode) {
            base_rtt_bw_two_points = calculate_oxc_pair_rtt(
                    oxc_ocs_cfg, src, dest, linkspeed, local_linkspeed, local_latency);
        } else {
            simtime_picosec transmission_delay = (Packet::data_packet_size() * 8 / speedAsGbps(linkspeed) * topo_cfg->get_diameter() * 1000)
                                                 + (UecBasePacket::get_ack_size() * 8 / speedAsGbps(linkspeed) * topo_cfg->get_diameter() * 1000);
            base_rtt_bw_two_points = 2 * topo_cfg->get_two_point_diameter_latency(src, dest) + transmission_delay;
        }
        if (!fabric_mode && local_tray_route) {
            simtime_picosec local_transmission_delay =
                (Packet::data_packet_size() * 8 / speedAsGbps(local_linkspeed) * 1000)
                + (UecBasePacket::get_ack_size() * 8 / speedAsGbps(local_linkspeed) * 1000);
            base_rtt_bw_two_points = 2 * local_latency + local_transmission_delay;
        }

        //cout << "Connection " << crt->src << "->" <<crt->dst << " starting at " << crt->start << " size " << crt->size << endl;

        if (!conn_reuse 
            || (crt->flowid and flowmap.find(crt->flowid) == flowmap.end())) {
            const bool mprail_local_flow = mprail_topology && local_tray_route;
            UecNIC& source_nic = mprail_local_flow
                    ? *mprail_local_nics.at(src) : *nics.at(src);
            UecNIC& destination_nic = mprail_local_flow
                    ? *mprail_local_nics.at(dest) : *nics.at(dest);
            const uint32_t flow_ports = mprail_local_flow ? 1 : ports;
            const linkspeed_bps flow_linkspeed = mprail_local_flow
                    ? local_linkspeed : linkspeed;
            unique_ptr<UecMultipath> mp = nullptr;
            if (load_balancing_algo == BITMAP){
                mp = make_unique<UecMpBitmap>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == REPS || load_balancing_algo == REPS_LEGACY){
                mp = make_unique<UecMpRepsLegacy>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == FREEZING){
                mp = make_unique<UecMpReps>(path_entropy_size, UecSrc::_debug, !disable_trim);
            }else if (load_balancing_algo == OBLIVIOUS){
                mp = make_unique<UecMpOblivious>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == MIXED){
                mp = make_unique<UecMpMixed>(path_entropy_size, UecSrc::_debug);
            } else if (load_balancing_algo == ECMP){
                mp = make_unique<UecMpEcmp>(path_entropy_size, UecSrc::_debug);
            } else {
                cout << "ERROR: Failed to set multipath algorithm, abort." << endl;
                abort();
            }

            uec_src = new UecSrc(
                    traffic_logger, eventlist, move(mp), source_nic, flow_ports);

            if (crt->flowid) {
                uec_src->setFlowId(crt->flowid);
                assert(flowmap.find(crt->flowid) == flowmap.end()); // don't have dups
            }

            if (conn_reuse) {
                stringstream uec_src_dbg_tag;
                uec_src_dbg_tag << "flow_id " << uec_src->flowId();
                UecPdcSes* pdc = new UecPdcSes(uec_src, EventList::getTheEventList(), UecSrc::_mss, UecSrc::_hdr_size, uec_src_dbg_tag.str());
                uec_src->makeReusable(pdc);
                flow_pdc_map[uec_src->flowId()] = pdc;
            }

            if (receiver_driven)
                uec_snk = new UecSink(
                        NULL,
                        mprail_local_flow ? mprail_local_pacers.at(dest).get()
                                          : pacers.at(dest).get(),
                        destination_nic, flow_ports);
            else //each connection has its own pacer, so receiver driven mode does not kick in! 
                uec_snk = new UecSink(
                        NULL, flow_linkspeed, 1.1,
                        UecBasePacket::unquantize(UecSink::_credit_per_pull),
                        eventlist, destination_nic, flow_ports);

            flowmap[uec_src->flowId()] = { uec_src, uec_snk };

            if (crt->flowid) {
                uec_snk->setFlowId(crt->flowid);
            }

            // If cwnd is 0 initXXcc will set a sensible default value 
            if (receiver_driven) {
                // uec_src->setCwnd(cwnd*Packet::data_packet_size());
                // uec_src->setMaxWnd(cwnd*Packet::data_packet_size());

                if (enable_accurate_base_rtt) {
                    uec_src->initRccc(cwnd_b, base_rtt_bw_two_points);
                } else {
                    uec_src->initRccc(cwnd_b, network_max_unloaded_rtt);
                }
            }

            if (sender_driven) {
                if (enable_accurate_base_rtt) {
                    uec_src->initNscc(cwnd_b, base_rtt_bw_two_points);
                } else {
                    uec_src->initNscc(cwnd_b, network_max_unloaded_rtt);
                }
            }
            uec_srcs.push_back(uec_src);
            uec_src->setSrc(src);
            uec_src->setDst(dest);

            if (log_flow_events) {
                uec_src->logFlowEvents(*event_logger);
            }
            

            uec_src->setName("Uec_" + ntoa(src) + "_" + ntoa(dest));
            logfile.writeName(*uec_src);
            uec_snk->setSrc(src);

            if (UecSink::_model_pcie && !mprail_local_flow){
                uec_snk->setPCIeModel(pcie_models[dest]);
            }
                            
            if (UecSink::_oversubscribed_cc){
                uec_snk->setOversubscribedCC(oversubscribed_ccs[dest]);
            }

            ((DataReceiver*)uec_snk)->setName("Uec_sink_" + ntoa(src) + "_" + ntoa(dest));
            logfile.writeName(*(DataReceiver*)uec_snk);

            if (!conn_reuse) {
                if (crt->size>0){
                    uec_src->setFlowsize(crt->size);
                }

                if (crt->trigger) {
                    Trigger* trig = conns->getTrigger(crt->trigger, eventlist);
                    trig->add_target(*uec_src);
                }

                if (crt->send_done_trigger) {
                    Trigger* trig = conns->getTrigger(crt->send_done_trigger, eventlist);
                    uec_src->setEndTrigger(*trig);
                }

                if (crt->recv_done_trigger) {
                    Trigger* trig = conns->getTrigger(crt->recv_done_trigger, eventlist);
                    uec_snk->setEndTrigger(*trig);
                }
            } else {
                assert(crt->size > 0);

                optional<simtime_picosec> start_ts = {};
                if (crt->start != TRIGGER_START) {
                    start_ts.emplace(timeFromUs((uint32_t)crt->start));
                } 

                UecPdcSes* pdc = flow_pdc_map.find(crt->flowid)->second;
                UecMsg* msg = pdc->enque(crt->size, start_ts, true);

                if (crt->trigger) {
                    Trigger* trig = conns->getTrigger(crt->trigger, eventlist);
                    trig->add_target(*msg);
                }

                if (crt->send_done_trigger) {
                    Trigger* trig = conns->getTrigger(crt->send_done_trigger, eventlist);
                    msg->setTrigger(UecMsg::MsgStatus::SentLast, trig);
                }

                if (crt->recv_done_trigger) {
                    Trigger* trig = conns->getTrigger(crt->recv_done_trigger, eventlist);
                    uec_snk->setEndTrigger(*trig);
                    msg->setTrigger(UecMsg::MsgStatus::RecvdLast, trig);
                }
            }

            //uec_snk->set_priority(crt->priority);
                            
            if (mprail_topology) {
                if (local_tray_route) {
                    local_flow_count++;
                } else {
                    fabric_flow_count++;
                }
                mprail_topology->connect_endpoints(
                        src, dest, *uec_src, *uec_snk, crt->start);
            } else if (oxc_topology) {
                if (local_tray_route) {
                    local_flow_count++;
                } else {
                    fabric_flow_count++;
                }
                oxc_topology->connect_endpoints(
                        src, dest, *uec_src, *uec_snk, crt->start);
            } else if (local_tray_route) {
                local_flow_count++;
                Route* routeout = make_direct_local_route(src, dest, 0, uec_snk->getPort(0),
                                                          local_linkspeed, queuesize,
                                                          local_latency, "DATA");
                Route* routeback = make_direct_local_route(dest, src, 0, uec_src->getPort(0),
                                                           local_linkspeed, queuesize,
                                                           local_latency, "ACK");
                routeout->set_reverse(routeback);
                routeback->set_reverse(routeout);
                for (uint32_t p = 0; p < ports; p++) {
                    // All UEC ports share the same local route. This keeps intra-tray
                    // traffic on one simulated server-internal link while still giving
                    // control packets a valid route if the NIC selects any port.
                    uec_src->connectPort(p, *routeout, *routeback, *uec_snk, crt->start);
                }
            } else {
                for (uint32_t p = 0; p < planes; p++) {
                    switch (route_strategy) {
                    case ECMP_FIB:
                    case ECMP_FIB_ECN:
                    case REACTIVE_ECN:
                        {
                            Route* srctotor = new Route();
                            srctotor->push_back(topo[p]->queues_ns_nlp[src][topo_cfg->HOST_POD_SWITCH(src)][0]);
                            srctotor->push_back(topo[p]->pipes_ns_nlp[src][topo_cfg->HOST_POD_SWITCH(src)][0]);
                            srctotor->push_back(topo[p]->queues_ns_nlp[src][topo_cfg->HOST_POD_SWITCH(src)][0]->getRemoteEndpoint());

                            Route* dsttotor = new Route();
                            dsttotor->push_back(topo[p]->queues_ns_nlp[dest][topo_cfg->HOST_POD_SWITCH(dest)][0]);
                            dsttotor->push_back(topo[p]->pipes_ns_nlp[dest][topo_cfg->HOST_POD_SWITCH(dest)][0]);
                            dsttotor->push_back(topo[p]->queues_ns_nlp[dest][topo_cfg->HOST_POD_SWITCH(dest)][0]->getRemoteEndpoint());

                            uec_src->connectPort(p, *srctotor, *dsttotor, *uec_snk, crt->start);
                            //uec_src->setPaths(path_entropy_size);
                            //uec_snk->setPaths(path_entropy_size);

                            //register src and snk to receive packets from their respective TORs. 
                            assert(topo[p]->switches_lp[topo_cfg->HOST_POD_SWITCH(src)]);
                            assert(topo[p]->switches_lp[topo_cfg->HOST_POD_SWITCH(src)]);
                            topo[p]->switches_lp[topo_cfg->HOST_POD_SWITCH(src)]->addHostPort(src,uec_snk->flowId(),uec_src->getPort(p));
                            topo[p]->switches_lp[topo_cfg->HOST_POD_SWITCH(dest)]->addHostPort(dest,uec_src->flowId(),uec_snk->getPort(p));
                            break;
                        }
                    default:
                        abort();
                    }
                }
            }

            // set up the triggers
            // xxx

            if (log_sink) {
                sink_logger->monitorSink(uec_snk);
            }
        } else {
            // Use existing connection for this message
            assert(crt->msgid.has_value());

            UecPdcSes* pdc = flow_pdc_map.find(crt->flowid)->second;
            uec_src = nullptr;
            uec_snk = nullptr;

            optional<simtime_picosec> start_ts = {};
            if (crt->start != TRIGGER_START) {
                start_ts.emplace(timeFromUs((uint32_t)crt->start));
            } 

            UecMsg* msg = pdc->enque(crt->size, start_ts, true);

            if (crt->trigger) {
                Trigger* trig = conns->getTrigger(crt->trigger, eventlist);
                trig->add_target(*msg);
            }

            if (crt->send_done_trigger) {
                Trigger* trig = conns->getTrigger(crt->send_done_trigger, eventlist);
                msg->setTrigger(UecMsg::MsgStatus::SentLast, trig);
            }

            if (crt->recv_done_trigger) {
                Trigger* trig = conns->getTrigger(crt->recv_done_trigger, eventlist);
                msg->setTrigger(UecMsg::MsgStatus::RecvdLast, trig);
            }
        }
    }

    if (local_tray_size > 0) {
        cout << "Server-local direct flows " << local_flow_count
             << " / " << uec_srcs.size() << endl;
    }
    if (fabric_mode) {
        cout << "Fabric flows " << fabric_flow_count
             << " / " << uec_srcs.size() << endl;
    }

    const uint64_t expected_flows = uec_srcs.size()
            + (oxc_dag_manager ? oxc_dag_manager->network_task_count() : 0);
    UecSrc::initProgressLogging(expected_flows);
    if (oxc_dag_manager) {
        oxc_dag_manager->start();
    }
    // Record the setup
    int pktsize = Packet::data_packet_size();
    logfile.write("# pktsize=" + ntoa(pktsize) + " bytes");
    logfile.write("# hostnicrate = " + ntoa(linkspeed/1000000) + " Mbps");
    //logfile.write("# corelinkrate = " + ntoa(HOST_NIC*CORE_TO_HOST) + " pkt/sec");
    //logfile.write("# buffer = " + ntoa((double) (queues_na_ni[0][1]->_maxsize) / ((double) pktsize)) + " pkt");
    
    // GO!
    cout << "Starting simulation" << endl;
    while (eventlist.doNextEvent()) {
    }

    if (oxc_dag_manager && !oxc_dag_manager->finished()) {
        cerr << "DAG did not finish before the simulation stopped" << endl;
        return 2;
    }

    cout << "Done" << endl;
    int new_pkts = 0, rtx_pkts = 0, bounce_pkts = 0, rts_pkts = 0, ack_pkts = 0, nack_pkts = 0, pull_pkts = 0, sleek_pkts = 0;
    for (size_t ix = 0; ix < uec_srcs.size(); ix++) {
        const struct UecSrc::Stats& s = uec_srcs[ix]->stats();
        new_pkts += s.new_pkts_sent;
        rtx_pkts += s.rtx_pkts_sent;
        rts_pkts += s.rts_pkts_sent;
        bounce_pkts += s.bounces_received;
        ack_pkts += s.acks_received;
        nack_pkts += s.nacks_received;
        pull_pkts += s.pulls_received;
        sleek_pkts += s._sleek_counter;
    }
    cout << "New: " << new_pkts << " Rtx: " << rtx_pkts << " RTS: " << rts_pkts << " Bounced: " << bounce_pkts << " ACKs: " << ack_pkts << " NACKs: " << nack_pkts << " Pulls: " << pull_pkts << " sleek_pkts: " << sleek_pkts << endl;
    /*
    list <const Route*>::iterator rt_i;
    int counts[10]; int hop;
    for (int i = 0; i < 10; i++)
        counts[i] = 0;
    cout << "route count: " << routes.size() << endl;
    for (rt_i = routes.begin(); rt_i != routes.end(); rt_i++) {
        const Route* r = (*rt_i);
        //print_route(*r);
#ifdef PRINTPATHS
        cout << "Path:" << endl;
#endif
        hop = 0;
        for (int i = 0; i < r->size(); i++) {
            PacketSink *ps = r->at(i); 
            CompositeQueue *q = dynamic_cast<CompositeQueue*>(ps);
            if (q == 0) {
#ifdef PRINTPATHS
                cout << ps->nodename() << endl;
#endif
            } else {
#ifdef PRINTPATHS
                cout << q->nodename() << " " << q->num_packets() << "pkts " 
                     << q->num_headers() << "hdrs " << q->num_acks() << "acks " << q->num_nacks() << "nacks " << q->num_stripped() << "stripped"
                     << endl;
#endif
                counts[hop] += q->num_stripped();
                hop++;
            }
        } 
#ifdef PRINTPATHS
        cout << endl;
#endif
    }
    for (int i = 0; i < 10; i++)
        cout << "Hop " << i << " Count " << counts[i] << endl;
    */  

    return EXIT_SUCCESS;
}
