// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef OXC_TOPOLOGY_H
#define OXC_TOPOLOGY_H

#include "oxc_ocs_graph.h"
#include "oxc_ocs_ksp.h"
#include "oxc_ocs_spraypoint.h"
#include "oxc_switch.h"
#include "pipe.h"
#include "queue.h"
#include "topology.h"
#include "uec.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class OxcOcsMode {
    OFF,
    SPRAYPOINT,
    KSP,
};

enum class OxcOcsChoice {
    FLOW_HASH,
    PACKET_RR,
};

struct OxcTopologyConfig {
    uint32_t nodes = 0;
    uint32_t groups = 0;
    uint32_t ranks_per_group = 0;
    uint32_t ranks_per_tray = 8;
    uint32_t l1_planes = 1;
    uint32_t source_ports = 0;
    uint32_t l1_eps_per_l1_plane = 4;
    uint32_t ocs_degree = 0;
    uint32_t ocs_seed = 42;

    OxcOcsMode ocs_mode = OxcOcsMode::OFF;
    OxcOcsChoice ocs_choice = OxcOcsChoice::PACKET_RR;

    uint32_t spray_p = 4;
    uint32_t spray_h = 2;
    int32_t spray_levels = -1;
    uint32_t ksp_k = 8;
    uint32_t ksp_max_hops = 0;
    uint32_t ksp_seed = 42;
    uint32_t ksp_max_paths_per_pair = 100000;

    linkspeed_bps external_linkspeed = 100000000000ULL;
    linkspeed_bps local_linkspeed = 800000000000ULL;
    mem_b queue_size = 0;
    bool enable_ecn = false;
    mem_b ecn_threshold = 0;
    simtime_picosec link_latency = 0;
    simtime_picosec local_latency = 0;
    simtime_picosec switch_latency = 0;

    std::string route_plan_path;
};

class OxcTopology : public Topology {
public:
    OxcTopology(const OxcTopologyConfig& cfg, EventList& eventlist);
    ~OxcTopology() override;

    vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) override;
    vector<uint32_t>* get_neighbours(uint32_t src) override;
    uint32_t no_of_nodes() const override { return _cfg.nodes; }

    void connect_endpoints(
            uint32_t src,
            uint32_t dst,
            UecSrc& uec_src,
            UecSink& uec_snk,
            simtime_picosec start_time);

    Route* l1_special_next_hop(OxcSwitch* sw, Packet& pkt);
    static Route* l1_special_next_hop_thunk(OxcSwitch* sw, Packet& pkt, void* context);

    uint32_t rank_group(uint32_t rank) const;
    uint32_t rank_tray(uint32_t rank) const;
    uint32_t rank_l0(uint32_t rank, uint32_t plane) const;
    uint32_t l1_physical_id(uint32_t group, uint32_t plane, uint32_t eps) const;
    uint32_t l1_group(uint32_t l1_id) const;
    uint32_t l1_plane(uint32_t l1_id) const;
    uint32_t l1_eps(uint32_t l1_id) const;
    uint32_t l1_logical_node(uint32_t l1_id) const;
    uint32_t l1_from_logical_node(uint32_t logical_node, uint32_t coupled_member) const;

private:
    struct RoutePlanEntry {
        uint32_t flowid = 0;
        uint32_t src = 0;
        uint32_t dst = 0;
        int32_t src_l0_plane = -1;
        int32_t src_egress_rank = -1;
        int32_t src_l1_id = -1;
        int32_t dst_l1_id = -1;
        int32_t dst_l0_plane = -1;
        int32_t dst_ingress_rank = -1;
        std::vector<uint32_t> ocs_l1_path;
    };

    struct CachedLink {
        Queue* queue = nullptr;
        Pipe* pipe = nullptr;
    };

    OxcTopologyConfig _cfg;
    EventList& _eventlist;
    uint32_t _trays = 0;
    uint32_t _l0_count = 0;
    uint32_t _l1_count = 0;
    uint32_t _logical_nodes_per_group = 0;

    std::vector<OxcSwitch*> _l0_switches;
    std::vector<OxcSwitch*> _l1_switches;
    std::unordered_map<std::string, CachedLink> _links;
    std::unordered_set<std::string> _installed_routes;
    std::unordered_set<std::string> _installed_host_routes;
    std::unordered_map<uint32_t, RoutePlanEntry> _route_plan;

    OxcOcsGraph _ocs_graph;
    std::unique_ptr<OxcOcsSprayPointRouter> _spraypoint;
    std::unique_ptr<OxcOcsKspRouter> _ksp;

    void validate_config() const;
    void init_switches();
    void init_ocs();
    void load_route_plan();
    const RoutePlanEntry* route_plan_entry(uint32_t flowid, uint32_t src, uint32_t dst) const;
    const RoutePlanEntry* route_plan_entry(uint32_t flowid) const;
    uint32_t source_ports() const;

    Route* make_initial_route(uint32_t rank, uint32_t l0, uint32_t plane);
    Route* make_initial_route_via_local(uint32_t src, uint32_t egress_rank, uint32_t l0, uint32_t plane);
    Route* make_local_route(uint32_t src, uint32_t dst, uint32_t bundle, PacketSink* final_sink);
    Route* make_l0_to_host_route(uint32_t l0, uint32_t rank, uint32_t plane, PacketSink* final_sink);
    Route* make_l0_to_host_via_local_route(
            uint32_t l0,
            uint32_t ingress_rank,
            uint32_t final_rank,
            uint32_t plane,
            PacketSink* final_sink);
    Route* make_switch_route(
            const std::string& src_name,
            const std::string& dst_name,
            uint32_t bundle,
            linkspeed_bps speed,
            simtime_picosec latency,
            OxcSwitch* src_switch,
            PacketSink* dst_sink);

    CachedLink& get_or_create_link(
            const std::string& src_name,
            const std::string& dst_name,
            uint32_t bundle,
            linkspeed_bps speed,
            simtime_picosec latency,
            OxcSwitch* src_switch,
            PacketSink* remote_endpoint);

    void install_routes_for_destination(uint32_t dst_rank, int dst_flowid, UecSinkPort* dst_port);
    void install_l0_up_routes(uint32_t l0, uint32_t dst_rank);
    void install_l1_down_routes(uint32_t dst_rank);
    void install_l0_host_route(uint32_t l0, uint32_t dst_rank, int flowid, uint32_t plane, PacketSink* final_sink);
    void install_l0_host_route_via_ingress(
            uint32_t l0,
            uint32_t dst_rank,
            int flowid,
            uint32_t ingress_rank,
            uint32_t plane,
            PacketSink* final_sink);
    void install_l0_flow_route(uint32_t l0, uint32_t dst_rank, int flowid, uint32_t dst_l1);
    void install_l1_flow_down_route(uint32_t l1, uint32_t dst_rank, int flowid, uint32_t dst_l0);

    Route* l1_to_l1_route(uint32_t src_l1, uint32_t dst_l1, uint32_t bundle);

    std::string route_key(const std::string& prefix, uint32_t sw, uint32_t dst, uint32_t next) const;
    std::string host_route_key(uint32_t l0, uint32_t dst, int flowid) const;

    static std::string host_src_name(uint32_t rank);
    static std::string host_dst_name(uint32_t rank);
    static std::string l0_name(uint32_t l0);
    static std::string l1_name(uint32_t l1);
};

#endif
