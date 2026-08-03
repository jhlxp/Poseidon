// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MPRAIL_TOPOLOGY_H
#define MPRAIL_TOPOLOGY_H

#include "mprail_switch.h"
#include "pipe.h"
#include "queue.h"
#include "topology.h"
#include "uec.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct MpRailTopologyConfig {
    uint32_t nodes = 0;
    uint32_t planes = 8;
    uint32_t gpus_per_server = 8;
    uint32_t servers_per_rail = 1;
    uint32_t l1_eps_per_plane = 1;
    uint32_t l0_l1_links_per_spine = 1;

    linkspeed_bps external_linkspeed = 100000000000ULL;
    linkspeed_bps local_linkspeed = 3200000000000ULL;
    mem_b queue_size = 0;
    bool enable_ecn = false;
    mem_b ecn_threshold = 0;
    simtime_picosec link_latency = 0;
    simtime_picosec local_latency = 0;
    simtime_picosec switch_latency = 0;
};

class MpRailTopology : public Topology {
public:
    MpRailTopology(const MpRailTopologyConfig& cfg, EventList& eventlist);
    ~MpRailTopology() override;

    vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) override;
    vector<uint32_t>* get_neighbours(uint32_t src) override;
    uint32_t no_of_nodes() const override { return _cfg.nodes; }

    void connect_endpoints(
            uint32_t src,
            uint32_t dst,
            UecSrc& uec_src,
            UecSink& uec_snk,
            simtime_picosec start_time);

    uint32_t server_count() const { return _server_count; }
    uint32_t rail_count() const { return _rail_count; }
    uint32_t l0_count() const { return _l0_count; }
    uint32_t l1_count() const { return _l1_count; }
    uint32_t planes() const { return _cfg.planes; }
    uint32_t gpus_per_server() const { return _cfg.gpus_per_server; }
    uint32_t rank_server(uint32_t rank) const;
    uint32_t rank_rail(uint32_t rank) const;
    uint32_t l0_id(uint32_t rail, uint32_t plane) const;
    uint32_t l1_id(uint32_t plane, uint32_t spine) const;

private:
    struct CachedLink {
        Queue* queue = nullptr;
        Pipe* pipe = nullptr;
    };

    MpRailTopologyConfig _cfg;
    EventList& _eventlist;
    uint32_t _server_count = 0;
    uint32_t _rail_count = 0;
    uint32_t _l0_count = 0;
    uint32_t _l1_count = 0;

    std::vector<MpRailSwitch*> _l0_switches;
    std::vector<MpRailSwitch*> _l1_switches;
    std::unordered_map<std::string, CachedLink> _links;
    std::unordered_set<std::string> _installed_routes;
    std::unordered_set<std::string> _installed_host_routes;

    void validate_config() const;
    void init_switches();
    uint32_t select_plane(uint32_t flowid, uint32_t src, uint32_t dst) const;
    uint32_t select_spine(uint32_t flowid, uint32_t src, uint32_t dst) const;
    uint32_t select_bundle(uint32_t flowid, uint32_t src, uint32_t dst) const;

    Route* make_initial_route(uint32_t rank, uint32_t l0, uint32_t plane);
    Route* make_local_route(
            uint32_t src, uint32_t dst, PacketSink* final_sink);
    Route* make_l0_to_host_route(
            uint32_t l0, uint32_t rank, uint32_t plane, PacketSink* final_sink);
    Route* make_switch_route(
            const std::string& src_name,
            const std::string& dst_name,
            uint32_t bundle,
            MpRailSwitch* src_switch,
            MpRailSwitch* dst_switch);

    CachedLink& get_or_create_link(
            const std::string& src_name,
            const std::string& dst_name,
            uint32_t bundle,
            linkspeed_bps speed,
            simtime_picosec latency,
            MpRailSwitch* src_switch,
            PacketSink* remote_endpoint);

    void install_l0_host_route(
            uint32_t l0,
            uint32_t dst_rank,
            int flowid,
            uint32_t plane,
            PacketSink* final_sink);
    void install_l0_flow_up_route(
            uint32_t l0,
            uint32_t dst_rank,
            int flowid,
            uint32_t l1,
            uint32_t bundle);
    void install_l1_flow_down_route(
            uint32_t l1,
            uint32_t dst_rank,
            int flowid,
            uint32_t l0,
            uint32_t bundle);

    static std::string host_src_name(uint32_t rank);
    static std::string host_dst_name(uint32_t rank);
    static std::string l0_name(uint32_t rail, uint32_t plane);
    static std::string l1_name(uint32_t plane, uint32_t spine);
};

#endif
