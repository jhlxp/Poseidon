// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef OXC_OCS_SPRAYPOINT_H
#define OXC_OCS_SPRAYPOINT_H

#include "oxc_ocs_graph.h"

#include <cstdint>
#include <string>
#include <vector>

struct OxcOcsSprayPointParams {
    uint32_t spray_p = 4;
    uint32_t spray_h = 2;
    int32_t spray_levels = -1;  // -1 means auto.
    uint32_t spray_seed = 42;
};

enum class OxcOcsSprayPointChoice {
    PACKET_RR,
    FLOW_HASH,
};

struct OxcOcsSprayPointDestinationState {
    uint32_t dst_group = 0;
    std::vector<uint32_t> dst_nodes;
    std::vector<std::vector<uint32_t>> waypoint_levels;
    std::vector<uint32_t> inner_ring;
    std::vector<uint32_t> outer_ring;
    std::vector<std::vector<uint32_t>> parents;
    std::vector<int32_t> node_level;
    std::vector<uint8_t> is_dst;
    std::vector<uint8_t> is_inner_ring;
    std::vector<uint8_t> is_outer_ring;
};

class OxcOcsSprayPointRouter {
public:
    OxcOcsSprayPointRouter(
            const OxcOcsGraph& graph,
            uint32_t groups,
            uint32_t l1_eps_per_plane,
            OxcOcsSprayPointParams params);

    const OxcOcsGraph& graph() const { return _graph; }
    uint32_t groups() const { return _groups; }
    uint32_t l1_eps_per_plane() const { return _l1_eps_per_plane; }
    uint32_t levels() const { return _levels; }
    const OxcOcsSprayPointParams& params() const { return _params; }

    const OxcOcsSprayPointDestinationState& state_for_group(uint32_t dst_group) const;
    std::vector<uint32_t> source_spray_next_hops(uint32_t current_template_node) const;
    const std::vector<uint32_t>& pointing_next_hops(
            uint32_t current_template_node,
            uint32_t dst_group) const;

    bool is_dst_group(uint32_t template_node, uint32_t dst_group) const;
    std::string role_for_node(uint32_t template_node, uint32_t dst_group) const;

    uint32_t choose_next_hop(
            uint32_t current_template_node,
            uint32_t dst_group,
            bool source_step,
            uint32_t flow_id,
            uint32_t path_id,
            uint32_t rr_counter,
            OxcOcsSprayPointChoice choice) const;

private:
    OxcOcsGraph _graph;
    uint32_t _groups;
    uint32_t _l1_eps_per_plane;
    OxcOcsSprayPointParams _params;
    uint32_t _levels;
    std::vector<OxcOcsSprayPointDestinationState> _states;

    uint32_t derive_levels() const;
    OxcOcsSprayPointDestinationState build_state(uint32_t dst_group) const;
    std::vector<uint32_t> pick(
            std::vector<uint32_t> items,
            uint32_t count,
            uint64_t tag,
            uint32_t dst_group,
            uint32_t level,
            uint32_t node) const;
    std::vector<uint32_t> shortest_next_hops(
            uint32_t node,
            const std::vector<int32_t>& dist,
            uint32_t count,
            uint64_t tag,
            uint32_t dst_group) const;
    std::vector<int32_t> multi_source_distances(const std::vector<uint32_t>& sources) const;
    uint64_t rank_key(uint32_t item, uint64_t tag, uint32_t dst_group, uint32_t level, uint32_t node) const;
};

void write_oxc_ocs_spraypoint_state_csv(
        const OxcOcsSprayPointRouter& router,
        const std::string& path);

#endif
