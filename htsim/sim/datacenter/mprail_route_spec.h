// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MPRAIL_ROUTE_SPEC_H
#define MPRAIL_ROUTE_SPEC_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum class MpRailRouteMode {
    EXPLICIT,
    SERVER_FORWARD,
};

enum class MpRailRouteNodeType {
    RANK,
    L0,
    L1,
};

struct MpRailRouteNode {
    MpRailRouteNodeType type = MpRailRouteNodeType::RANK;
    uint32_t rank = 0;
    uint32_t rail = 0;
    uint32_t plane = 0;
    uint32_t spine = 0;
    std::optional<uint32_t> egress_bundle;
};

struct MpRailRouteSpec {
    MpRailRouteMode mode = MpRailRouteMode::EXPLICIT;
    std::vector<MpRailRouteNode> explicit_nodes;
    uint32_t src_relay = 0;
    uint32_t dst_relay = 0;
};

MpRailRouteSpec parse_mprail_route_spec(
        const std::vector<std::string>& tokens,
        const std::string& context);
std::string mprail_route_spec_to_string(const MpRailRouteSpec& spec);

#endif
