// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "mprail_route_spec.h"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

uint32_t parse_uint(const std::string& value, const std::string& context) {
    try {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()
                || parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("outside uint32 range");
        }
        return static_cast<uint32_t>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(context + ": invalid non-negative integer '"
                                    + value + "'");
    }
}

std::vector<std::string> split(const std::string& token, char separator) {
    std::vector<std::string> fields;
    std::istringstream input(token);
    std::string field;
    while (std::getline(input, field, separator)) {
        fields.push_back(field);
    }
    return fields;
}

uint32_t parse_prefixed(
        const std::string& field,
        char prefix,
        const std::string& context) {
    if (field.size() < 2 || field[0] != prefix) {
        throw std::invalid_argument(context + ": expected '"
                                    + std::string(1, prefix) + "N', got '"
                                    + field + "'");
    }
    return parse_uint(field.substr(1), context);
}

MpRailRouteNode parse_node(const std::string& token, const std::string& context) {
    const std::vector<std::string> fields = split(token, ':');
    MpRailRouteNode node;
    if (fields.size() == 2 && fields[0] == "rank") {
        node.type = MpRailRouteNodeType::RANK;
        node.rank = parse_uint(fields[1], context);
        return node;
    }
    if ((fields.size() == 3 || fields.size() == 4) && fields[0] == "l0") {
        node.type = MpRailRouteNodeType::L0;
        node.rail = parse_prefixed(fields[1], 'r', context);
        node.plane = parse_prefixed(fields[2], 'p', context);
        if (fields.size() == 4) {
            node.egress_bundle = parse_prefixed(fields[3], 'b', context);
        }
        return node;
    }
    if ((fields.size() == 3 || fields.size() == 4) && fields[0] == "l1") {
        node.type = MpRailRouteNodeType::L1;
        node.plane = parse_prefixed(fields[1], 'p', context);
        node.spine = parse_prefixed(fields[2], 's', context);
        if (fields.size() == 4) {
            node.egress_bundle = parse_prefixed(fields[3], 'b', context);
        }
        return node;
    }
    throw std::invalid_argument(context + ": malformed explicit route node '"
                                + token + "'");
}

std::string node_to_string(const MpRailRouteNode& node) {
    std::ostringstream output;
    switch (node.type) {
    case MpRailRouteNodeType::RANK:
        output << "rank:" << node.rank;
        break;
    case MpRailRouteNodeType::L0:
        output << "l0:r" << node.rail << ":p" << node.plane;
        break;
    case MpRailRouteNodeType::L1:
        output << "l1:p" << node.plane << ":s" << node.spine;
        break;
    }
    if (node.egress_bundle.has_value()) {
        output << ":b" << node.egress_bundle.value();
    }
    return output.str();
}

}  // namespace

MpRailRouteSpec parse_mprail_route_spec(
        const std::vector<std::string>& tokens,
        const std::string& context) {
    if (tokens.empty()) {
        throw std::invalid_argument(context + ": missing route mode");
    }

    MpRailRouteSpec spec;
    if (tokens[0] == "explicit") {
        spec.mode = MpRailRouteMode::EXPLICIT;
        if (tokens.size() < 3) {
            throw std::invalid_argument(
                    context + ": explicit route requires at least two rank nodes");
        }
        for (size_t index = 1; index < tokens.size(); ++index) {
            spec.explicit_nodes.push_back(parse_node(tokens[index], context));
        }
        return spec;
    }

    if (tokens[0] == "server_forward") {
        spec.mode = MpRailRouteMode::SERVER_FORWARD;
        bool saw_src_relay = false;
        bool saw_dst_relay = false;
        for (size_t index = 1; index < tokens.size(); ++index) {
            const std::vector<std::string> fields = split(tokens[index], ':');
            if (fields.size() != 2) {
                throw std::invalid_argument(context + ": malformed server_forward field '"
                                            + tokens[index] + "'");
            }
            if (fields[0] == "src_relay" && !saw_src_relay) {
                spec.src_relay = parse_uint(fields[1], context);
                saw_src_relay = true;
            } else if (fields[0] == "dst_relay" && !saw_dst_relay) {
                spec.dst_relay = parse_uint(fields[1], context);
                saw_dst_relay = true;
            } else {
                throw std::invalid_argument(context + ": unknown or repeated field '"
                                            + fields[0] + "'");
            }
        }
        if (!saw_src_relay || !saw_dst_relay) {
            throw std::invalid_argument(
                    context + ": server_forward requires src_relay and dst_relay");
        }
        return spec;
    }

    throw std::invalid_argument(context + ": unknown route mode '" + tokens[0] + "'");
}

std::string mprail_route_spec_to_string(const MpRailRouteSpec& spec) {
    std::ostringstream output;
    if (spec.mode == MpRailRouteMode::SERVER_FORWARD) {
        output << "server_forward src_relay:" << spec.src_relay
               << " dst_relay:" << spec.dst_relay;
        return output.str();
    }

    output << "explicit";
    for (const MpRailRouteNode& node : spec.explicit_nodes) {
        output << " " << node_to_string(node);
    }
    return output.str();
}
