// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "mprail_topology.h"

#include "ecnqueue.h"

#include <iostream>
#include <stdexcept>

using namespace std;

namespace {

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint32_t stable_choice(
        uint32_t count, uint32_t flowid, uint32_t src, uint32_t dst, uint64_t salt) {
    if (count == 0) {
        throw invalid_argument("MpRail stable choice requires a positive count");
    }
    uint64_t value = mix64(flowid);
    value ^= mix64((static_cast<uint64_t>(src) << 32) | dst);
    value ^= mix64(salt);
    return static_cast<uint32_t>(value % count);
}

}  // namespace

MpRailTopology::MpRailTopology(const MpRailTopologyConfig& cfg, EventList& eventlist)
    : _cfg(cfg), _eventlist(eventlist) {
    validate_config();
    _server_count = (_cfg.nodes + _cfg.gpus_per_server - 1) / _cfg.gpus_per_server;
    _rail_count = _cfg.gpus_per_server;
    _l0_count = _rail_count * _cfg.planes;
    _l1_count = _cfg.planes * _cfg.l1_eps_per_plane;
    init_switches();

    cout << "MPRAIL_TOPOLOGY nodes=" << _cfg.nodes
         << " servers=" << _server_count
         << " rails=" << _rail_count
         << " planes=" << _cfg.planes
         << " l0=" << _l0_count
         << " l1=" << _l1_count
         << " gpus_per_server=" << _cfg.gpus_per_server
         << " rail_mapping=gpu_local_index"
         << " l1_eps_per_plane=" << _cfg.l1_eps_per_plane
         << " bundles=" << _cfg.l0_l1_links_per_spine
         << endl;
}

MpRailTopology::~MpRailTopology() {
    for (MpRailSwitch* sw : _l0_switches) {
        delete sw;
    }
    for (MpRailSwitch* sw : _l1_switches) {
        delete sw;
    }
    for (auto& entry : _links) {
        delete entry.second.queue;
        delete entry.second.pipe;
    }
}

void MpRailTopology::validate_config() const {
    if (_cfg.nodes == 0 || _cfg.planes == 0 || _cfg.gpus_per_server == 0
            || _cfg.l1_eps_per_plane == 0 || _cfg.l0_l1_links_per_spine == 0) {
        throw invalid_argument("MpRail topology dimensions must be positive");
    }
    if (_cfg.queue_size == 0) {
        throw invalid_argument("MpRail topology requires a positive queue size");
    }
    if (_cfg.external_linkspeed == 0 || _cfg.local_linkspeed == 0) {
        throw invalid_argument("MpRail link speeds must be positive");
    }
}

void MpRailTopology::init_switches() {
    _l0_switches.resize(_l0_count, nullptr);
    for (uint32_t rail = 0; rail < _rail_count; ++rail) {
        for (uint32_t plane = 0; plane < _cfg.planes; ++plane) {
            uint32_t id = l0_id(rail, plane);
            _l0_switches[id] = new MpRailSwitch(
                    _eventlist, l0_name(rail, plane), MpRailSwitch::L0, id,
                    _cfg.switch_latency);
        }
    }

    _l1_switches.resize(_l1_count, nullptr);
    for (uint32_t plane = 0; plane < _cfg.planes; ++plane) {
        for (uint32_t spine = 0; spine < _cfg.l1_eps_per_plane; ++spine) {
            uint32_t id = l1_id(plane, spine);
            _l1_switches[id] = new MpRailSwitch(
                    _eventlist, l1_name(plane, spine), MpRailSwitch::L1, id,
                    _cfg.switch_latency);
        }
    }
}

vector<const Route*>* MpRailTopology::get_bidir_paths(
        uint32_t src, uint32_t dest, bool reverse) {
    (void)src;
    (void)dest;
    (void)reverse;
    throw logic_error("MpRailTopology uses switch-local FIB routes");
}

vector<uint32_t>* MpRailTopology::get_neighbours(uint32_t src) {
    (void)src;
    throw logic_error("MpRailTopology does not expose a logical neighbour list");
}

uint32_t MpRailTopology::rank_server(uint32_t rank) const {
    if (rank >= _cfg.nodes) {
        throw out_of_range("MpRail rank outside node range");
    }
    return rank / _cfg.gpus_per_server;
}

uint32_t MpRailTopology::rank_rail(uint32_t rank) const {
    rank_server(rank);
    return rank % _cfg.gpus_per_server;
}

uint32_t MpRailTopology::l0_id(uint32_t rail, uint32_t plane) const {
    if (rail >= _rail_count || plane >= _cfg.planes) {
        throw out_of_range("MpRail L0 coordinate out of range");
    }
    return rail * _cfg.planes + plane;
}

uint32_t MpRailTopology::l1_id(uint32_t plane, uint32_t spine) const {
    if (plane >= _cfg.planes || spine >= _cfg.l1_eps_per_plane) {
        throw out_of_range("MpRail L1 coordinate out of range");
    }
    return plane * _cfg.l1_eps_per_plane + spine;
}

uint32_t MpRailTopology::select_plane(
        uint32_t flowid, uint32_t src, uint32_t dst) const {
    return stable_choice(_cfg.planes, flowid, src, dst, 0x504c414e45ULL);
}

void MpRailTopology::connect_endpoints(
        uint32_t src,
        uint32_t dst,
        UecSrc& uec_src,
        UecSink& uec_snk,
        simtime_picosec start_time,
        const MpRailRouteSpec* route) {
    if (route) {
        if (route->mode != MpRailRouteMode::EXPLICIT) {
            throw invalid_argument(
                    "server_forward must be expanded before connecting MpRail endpoints");
        }
        connect_explicit_endpoints(
                src, dst, uec_src, uec_snk, start_time, *route);
        return;
    }

    const uint32_t src_server = rank_server(src);
    const uint32_t dst_server = rank_server(dst);
    if (src_server == dst_server) {
        Route* routeout = make_local_route(src, dst, uec_snk.getPort(0));
        Route* routeback = make_local_route(dst, src, uec_src.getPort(0));
        routeout->set_reverse(routeback);
        routeback->set_reverse(routeout);
        // Server-local traffic uses one plane-independent NVLink/NVSwitch
        // injection resource, not one RDMA NIC port per fabric plane.
        uec_src.connectPort(0, *routeout, *routeback, uec_snk, start_time);
        cout << "MPRAIL_FLOW flow=" << uec_src.flowId()
             << " src=" << src << " dst=" << dst
             << " scope=same_server plane=-1 spine=-1" << endl;
        return;
    }

    const uint32_t src_rail = rank_rail(src);
    const uint32_t dst_rail = rank_rail(dst);
    const uint32_t preferred_plane = select_plane(uec_src.flowId(), src, dst);
    if (!_cfg.packet_spray) {
        uec_src.setPreferredNicPort(preferred_plane);
    }

    for (uint32_t plane = 0; plane < _cfg.planes; ++plane) {
        const uint32_t src_l0 = l0_id(src_rail, plane);
        const uint32_t dst_l0 = l0_id(dst_rail, plane);
        Route* routeout = make_initial_route(src, src_l0, plane);
        Route* routeback = make_initial_route(dst, dst_l0, plane);
        routeout->set_reverse(routeback);
        routeback->set_reverse(routeout);
        uec_src.connectPort(plane, *routeout, *routeback, uec_snk, start_time);

        install_l0_host_route(
                dst_l0, dst, uec_src.flowId(), plane, uec_snk.getPort(plane));
        install_l0_host_route(
                src_l0, src, uec_snk.flowId(), plane, uec_src.getPort(plane));

        if (src_rail != dst_rail) {
            for (uint32_t spine = 0;
                    spine < _cfg.l1_eps_per_plane; ++spine) {
                const uint32_t l1 = l1_id(plane, spine);
                for (uint32_t bundle = 0;
                        bundle < _cfg.l0_l1_links_per_spine; ++bundle) {
                    install_l0_ecmp_up_route(src_l0, dst, l1, bundle);
                    install_l1_ecmp_down_route(l1, dst, dst_l0, bundle);
                    install_l0_ecmp_up_route(dst_l0, src, l1, bundle);
                    install_l1_ecmp_down_route(l1, src, src_l0, bundle);
                }
            }
        }
    }

    const char* routing_mode = _cfg.packet_spray
            ? "packet_spray_ecmp" : "flow_ecmp";

    if (src_rail == dst_rail) {
        cout << "MPRAIL_FLOW flow=" << uec_src.flowId()
             << " src=" << src << " dst=" << dst
             << " scope=same_rail rail=" << src_rail
             << " routing=" << routing_mode
             << " plane=" << (_cfg.packet_spray ? -1 : static_cast<int>(preferred_plane))
             << " spine=-1" << endl;
    } else {
        cout << "MPRAIL_FLOW flow=" << uec_src.flowId()
             << " src=" << src << " dst=" << dst
             << " scope=cross_rail src_rail=" << src_rail
             << " dst_rail=" << dst_rail
             << " routing=" << routing_mode
             << " plane=" << (_cfg.packet_spray ? -1 : static_cast<int>(preferred_plane))
             << " ecmp_spines=" << _cfg.l1_eps_per_plane
             << " ecmp_bundles=" << _cfg.l0_l1_links_per_spine
             << endl;
    }
}

void MpRailTopology::validate_route_spec(
        uint32_t src, uint32_t dst, const MpRailRouteSpec& route) const {
    const uint32_t src_server = rank_server(src);
    const uint32_t dst_server = rank_server(dst);
    if (route.mode == MpRailRouteMode::SERVER_FORWARD) {
        if (src_server == dst_server) {
            throw invalid_argument(
                    "server_forward requires logical endpoints on different servers");
        }
        if (rank_server(route.src_relay) != src_server) {
            throw invalid_argument("src_relay is not on the logical source server");
        }
        if (rank_server(route.dst_relay) != dst_server) {
            throw invalid_argument("dst_relay is not on the logical destination server");
        }
        return;
    }

    const vector<MpRailRouteNode>& nodes = route.explicit_nodes;
    auto require_rank = [&](size_t index, uint32_t expected) {
        if (nodes.at(index).type != MpRailRouteNodeType::RANK
                || nodes.at(index).rank != expected
                || nodes.at(index).egress_bundle.has_value()) {
            throw invalid_argument("explicit route endpoint rank does not match the flow");
        }
    };
    require_rank(0, src);
    require_rank(nodes.size() - 1, dst);

    if (src_server == dst_server) {
        if (nodes.size() != 2) {
            throw invalid_argument(
                    "same-server explicit route must contain exactly two rank nodes");
        }
        return;
    }

    const uint32_t src_rail = rank_rail(src);
    const uint32_t dst_rail = rank_rail(dst);
    if (src_rail == dst_rail) {
        if (nodes.size() != 3 || nodes[1].type != MpRailRouteNodeType::L0
                || nodes[1].rail != src_rail
                || nodes[1].plane >= _cfg.planes
                || nodes[1].egress_bundle.has_value()) {
            throw invalid_argument(
                    "same-rail explicit route must be rank L0 rank without a bundle");
        }
        return;
    }

    if (nodes.size() != 5
            || nodes[1].type != MpRailRouteNodeType::L0
            || nodes[2].type != MpRailRouteNodeType::L1
            || nodes[3].type != MpRailRouteNodeType::L0) {
        throw invalid_argument(
                "cross-rail explicit route must be rank L0 L1 L0 rank");
    }
    const MpRailRouteNode& src_l0 = nodes[1];
    const MpRailRouteNode& l1 = nodes[2];
    const MpRailRouteNode& dst_l0 = nodes[3];
    if (src_l0.rail != src_rail || dst_l0.rail != dst_rail) {
        throw invalid_argument("explicit route L0 rail does not match its endpoint");
    }
    if (src_l0.plane >= _cfg.planes || l1.plane >= _cfg.planes
            || dst_l0.plane >= _cfg.planes
            || src_l0.plane != l1.plane || l1.plane != dst_l0.plane) {
        throw invalid_argument("explicit route must remain within one valid plane");
    }
    if (l1.spine >= _cfg.l1_eps_per_plane) {
        throw invalid_argument("explicit route spine is outside the configured range");
    }
    if (!src_l0.egress_bundle.has_value()
            || src_l0.egress_bundle.value() >= _cfg.l0_l1_links_per_spine
            || !l1.egress_bundle.has_value()
            || l1.egress_bundle.value() >= _cfg.l0_l1_links_per_spine) {
        throw invalid_argument(
                "explicit switch-to-switch hops require valid bundle coordinates");
    }
    if (dst_l0.egress_bundle.has_value()) {
        throw invalid_argument("destination L0 to rank must not specify a bundle");
    }
}

void MpRailTopology::connect_explicit_endpoints(
        uint32_t src,
        uint32_t dst,
        UecSrc& uec_src,
        UecSink& uec_snk,
        simtime_picosec start_time,
        const MpRailRouteSpec& route) {
    validate_route_spec(src, dst, route);
    const vector<MpRailRouteNode>& nodes = route.explicit_nodes;
    if (rank_server(src) == rank_server(dst)) {
        Route* routeout = make_local_route(src, dst, uec_snk.getPort(0));
        Route* routeback = make_local_route(dst, src, uec_src.getPort(0));
        routeout->set_reverse(routeback);
        routeback->set_reverse(routeout);
        uec_src.connectPort(0, *routeout, *routeback, uec_snk, start_time);
        cout << "MPRAIL_EXPLICIT_FLOW flow=" << uec_src.flowId()
             << " src=" << src << " dst=" << dst
             << " path=" << mprail_route_spec_to_string(route) << endl;
        return;
    }

    const uint32_t plane = nodes[1].plane;
    const uint32_t src_l0_id = l0_id(rank_rail(src), plane);
    const uint32_t dst_l0_id = l0_id(rank_rail(dst), plane);
    Route* routeout = make_initial_route(src, src_l0_id, plane);
    Route* routeback = make_initial_route(dst, dst_l0_id, plane);
    routeout->set_reverse(routeback);
    routeback->set_reverse(routeout);
    for (uint32_t port = 0; port < _cfg.planes; ++port) {
        uec_src.connectPort(port, *routeout, *routeback, uec_snk, start_time);
    }
    uec_src.setPreferredNicPort(plane);

    install_l0_host_route(
            dst_l0_id, dst, uec_src.flowId(), plane, uec_snk.getPort(plane));
    install_l0_host_route(
            src_l0_id, src, uec_snk.flowId(), plane, uec_src.getPort(plane));

    if (rank_rail(src) != rank_rail(dst)) {
        const MpRailRouteNode& src_l0 = nodes[1];
        const MpRailRouteNode& l1_node = nodes[2];
        const uint32_t l1_switch_id = l1_id(plane, l1_node.spine);
        MpRailSwitch* src_switch = _l0_switches.at(src_l0_id);
        MpRailSwitch* spine_switch = _l1_switches.at(l1_switch_id);
        MpRailSwitch* dst_switch = _l0_switches.at(dst_l0_id);

        Route* forward_up = make_switch_route(
                l0_name(src_l0.rail, plane), l1_name(plane, l1_node.spine),
                src_l0.egress_bundle.value(), src_switch, spine_switch);
        Route* forward_down = make_switch_route(
                l1_name(plane, l1_node.spine), l0_name(nodes[3].rail, plane),
                l1_node.egress_bundle.value(), spine_switch, dst_switch);
        Route* reverse_up = make_switch_route(
                l0_name(nodes[3].rail, plane), l1_name(plane, l1_node.spine),
                l1_node.egress_bundle.value(), dst_switch, spine_switch);
        Route* reverse_down = make_switch_route(
                l1_name(plane, l1_node.spine), l0_name(src_l0.rail, plane),
                src_l0.egress_bundle.value(), spine_switch, src_switch);

        install_explicit_route(
                src_switch, dst, uec_src.flowId(), forward_up, UP);
        install_explicit_route(
                spine_switch, dst, uec_src.flowId(), forward_down, DOWN);
        install_explicit_route(
                dst_switch, src, uec_snk.flowId(), reverse_up, UP);
        install_explicit_route(
                spine_switch, src, uec_snk.flowId(), reverse_down, DOWN);
    }

    cout << "MPRAIL_EXPLICIT_FLOW flow=" << uec_src.flowId()
         << " src=" << src << " dst=" << dst
         << " plane=" << plane
         << " path=" << mprail_route_spec_to_string(route) << endl;
}

Route* MpRailTopology::make_initial_route(
        uint32_t rank, uint32_t l0, uint32_t plane) {
    const uint32_t rail = rank_rail(rank);
    CachedLink& link = get_or_create_link(
            host_src_name(rank), l0_name(rail, plane), 0,
            _cfg.external_linkspeed, _cfg.link_latency,
            nullptr, _l0_switches.at(l0));
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(_l0_switches.at(l0));
    return route;
}

Route* MpRailTopology::make_local_route(
        uint32_t src, uint32_t dst, PacketSink* final_sink) {
    CachedLink& link = get_or_create_link(
            "MPRAIL_LOCAL_" + host_src_name(src), host_dst_name(dst), 0,
            _cfg.local_linkspeed, _cfg.local_latency, nullptr, nullptr);
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(final_sink);
    return route;
}

Route* MpRailTopology::make_l0_to_host_route(
        uint32_t l0, uint32_t rank, uint32_t plane, PacketSink* final_sink) {
    const uint32_t rail = rank_rail(rank);
    CachedLink& link = get_or_create_link(
            l0_name(rail, plane), host_dst_name(rank), 0,
            _cfg.external_linkspeed, _cfg.link_latency,
            _l0_switches.at(l0), nullptr);
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(final_sink);
    return route;
}

Route* MpRailTopology::make_switch_route(
        const string& src_name,
        const string& dst_name,
        uint32_t bundle,
        MpRailSwitch* src_switch,
        MpRailSwitch* dst_switch) {
    CachedLink& link = get_or_create_link(
            src_name, dst_name, bundle,
            _cfg.external_linkspeed, _cfg.link_latency,
            src_switch, dst_switch);
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(dst_switch);
    return route;
}

MpRailTopology::CachedLink& MpRailTopology::get_or_create_link(
        const string& src_name,
        const string& dst_name,
        uint32_t bundle,
        linkspeed_bps speed,
        simtime_picosec latency,
        MpRailSwitch* src_switch,
        PacketSink* remote_endpoint) {
    const string key = src_name + "->" + dst_name + "(b" + to_string(bundle) + ")";
    auto found = _links.find(key);
    if (found != _links.end()) {
        return found->second;
    }

    CachedLink link;
    if (_cfg.enable_ecn && _cfg.ecn_threshold > 0) {
        link.queue = new ECNQueue(
                speed, _cfg.queue_size, _eventlist, nullptr, _cfg.ecn_threshold);
    } else {
        link.queue = new Queue(speed, _cfg.queue_size, _eventlist, nullptr);
    }
    link.queue->setName(key);
    if (remote_endpoint) {
        link.queue->setRemoteEndpoint(remote_endpoint);
    }
    if (src_switch) {
        src_switch->addPort(link.queue);
    }
    link.pipe = new Pipe(latency, _eventlist);
    link.pipe->setName("Pipe-" + key);
    cout << "MPRAIL_LINK src=" << src_name
         << " dst=" << dst_name
         << " bundle=" << bundle
         << " speed_gbps=" << speed / 1000000000
         << " latency_ns=" << timeAsNs(latency)
         << endl;
    auto inserted = _links.emplace(key, link);
    return inserted.first->second;
}

void MpRailTopology::install_l0_host_route(
        uint32_t l0,
        uint32_t dst_rank,
        int flowid,
        uint32_t plane,
        PacketSink* final_sink) {
    const string key = "host:" + to_string(l0) + ":" + to_string(dst_rank)
                       + ":" + to_string(flowid);
    if (!_installed_host_routes.insert(key).second) {
        return;
    }
    Route* route = make_l0_to_host_route(l0, dst_rank, plane, final_sink);
    _l0_switches.at(l0)->addHostRoute(dst_rank, flowid, route);
}

void MpRailTopology::install_l0_ecmp_up_route(
        uint32_t l0,
        uint32_t dst_rank,
        uint32_t l1,
        uint32_t bundle) {
    const string key = "up:" + to_string(l0) + ":" + to_string(l1)
                       + ":" + to_string(dst_rank) + ":" + to_string(bundle);
    if (!_installed_routes.insert(key).second) {
        return;
    }
    const uint32_t rail = l0 / _cfg.planes;
    const uint32_t plane = l0 % _cfg.planes;
    const uint32_t spine = l1 % _cfg.l1_eps_per_plane;
    Route* route = make_switch_route(
            l0_name(rail, plane), l1_name(plane, spine), bundle,
            _l0_switches.at(l0), _l1_switches.at(l1));
    _l0_switches.at(l0)->addRoute(dst_rank, route, UP);
}

void MpRailTopology::install_l1_ecmp_down_route(
        uint32_t l1,
        uint32_t dst_rank,
        uint32_t l0,
        uint32_t bundle) {
    const string key = "down:" + to_string(l1) + ":" + to_string(l0)
                       + ":" + to_string(dst_rank) + ":" + to_string(bundle);
    if (!_installed_routes.insert(key).second) {
        return;
    }
    const uint32_t plane = l1 / _cfg.l1_eps_per_plane;
    const uint32_t spine = l1 % _cfg.l1_eps_per_plane;
    const uint32_t rail = l0 / _cfg.planes;
    Route* route = make_switch_route(
            l1_name(plane, spine), l0_name(rail, plane), bundle,
            _l1_switches.at(l1), _l0_switches.at(l0));
    _l1_switches.at(l1)->addRoute(dst_rank, route, DOWN);
}

void MpRailTopology::install_explicit_route(
        MpRailSwitch* sw,
        uint32_t dst_rank,
        int flowid,
        Route* route,
        packet_direction direction) {
    sw->addExplicitRoute(dst_rank, flowid, route, direction);
}

string MpRailTopology::host_src_name(uint32_t rank) {
    return "MPRAIL_HOST_SRC_" + to_string(rank);
}

string MpRailTopology::host_dst_name(uint32_t rank) {
    return "MPRAIL_HOST_DST_" + to_string(rank);
}

string MpRailTopology::l0_name(uint32_t rail, uint32_t plane) {
    return "MPRAIL_L0_r" + to_string(rail) + "_p" + to_string(plane);
}

string MpRailTopology::l1_name(uint32_t plane, uint32_t spine) {
    return "MPRAIL_L1_p" + to_string(plane) + "_s" + to_string(spine);
}
