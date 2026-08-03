// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_topology.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#include "ecnqueue.h"

using namespace std;

static uint64_t oxc_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static uint32_t oxc_stable_source_port(
        uint32_t source_ports,
        uint32_t flowid,
        uint32_t src,
        uint32_t dst) {
    uint64_t h = oxc_mix64(flowid);
    h ^= oxc_mix64(src);
    h ^= oxc_mix64(dst);
    return static_cast<uint32_t>(h % source_ports);
}

OxcTopology::OxcTopology(const OxcTopologyConfig& cfg, EventList& eventlist)
    : _cfg(cfg), _eventlist(eventlist) {
    validate_config();
    _trays = (_cfg.nodes + _cfg.ranks_per_tray - 1) / _cfg.ranks_per_tray;
    _l0_count = _trays * _cfg.l1_planes;
    _l1_count = _cfg.groups * _cfg.l1_planes * _cfg.l1_eps_per_l1_plane;
    _logical_nodes_per_group = oxc_ocs_coupled_logical_nodes_per_group(
            _cfg.l1_planes, _cfg.l1_eps_per_l1_plane);

    init_switches();
    init_ocs();
    load_route_plan();
}

OxcTopology::~OxcTopology() {
    for (OxcSwitch* sw : _l0_switches) {
        delete sw;
    }
    for (OxcSwitch* sw : _l1_switches) {
        delete sw;
    }
    for (auto& it : _links) {
        delete it.second.queue;
        delete it.second.pipe;
    }
}

void OxcTopology::validate_config() const {
    if (_cfg.nodes == 0) {
        throw invalid_argument("OxcTopology requires positive nodes");
    }
    if (_cfg.groups == 0) {
        throw invalid_argument("OxcTopology requires positive groups");
    }
    if (_cfg.ranks_per_tray == 0) {
        throw invalid_argument("OxcTopology requires positive ranks_per_tray");
    }
    if (_cfg.ranks_per_group == 0) {
        throw invalid_argument("OxcTopology requires positive ranks_per_group");
    }
    if (_cfg.l1_planes == 0) {
        throw invalid_argument("OxcTopology requires positive l1_planes");
    }
    if (_cfg.source_ports > _cfg.l1_planes) {
        throw invalid_argument("OxcTopology source_ports cannot exceed l1_planes");
    }
    if (_cfg.l1_eps_per_l1_plane == 0 || _cfg.l1_eps_per_l1_plane % 2 != 0) {
        throw invalid_argument("OxcTopology requires even positive l1_eps_per_l1_plane");
    }
    if (_cfg.groups * _cfg.ranks_per_group < _cfg.nodes) {
        throw invalid_argument("OxcTopology groups * ranks_per_group is smaller than nodes");
    }
    if (_cfg.queue_size == 0) {
        throw invalid_argument("OxcTopology requires positive queue_size");
    }
}

static vector<string> oxc_split_csv_line(const string& line) {
    vector<string> fields;
    string field;
    stringstream ss(line);
    while (getline(ss, field, ',')) {
        if (!field.empty() && field.back() == '\r') {
            field.pop_back();
        }
        fields.push_back(field);
    }
    return fields;
}

static vector<uint32_t> oxc_parse_l1_path(const string& value) {
    vector<uint32_t> path;
    string token;
    stringstream ss(value);
    while (getline(ss, token, ';')) {
        if (token.empty()) {
            continue;
        }
        path.push_back(static_cast<uint32_t>(stoul(token)));
    }
    return path;
}

void OxcTopology::load_route_plan() {
    if (_cfg.route_plan_path.empty()) {
        return;
    }

    ifstream in(_cfg.route_plan_path);
    if (!in.good()) {
        throw invalid_argument("Oxc route plan cannot be opened: " + _cfg.route_plan_path);
    }

    string header_line;
    while (getline(in, header_line)) {
        if (!header_line.empty() && header_line[0] != '#') {
            break;
        }
    }
    if (header_line.empty()) {
        throw invalid_argument("Oxc route plan is empty: " + _cfg.route_plan_path);
    }

    vector<string> header = oxc_split_csv_line(header_line);
    map<string, size_t> col;
    for (size_t i = 0; i < header.size(); i++) {
        col[header[i]] = i;
    }
    if (col.find("flowid") == col.end()) {
        throw invalid_argument("Oxc route plan requires a flowid column");
    }

    auto parse_i64 = [](const vector<string>& fields, const map<string, size_t>& columns,
                        const string& name, int64_t default_value) -> int64_t {
        auto it = columns.find(name);
        if (it == columns.end() || it->second >= fields.size() || fields[it->second].empty()) {
            return default_value;
        }
        return stoll(fields[it->second]);
    };

    string line;
    uint64_t loaded = 0;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> fields = oxc_split_csv_line(line);
        RoutePlanEntry entry;
        entry.flowid = static_cast<uint32_t>(parse_i64(fields, col, "flowid", 0));
        entry.src = static_cast<uint32_t>(parse_i64(fields, col, "src", numeric_limits<int32_t>::max()));
        entry.dst = static_cast<uint32_t>(parse_i64(fields, col, "dst", numeric_limits<int32_t>::max()));
        entry.src_l0_plane = static_cast<int32_t>(parse_i64(fields, col, "src_l0_plane", -1));
        entry.src_egress_rank = static_cast<int32_t>(parse_i64(fields, col, "src_egress_rank", entry.src));
        entry.src_l1_id = static_cast<int32_t>(parse_i64(fields, col, "src_l1_id", -1));
        entry.dst_l1_id = static_cast<int32_t>(parse_i64(fields, col, "dst_l1_id", -1));
        entry.dst_l0_plane = static_cast<int32_t>(parse_i64(fields, col, "dst_l0_plane", -1));
        entry.dst_ingress_rank = static_cast<int32_t>(parse_i64(fields, col, "dst_ingress_rank", entry.dst));
        auto path_col = col.find("ocs_l1_path");
        if (path_col != col.end() && path_col->second < fields.size() && !fields[path_col->second].empty()) {
            entry.ocs_l1_path = oxc_parse_l1_path(fields[path_col->second]);
        }

        if (entry.flowid == 0) {
            throw invalid_argument("Oxc route plan has flowid 0");
        }
        if (entry.src >= _cfg.nodes || entry.dst >= _cfg.nodes
            || entry.src_egress_rank < 0
            || static_cast<uint32_t>(entry.src_egress_rank) >= _cfg.nodes
            || entry.dst_ingress_rank < 0
            || static_cast<uint32_t>(entry.dst_ingress_rank) >= _cfg.nodes) {
            throw invalid_argument("Oxc route plan src/dst out of range for flowid "
                                   + to_string(entry.flowid));
        }
        if (rank_tray(static_cast<uint32_t>(entry.src_egress_rank)) != rank_tray(entry.src)) {
            throw invalid_argument("Oxc route plan src_egress_rank is not in source tray for flowid "
                                   + to_string(entry.flowid));
        }
        if (rank_tray(static_cast<uint32_t>(entry.dst_ingress_rank)) != rank_tray(entry.dst)) {
            throw invalid_argument("Oxc route plan dst_ingress_rank is not in destination tray for flowid "
                                   + to_string(entry.flowid));
        }
        if (entry.src_l0_plane >= static_cast<int32_t>(_cfg.l1_planes)
            || entry.dst_l0_plane >= static_cast<int32_t>(_cfg.l1_planes)) {
            throw invalid_argument("Oxc route plan L0 plane out of range for flowid "
                                   + to_string(entry.flowid));
        }
        if (entry.src_l1_id >= static_cast<int32_t>(_l1_count)
            || entry.dst_l1_id >= static_cast<int32_t>(_l1_count)) {
            throw invalid_argument("Oxc route plan L1 id out of range for flowid "
                                   + to_string(entry.flowid));
        }
        for (uint32_t l1 : entry.ocs_l1_path) {
            if (l1 >= _l1_count) {
                throw invalid_argument("Oxc route plan OCS path L1 id out of range for flowid "
                                       + to_string(entry.flowid));
            }
        }
        if (!entry.ocs_l1_path.empty()) {
            if (entry.src_l1_id < 0 || entry.dst_l1_id < 0
                || entry.ocs_l1_path.front() != static_cast<uint32_t>(entry.src_l1_id)
                || entry.ocs_l1_path.back() != static_cast<uint32_t>(entry.dst_l1_id)) {
                throw invalid_argument("Oxc route plan OCS path must start at src_l1_id and end at dst_l1_id for flowid "
                                       + to_string(entry.flowid));
            }
        }
        _route_plan[entry.flowid] = entry;
        loaded++;
    }

    cout << "#----------- OXC ROUTE_PLAN begin ------------" << endl;
    cout << "path " << _cfg.route_plan_path << endl;
    cout << "entries " << loaded << endl;
    cout << "#----------- OXC ROUTE_PLAN END ------------" << endl;
}

const OxcTopology::RoutePlanEntry* OxcTopology::route_plan_entry(
        uint32_t flowid, uint32_t src, uint32_t dst) const {
    auto it = _route_plan.find(flowid);
    if (it == _route_plan.end()) {
        return nullptr;
    }
    const RoutePlanEntry& entry = it->second;
    if (entry.src != src || entry.dst != dst) {
        return nullptr;
    }
    return &entry;
}

const OxcTopology::RoutePlanEntry* OxcTopology::route_plan_entry(uint32_t flowid) const {
    auto it = _route_plan.find(flowid);
    if (it == _route_plan.end()) {
        return nullptr;
    }
    return &it->second;
}

uint32_t OxcTopology::source_ports() const {
    return _cfg.source_ports == 0 ? _cfg.l1_planes : _cfg.source_ports;
}

void OxcTopology::init_switches() {
    _l0_switches.resize(_l0_count, nullptr);
    for (uint32_t l0 = 0; l0 < _l0_count; l0++) {
        _l0_switches[l0] = new OxcSwitch(
                _eventlist, "Oxc_L0_" + to_string(l0), OxcSwitch::L0, l0,
                _cfg.switch_latency);
    }

    _l1_switches.resize(_l1_count, nullptr);
    for (uint32_t l1 = 0; l1 < _l1_count; l1++) {
        _l1_switches[l1] = new OxcSwitch(
                _eventlist, "Oxc_L1_" + to_string(l1), OxcSwitch::L1, l1,
                _cfg.switch_latency);
        _l1_switches[l1]->set_special_next_hop_resolver(
                OxcTopology::l1_special_next_hop_thunk, this);
    }
}

void OxcTopology::init_ocs() {
    if (_cfg.ocs_mode == OxcOcsMode::OFF) {
        return;
    }
    _ocs_graph = build_oxc_ocs_coupled_template(
            _cfg.groups,
            _cfg.l1_planes,
            _cfg.l1_eps_per_l1_plane,
            _cfg.ocs_degree,
            _cfg.ocs_seed);

    if (_cfg.ocs_mode == OxcOcsMode::SPRAYPOINT) {
        OxcOcsSprayPointParams params;
        params.spray_p = _cfg.spray_p;
        params.spray_h = _cfg.spray_h;
        params.spray_levels = _cfg.spray_levels;
        params.spray_seed = _cfg.ocs_seed;
        _spraypoint = make_unique<OxcOcsSprayPointRouter>(
                _ocs_graph, _cfg.groups, _logical_nodes_per_group, params);
    } else if (_cfg.ocs_mode == OxcOcsMode::KSP) {
        OxcOcsKspParams params;
        params.k = _cfg.ksp_k;
        params.max_hops = _cfg.ksp_max_hops;
        params.seed = _cfg.ksp_seed;
        params.max_paths_per_pair = _cfg.ksp_max_paths_per_pair;
        _ksp = make_unique<OxcOcsKspRouter>(
                _ocs_graph, _cfg.groups, _logical_nodes_per_group, params);
    }
}

vector<const Route*>* OxcTopology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    (void)src;
    (void)dest;
    (void)reverse;
    throw logic_error("OxcTopology uses switch-local FIB and does not enumerate paths");
}

vector<uint32_t>* OxcTopology::get_neighbours(uint32_t src) {
    (void)src;
    throw logic_error("OxcTopology::get_neighbours is not implemented for physical switches yet");
}

uint32_t OxcTopology::rank_group(uint32_t rank) const {
    if (rank >= _cfg.nodes) {
        throw out_of_range("Oxc rank out of range");
    }
    return rank / _cfg.ranks_per_group;
}

uint32_t OxcTopology::rank_tray(uint32_t rank) const {
    if (rank >= _cfg.nodes) {
        throw out_of_range("Oxc rank out of range");
    }
    return rank / _cfg.ranks_per_tray;
}

uint32_t OxcTopology::rank_l0(uint32_t rank, uint32_t plane) const {
    if (plane >= _cfg.l1_planes) {
        throw out_of_range("Oxc plane out of range");
    }
    return rank_tray(rank) * _cfg.l1_planes + plane;
}

uint32_t OxcTopology::l1_physical_id(uint32_t group, uint32_t plane, uint32_t eps) const {
    return oxc_ocs_coupled_endpoint_id(
            group, plane, eps, _cfg.l1_planes, _cfg.l1_eps_per_l1_plane);
}

uint32_t OxcTopology::l1_group(uint32_t l1_id) const {
    return oxc_ocs_decode_coupled_endpoint(
            l1_id, _cfg.l1_planes, _cfg.l1_eps_per_l1_plane).group;
}

uint32_t OxcTopology::l1_plane(uint32_t l1_id) const {
    return oxc_ocs_decode_coupled_endpoint(
            l1_id, _cfg.l1_planes, _cfg.l1_eps_per_l1_plane).l1_plane;
}

uint32_t OxcTopology::l1_eps(uint32_t l1_id) const {
    return oxc_ocs_decode_coupled_endpoint(
            l1_id, _cfg.l1_planes, _cfg.l1_eps_per_l1_plane).l1_eps;
}

uint32_t OxcTopology::l1_logical_node(uint32_t l1_id) const {
    OxcOcsEndpoint ep = oxc_ocs_decode_coupled_endpoint(
            l1_id, _cfg.l1_planes, _cfg.l1_eps_per_l1_plane);
    return oxc_ocs_coupled_logical_node_id(
            ep.group, ep.l1_plane, ep.coupled_pair,
            _cfg.l1_planes, _cfg.l1_eps_per_l1_plane);
}

uint32_t OxcTopology::l1_from_logical_node(uint32_t logical_node, uint32_t coupled_member) const {
    OxcOcsLogicalNode node = oxc_ocs_decode_coupled_logical_node(
            logical_node, _cfg.l1_planes, _cfg.l1_eps_per_l1_plane);
    uint32_t eps = coupled_member == 0 ? node.l1_eps_member0 : node.l1_eps_member1;
    return l1_physical_id(node.group, node.l1_plane, eps);
}

void OxcTopology::connect_endpoints(
        uint32_t src,
        uint32_t dst,
        UecSrc& uec_src,
        UecSink& uec_snk,
        simtime_picosec start_time) {
    bool same_tray = src != dst && rank_tray(src) == rank_tray(dst);
    if (same_tray) {
        Route* routeout = make_local_route(src, dst, 0, uec_snk.getPort(0));
        Route* routeback = make_local_route(dst, src, 0, uec_src.getPort(0));
        routeout->set_reverse(routeback);
        routeback->set_reverse(routeout);
        for (uint32_t p = 0; p < source_ports(); p++) {
            uec_src.connectPort(p, *routeout, *routeback, uec_snk, start_time);
        }
        return;
    }

    const RoutePlanEntry* plan = route_plan_entry(uec_src.flowId(), src, dst);
    if (plan && plan->src_l1_id >= 0 && plan->src_l0_plane >= 0
        && l1_plane(static_cast<uint32_t>(plan->src_l1_id))
           != static_cast<uint32_t>(plan->src_l0_plane)) {
        throw invalid_argument("Oxc route plan source L0 plane does not match source L1 plane for flowid "
                               + to_string(uec_src.flowId()));
    }
    if (plan && plan->dst_l1_id >= 0 && plan->dst_l0_plane >= 0
        && l1_plane(static_cast<uint32_t>(plan->dst_l1_id))
           != static_cast<uint32_t>(plan->dst_l0_plane)) {
        throw invalid_argument("Oxc route plan destination L0 plane does not match destination L1 plane for flowid "
                               + to_string(uec_src.flowId()));
    }

    const uint32_t dst_ingress = plan && plan->dst_ingress_rank >= 0
            ? static_cast<uint32_t>(plan->dst_ingress_rank)
            : dst;
    const uint32_t src_egress = plan && plan->src_egress_rank >= 0
            ? static_cast<uint32_t>(plan->src_egress_rank)
            : src;

    if (source_ports() > 1 && _cfg.ocs_choice == OxcOcsChoice::FLOW_HASH) {
        uint32_t source_port = plan && plan->src_l0_plane >= 0
                ? static_cast<uint32_t>(plan->src_l0_plane) % source_ports()
                : oxc_stable_source_port(source_ports(), uec_src.flowId(), src, dst);
        uec_src.setPreferredNicPort(source_port);
    }

    for (uint32_t plane = 0; plane < _cfg.l1_planes; plane++) {
        install_l0_host_route(
                rank_l0(dst, plane), dst, uec_src.flowId(), plane, uec_snk.getPort(0));
        install_l0_host_route(
                rank_l0(src, plane), src, uec_snk.flowId(), plane, uec_src.getPort(0));
        if (dst_ingress != dst) {
            install_l0_host_route_via_ingress(
                    rank_l0(dst_ingress, plane), dst, uec_src.flowId(), dst_ingress,
                    plane, uec_snk.getPort(0));
        }
    }

    for (uint32_t p = 0; p < source_ports(); p++) {
        uint32_t src_plane = plan && plan->src_l0_plane >= 0
                ? static_cast<uint32_t>(plan->src_l0_plane)
                : (p % _cfg.l1_planes);
        uint32_t dst_plane = plan && plan->dst_l0_plane >= 0
                ? static_cast<uint32_t>(plan->dst_l0_plane)
                : (p % _cfg.l1_planes);
        uint32_t src_l0 = rank_l0(src_egress, src_plane);
        uint32_t dst_l0 = rank_l0(dst_ingress, dst_plane);

        Route* routeout = src_egress == src
                ? make_initial_route(src, src_l0, src_plane)
                : make_initial_route_via_local(src, src_egress, src_l0, src_plane);
        Route* routeback = make_initial_route(dst, dst_l0, dst_plane);
        routeout->set_reverse(routeback);
        routeback->set_reverse(routeout);
        uec_src.connectPort(p, *routeout, *routeback, uec_snk, start_time);

        if (dst_ingress != dst) {
            install_l0_host_route_via_ingress(
                    dst_l0, dst, uec_src.flowId(), dst_ingress, dst_plane, uec_snk.getPort(p));
        } else {
            install_l0_host_route(dst_l0, dst, uec_src.flowId(), dst_plane, uec_snk.getPort(p));
        }
        if (src_egress != src) {
            install_l0_host_route_via_ingress(
                    src_l0, src, uec_snk.flowId(), src_egress, src_plane, uec_src.getPort(p));
        } else {
            install_l0_host_route(src_l0, src, uec_snk.flowId(), src_plane, uec_src.getPort(p));
        }

        if (plan && plan->src_l1_id >= 0) {
            install_l0_flow_route(src_l0, dst, uec_src.flowId(),
                                  static_cast<uint32_t>(plan->src_l1_id));
        } else {
            install_l0_up_routes(src_l0, dst);
        }
        install_l0_up_routes(dst_l0, src);
    }

    install_l1_down_routes(dst);
    install_l1_down_routes(src);

    if (plan && plan->dst_l1_id >= 0 && plan->dst_l0_plane >= 0) {
        uint32_t dst_l0 = rank_l0(dst_ingress, static_cast<uint32_t>(plan->dst_l0_plane));
        install_l1_flow_down_route(
                static_cast<uint32_t>(plan->dst_l1_id), dst, uec_src.flowId(), dst_l0);
    }
}

Route* OxcTopology::make_initial_route(uint32_t rank, uint32_t l0, uint32_t plane) {
    return make_switch_route(
            host_src_name(rank),
            l0_name(l0),
            plane,
            _cfg.external_linkspeed,
            _cfg.link_latency,
            nullptr,
            _l0_switches.at(l0));
}

Route* OxcTopology::make_initial_route_via_local(
        uint32_t src,
        uint32_t egress_rank,
        uint32_t l0,
        uint32_t plane) {
    CachedLink& local = get_or_create_link(
            "LOCAL_" + host_src_name(src),
            host_dst_name(egress_rank),
            0,
            _cfg.local_linkspeed,
            _cfg.local_latency,
            nullptr,
            nullptr);
    CachedLink& uplink = get_or_create_link(
            host_src_name(egress_rank),
            l0_name(l0),
            plane,
            _cfg.external_linkspeed,
            _cfg.link_latency,
            nullptr,
            _l0_switches.at(l0));
    Route* route = new Route();
    route->push_back(local.queue);
    route->push_back(local.pipe);
    route->push_back(uplink.queue);
    route->push_back(uplink.pipe);
    route->push_back(_l0_switches.at(l0));
    return route;
}

Route* OxcTopology::make_local_route(uint32_t src, uint32_t dst, uint32_t bundle, PacketSink* final_sink) {
    CachedLink& link = get_or_create_link(
            "LOCAL_" + host_src_name(src),
            host_dst_name(dst),
            bundle,
            _cfg.local_linkspeed,
            _cfg.local_latency,
            nullptr,
            nullptr);
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(final_sink);
    return route;
}

Route* OxcTopology::make_l0_to_host_route(
        uint32_t l0,
        uint32_t rank,
        uint32_t plane,
        PacketSink* final_sink) {
    CachedLink& link = get_or_create_link(
            l0_name(l0),
            host_dst_name(rank),
            plane,
            _cfg.external_linkspeed,
            _cfg.link_latency,
            _l0_switches.at(l0),
            nullptr);
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(final_sink);
    return route;
}

Route* OxcTopology::make_l0_to_host_via_local_route(
        uint32_t l0,
        uint32_t ingress_rank,
        uint32_t final_rank,
        uint32_t plane,
        PacketSink* final_sink) {
    CachedLink& downlink = get_or_create_link(
            l0_name(l0),
            host_dst_name(ingress_rank),
            plane,
            _cfg.external_linkspeed,
            _cfg.link_latency,
            _l0_switches.at(l0),
            nullptr);
    CachedLink& local = get_or_create_link(
            "LOCAL_" + host_src_name(ingress_rank),
            host_dst_name(final_rank),
            0,
            _cfg.local_linkspeed,
            _cfg.local_latency,
            nullptr,
            nullptr);
    Route* route = new Route();
    route->push_back(downlink.queue);
    route->push_back(downlink.pipe);
    route->push_back(local.queue);
    route->push_back(local.pipe);
    route->push_back(final_sink);
    return route;
}

Route* OxcTopology::make_switch_route(
        const string& src_name,
        const string& dst_name,
        uint32_t bundle,
        linkspeed_bps speed,
        simtime_picosec latency,
        OxcSwitch* src_switch,
        PacketSink* dst_sink) {
    CachedLink& link = get_or_create_link(
            src_name, dst_name, bundle, speed, latency, src_switch, dst_sink);
    Route* route = new Route();
    route->push_back(link.queue);
    route->push_back(link.pipe);
    route->push_back(dst_sink);
    return route;
}

OxcTopology::CachedLink& OxcTopology::get_or_create_link(
        const string& src_name,
        const string& dst_name,
        uint32_t bundle,
        linkspeed_bps speed,
        simtime_picosec latency,
        OxcSwitch* src_switch,
        PacketSink* remote_endpoint) {
    string key = src_name + "->" + dst_name + "(" + to_string(bundle) + ")";
    auto it = _links.find(key);
    if (it != _links.end()) {
        return it->second;
    }

    CachedLink link;
    if (_cfg.enable_ecn && _cfg.ecn_threshold > 0) {
        link.queue = new ECNQueue(speed, _cfg.queue_size, _eventlist, nullptr, _cfg.ecn_threshold);
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

    auto inserted = _links.emplace(key, link);
    return inserted.first->second;
}

void OxcTopology::install_l0_host_route(
        uint32_t l0,
        uint32_t dst_rank,
        int flowid,
        uint32_t plane,
        PacketSink* final_sink) {
    string key = host_route_key(l0, dst_rank, flowid);
    if (!_installed_host_routes.insert(key).second) {
        return;
    }
    Route* route = make_l0_to_host_route(l0, dst_rank, plane, final_sink);
    _l0_switches.at(l0)->addHostRoute(dst_rank, flowid, route);
}

void OxcTopology::install_l0_host_route_via_ingress(
        uint32_t l0,
        uint32_t dst_rank,
        int flowid,
        uint32_t ingress_rank,
        uint32_t plane,
        PacketSink* final_sink) {
    string key = host_route_key(l0, dst_rank, flowid) + ":via:" + to_string(ingress_rank);
    if (!_installed_host_routes.insert(key).second) {
        return;
    }
    Route* route = make_l0_to_host_via_local_route(l0, ingress_rank, dst_rank, plane, final_sink);
    _l0_switches.at(l0)->addHostRoute(dst_rank, flowid, route);
}

void OxcTopology::install_l0_up_routes(uint32_t l0, uint32_t dst_rank) {
    uint32_t plane = l0 % _cfg.l1_planes;
    uint32_t tray = l0 / _cfg.l1_planes;
    uint32_t src_group = (tray * _cfg.ranks_per_tray) / _cfg.ranks_per_group;
    if (src_group >= _cfg.groups) {
        throw out_of_range("Oxc L0 group out of range");
    }

    for (uint32_t eps = 0; eps < _cfg.l1_eps_per_l1_plane; eps++) {
        uint32_t l1 = l1_physical_id(src_group, plane, eps);
        string key = route_key("l0up", l0, dst_rank, l1);
        if (!_installed_routes.insert(key).second) {
            continue;
        }
        Route* route = make_switch_route(
                l0_name(l0), l1_name(l1), eps,
                _cfg.external_linkspeed, _cfg.link_latency,
                _l0_switches.at(l0), _l1_switches.at(l1));
        _l0_switches.at(l0)->addRoute(dst_rank, route, UP);
    }
}

void OxcTopology::install_l0_flow_route(
        uint32_t l0,
        uint32_t dst_rank,
        int flowid,
        uint32_t dst_l1) {
    string key = route_key("l0flow:" + to_string(flowid), l0, dst_rank, dst_l1);
    if (!_installed_routes.insert(key).second) {
        return;
    }
    Route* route = make_switch_route(
            l0_name(l0), l1_name(dst_l1), l1_eps(dst_l1),
            _cfg.external_linkspeed, _cfg.link_latency,
            _l0_switches.at(l0), _l1_switches.at(dst_l1));
    _l0_switches.at(l0)->addFlowRoute(dst_rank, flowid, route, UP);
}

void OxcTopology::install_l1_down_routes(uint32_t dst_rank) {
    uint32_t group = rank_group(dst_rank);
    for (uint32_t plane = 0; plane < _cfg.l1_planes; plane++) {
        uint32_t dst_l0 = rank_l0(dst_rank, plane);
        for (uint32_t eps = 0; eps < _cfg.l1_eps_per_l1_plane; eps++) {
            uint32_t l1 = l1_physical_id(group, plane, eps);
            string key = route_key("l1down", l1, dst_rank, dst_l0);
            if (!_installed_routes.insert(key).second) {
                continue;
            }
            Route* route = make_switch_route(
                    l1_name(l1), l0_name(dst_l0), eps,
                    _cfg.external_linkspeed, _cfg.link_latency,
                    _l1_switches.at(l1), _l0_switches.at(dst_l0));
            _l1_switches.at(l1)->addRoute(dst_rank, route, DOWN);
        }
    }
}

void OxcTopology::install_l1_flow_down_route(
        uint32_t l1,
        uint32_t dst_rank,
        int flowid,
        uint32_t dst_l0) {
    string key = route_key("l1flow:" + to_string(flowid), l1, dst_rank, dst_l0);
    if (!_installed_routes.insert(key).second) {
        return;
    }
    Route* route = make_switch_route(
            l1_name(l1), l0_name(dst_l0), l1_eps(l1),
            _cfg.external_linkspeed, _cfg.link_latency,
            _l1_switches.at(l1), _l0_switches.at(dst_l0));
    _l1_switches.at(l1)->addFlowRoute(dst_rank, flowid, route, DOWN);
}

Route* OxcTopology::l1_special_next_hop_thunk(OxcSwitch* sw, Packet& pkt, void* context) {
    return static_cast<OxcTopology*>(context)->l1_special_next_hop(sw, pkt);
}

Route* OxcTopology::l1_special_next_hop(OxcSwitch* sw, Packet& pkt) {
    if (sw->getType() != OxcSwitch::L1 || _cfg.ocs_mode == OxcOcsMode::OFF) {
        return nullptr;
    }

    uint32_t current_l1 = sw->getID();
    uint32_t current_group = l1_group(current_l1);
    uint32_t dst_group = rank_group(pkt.dst());
    const RoutePlanEntry* flow_plan = route_plan_entry(pkt.flow_id());
    bool source_route_packet = flow_plan
            && flow_plan->src_l1_id >= 0
            && flow_plan->dst_l1_id >= 0
            && (pkt.type() == UECDATA || pkt.type() == UECRTS)
            && pkt.dst() == flow_plan->dst;

    if (source_route_packet) {
        if (!flow_plan->ocs_l1_path.empty()) {
            const vector<uint32_t>& path = flow_plan->ocs_l1_path;
            auto it = find(path.begin(), path.end(), current_l1);
            if (it == path.end()) {
                throw runtime_error("Oxc route plan OCS path does not contain current L1 for flowid "
                                    + to_string(flow_plan->flowid));
            }
            if (it + 1 == path.end()) {
                if (pkt.has_ocs_ksp_route()) {
                    pkt.clear_ocs_ksp_route();
                }
                return nullptr;
            }
            pkt.set_direction(UP);
            return l1_to_l1_route(current_l1, *(it + 1), 0);
        }

        if (current_group == dst_group) {
            if (pkt.has_ocs_ksp_route()) {
                pkt.clear_ocs_ksp_route();
            }
            return nullptr;
        }

        throw runtime_error("Oxc route plan cross-group flow requires ocs_l1_path for flowid "
                            + to_string(flow_plan->flowid));
    }

    if (current_group == dst_group) {
        if (pkt.has_ocs_ksp_route()) {
            pkt.clear_ocs_ksp_route();
        }
        return nullptr;
    }

    uint32_t current_node = l1_logical_node(current_l1);
    uint32_t current_member = l1_eps(current_l1) % 2;
    uint32_t next_node = numeric_limits<uint32_t>::max();

    if (_cfg.ocs_mode == OxcOcsMode::SPRAYPOINT) {
        if (!_spraypoint) {
            throw logic_error("Oxc SprayPoint router is not initialized");
        }
        bool source_step = !pkt.ocs_source_sprayed()
                && pkt.src() < _cfg.nodes
                && current_group == rank_group(pkt.src());
        if (source_step) {
            pkt.mark_ocs_source_sprayed();
        }
        OxcOcsSprayPointChoice choice =
                _cfg.ocs_choice == OxcOcsChoice::FLOW_HASH
                        ? OxcOcsSprayPointChoice::FLOW_HASH
                        : OxcOcsSprayPointChoice::PACKET_RR;
        uint32_t rr = _cfg.ocs_choice == OxcOcsChoice::FLOW_HASH ? 0 : sw->next_special_rr();
        next_node = _spraypoint->choose_next_hop(
                current_node, dst_group, source_step,
                pkt.flow_id(), pkt.pathid(), rr, choice);
    } else if (_cfg.ocs_mode == OxcOcsMode::KSP) {
        if (!_ksp) {
            throw logic_error("Oxc KSP router is not initialized");
        }
        uint32_t src_node = current_node;
        uint32_t path_id = numeric_limits<uint32_t>::max();
        if (pkt.has_ocs_ksp_route()) {
            src_node = pkt.ocs_ksp_src_node();
            if (pkt.ocs_ksp_dst_group() != dst_group) {
                throw logic_error("Oxc KSP packet dst_group metadata mismatch");
            }
            path_id = pkt.ocs_ksp_path_id();
        } else {
            OxcOcsKspChoice choice =
                    _cfg.ocs_choice == OxcOcsChoice::FLOW_HASH
                            ? OxcOcsKspChoice::FLOW_HASH
                            : OxcOcsKspChoice::PACKET_RR;
            uint32_t rr = _cfg.ocs_choice == OxcOcsChoice::FLOW_HASH ? 0 : sw->next_special_rr();
            path_id = _ksp->choose_path(
                    src_node, dst_group, pkt.flow_id(), pkt.pathid(), pkt.id(), rr, choice);
            if (path_id == numeric_limits<uint32_t>::max()) {
                throw runtime_error("Oxc KSP could not select a path");
            }
            pkt.set_ocs_ksp_route(src_node, dst_group, path_id);
        }
        next_node = _ksp->next_hop(src_node, dst_group, path_id, current_node);
    }

    if (next_node == numeric_limits<uint32_t>::max()) {
        throw runtime_error("Oxc OCS route ended before reaching destination group");
    }

    pkt.set_direction(UP);
    uint32_t next_l1 = l1_from_logical_node(next_node, current_member);
    return l1_to_l1_route(current_l1, next_l1, 0);
}

Route* OxcTopology::l1_to_l1_route(uint32_t src_l1, uint32_t dst_l1, uint32_t bundle) {
    return make_switch_route(
            l1_name(src_l1), l1_name(dst_l1), bundle,
            _cfg.external_linkspeed, _cfg.link_latency,
            _l1_switches.at(src_l1), _l1_switches.at(dst_l1));
}

string OxcTopology::route_key(const string& prefix, uint32_t sw, uint32_t dst, uint32_t next) const {
    return prefix + ":" + to_string(sw) + ":" + to_string(dst) + ":" + to_string(next);
}

string OxcTopology::host_route_key(uint32_t l0, uint32_t dst, int flowid) const {
    return to_string(l0) + ":" + to_string(dst) + ":" + to_string(flowid);
}

string OxcTopology::host_src_name(uint32_t rank) {
    return "SRC" + to_string(rank);
}

string OxcTopology::host_dst_name(uint32_t rank) {
    return "DST" + to_string(rank);
}

string OxcTopology::l0_name(uint32_t l0) {
    return "L0_" + to_string(l0);
}

string OxcTopology::l1_name(uint32_t l1) {
    return "L1_" + to_string(l1);
}
