// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "mprail_switch.h"

#include "eth_pause_packet.h"
#include "routetable.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace std;

#define MPRAIL_MIX(a, b, c)                    \
    do {                                       \
        a -= b; a -= c; a ^= (c >> 13);        \
        b -= c; b -= a; b ^= (a << 8);         \
        c -= a; c -= b; c ^= (b >> 13);        \
        a -= b; a -= c; a ^= (c >> 12);        \
        b -= c; b -= a; b ^= (a << 16);        \
        c -= a; c -= b; c ^= (b >> 5);         \
        a -= b; a -= c; a ^= (c >> 3);         \
        b -= c; b -= a; b ^= (a << 10);        \
        c -= a; c -= b; c ^= (b >> 15);        \
    } while (0)

static inline uint32_t mprail_hash(
        uint32_t target1, uint32_t target2 = 0, uint32_t target3 = 0) {
    uint32_t a = 0x9e3779b9;
    uint32_t b = 0x9e3779b9;
    uint32_t c = 0;
    b += target3;
    c += target2;
    a += target1;
    MPRAIL_MIX(a, b, c);
    return c;
}

#undef MPRAIL_MIX

MpRailSwitch::routing_strategy MpRailSwitch::_strategy = MpRailSwitch::NIX;

MpRailSwitch::MpRailSwitch(
        EventList& eventlist,
        const string& name,
        switch_type type,
        uint32_t id,
        simtime_picosec switch_delay)
    : Switch(eventlist, name),
      _type(type),
      _pipe(new CallbackPipe(switch_delay, eventlist, this)),
      _crt_route(0),
      _hash_salt(random()) {
    _id = id;
    _fib = new RouteTable();
}

MpRailSwitch::~MpRailSwitch() {
    delete _pipe;
    delete _fib;
}

void MpRailSwitch::receivePacket(Packet& pkt) {
    if (pkt.type() == ETH_PAUSE) {
        EthPausePacket* pause = reinterpret_cast<EthPausePacket*>(&pkt);
        for (BaseQueue* port : _ports) {
            if (port->getRemoteEndpoint()
                    && static_cast<Switch*>(port->getRemoteEndpoint())->getID()
                       == pause->senderID()) {
                port->receivePacket(pkt);
                break;
            }
        }
        return;
    }

    if (_packets.find(&pkt) == _packets.end()) {
        _packets[&pkt] = true;
        const Route* next_hop = getNextHop(pkt, nullptr);
        pkt.set_route(*next_hop);
        _pipe->receivePacket(pkt);
    } else {
        _packets.erase(&pkt);
        pkt.sendOn();
    }
}

void MpRailSwitch::addRoute(int dst, Route* egress, packet_direction direction) {
    if (!egress) {
        throw invalid_argument("MpRailSwitch::addRoute got null egress route");
    }
    _fib->addRoute(dst, egress, 1, direction);
}

void MpRailSwitch::addHostRoute(int dst, int flowid, Route* egress) {
    if (!egress) {
        throw invalid_argument("MpRailSwitch::addHostRoute got null egress route");
    }
    _fib->addHostRoute(dst, egress, flowid);
}

void MpRailSwitch::addFlowRoute(
        int dst, int flowid, Route* egress, packet_direction direction) {
    if (!egress) {
        throw invalid_argument("MpRailSwitch::addFlowRoute got null egress route");
    }
    _flow_routes[dst][flowid] = FlowFibEntry{egress, direction};
}

void MpRailSwitch::addHostPort(int addr, int flowid, PacketSink* transport) {
    (void)addr;
    (void)flowid;
    (void)transport;
    throw logic_error("MpRailSwitch host routes require MpRailTopology link context");
}

void MpRailSwitch::permute_paths(vector<FibEntry*>* routes) {
    int len = routes ? static_cast<int>(routes->size()) : 0;
    for (int i = 0; i < len; i++) {
        int ix = random() % (len - i);
        FibEntry* tmp = (*routes)[ix];
        (*routes)[ix] = (*routes)[len - 1 - i];
        (*routes)[len - 1 - i] = tmp;
    }
}

Route* MpRailSwitch::getNextHop(Packet& pkt, BaseQueue* ingress_port) {
    (void)ingress_port;

    auto dst_it = _flow_routes.find(pkt.dst());
    if (dst_it != _flow_routes.end()) {
        auto flow_it = dst_it->second.find(pkt.flow_id());
        if (flow_it != dst_it->second.end()) {
            pkt.set_direction(flow_it->second.direction);
            return flow_it->second.egress;
        }
    }

    HostFibEntry* host_entry = _fib->getHostRoute(pkt.dst(), pkt.flow_id());
    if (host_entry) {
        pkt.set_direction(DOWN);
        return host_entry->getEgressPort();
    }

    vector<FibEntry*>* available_hops = _fib->getRoutes(pkt.dst());
    if (!available_hops || available_hops->empty()) {
        cerr << "MpRailSwitch " << nodename()
             << " id " << _id
             << " type " << _type
             << " has no route for dst " << pkt.dst()
             << " flow " << pkt.flow_id()
             << " src " << pkt.src()
             << endl;
        abort();
    }

    uint32_t choice = 0;
    if (available_hops->size() > 1) {
        switch (_strategy) {
        case NIX:
            cerr << "MpRailSwitch route strategy not set" << endl;
            abort();
        case ECMP:
            choice = mprail_hash(pkt.flow_id(), pkt.pathid(), _hash_salt)
                     % available_hops->size();
            break;
        case RR:
            if (pkt.size() < 128) {
                choice = mprail_hash(pkt.flow_id(), pkt.pathid(), _hash_salt)
                         % available_hops->size();
            } else {
                if (_crt_route >= available_hops->size()) {
                    _crt_route = 0;
                    permute_paths(available_hops);
                }
                choice = _crt_route++ % available_hops->size();
            }
            break;
        }
    }

    FibEntry* entry = available_hops->at(choice);
    pkt.set_direction(entry->getDirection());
    return entry->getEgressPort();
}
