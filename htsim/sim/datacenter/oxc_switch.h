// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef OXC_SWITCH_H
#define OXC_SWITCH_H

#include "callback_pipe.h"
#include "switch.h"

#include <unordered_map>

class OxcSwitch : public Switch {
public:
    enum switch_type {
        NONE = 0,
        L0 = 1,
        L1 = 2,
    };

    enum routing_strategy {
        NIX = 0,
        ECMP = 1,
        RR = 2,
    };

    OxcSwitch(
            EventList& eventlist,
            const string& name,
            switch_type type,
            uint32_t id,
            simtime_picosec switch_delay);
    ~OxcSwitch() override;

    void receivePacket(Packet& pkt) override;
    Route* getNextHop(Packet& pkt, BaseQueue* ingress_port) override;
    uint32_t getType() override { return _type; }

    void addRoute(int dst, Route* egress, packet_direction direction);
    void addHostRoute(int dst, int flowid, Route* egress);
    void addFlowRoute(int dst, int flowid, Route* egress, packet_direction direction);
    void set_special_next_hop_resolver(Route* (*resolver)(OxcSwitch*, Packet&, void*), void* context);

    // OxcTopology owns the exact switch->host queue/pipe objects, so it
    // installs host routes through addHostRoute().
    void addHostPort(int addr, int flowid, PacketSink* transport) override;

    void permute_paths(vector<FibEntry*>* routes);
    uint32_t next_special_rr();

    static void set_strategy(routing_strategy strategy) { _strategy = strategy; }
    static routing_strategy strategy() { return _strategy; }

private:
    switch_type _type;
    Pipe* _pipe;
    uint32_t _crt_route;
    uint32_t _special_crt_route;
    uint32_t _hash_salt;
    unordered_map<Packet*, bool> _packets;
    struct FlowFibEntry {
        Route* egress = nullptr;
        packet_direction direction = UP;
    };
    unordered_map<int, unordered_map<int, FlowFibEntry>> _flow_routes;
    Route* (*_special_next_hop_resolver)(OxcSwitch*, Packet&, void*);
    void* _special_next_hop_context;

    static routing_strategy _strategy;
};

#endif
