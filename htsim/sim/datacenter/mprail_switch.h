// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MPRAIL_SWITCH_H
#define MPRAIL_SWITCH_H

#include "callback_pipe.h"
#include "switch.h"

#include <unordered_map>

class MpRailSwitch : public Switch {
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

    MpRailSwitch(
            EventList& eventlist,
            const string& name,
            switch_type type,
            uint32_t id,
            simtime_picosec switch_delay);
    ~MpRailSwitch() override;

    void receivePacket(Packet& pkt) override;
    Route* getNextHop(Packet& pkt, BaseQueue* ingress_port) override;
    uint32_t getType() override { return _type; }

    void addRoute(int dst, Route* egress, packet_direction direction);
    void addHostRoute(int dst, int flowid, Route* egress);
    void addExplicitRoute(
            int dst, int flowid, Route* egress, packet_direction direction);
    void addHostPort(int addr, int flowid, PacketSink* transport) override;

    void permute_paths(vector<FibEntry*>* routes);

    static void set_strategy(routing_strategy strategy) { _strategy = strategy; }
    static routing_strategy strategy() { return _strategy; }

private:
    switch_type _type;
    Pipe* _pipe;
    uint32_t _crt_route;
    uint32_t _hash_salt;
    unordered_map<Packet*, bool> _packets;

    struct ExplicitRouteEntry {
        Route* egress = nullptr;
        packet_direction direction = static_cast<packet_direction>(0);
    };
    unordered_map<uint64_t, ExplicitRouteEntry> _explicit_routes;

    static routing_strategy _strategy;
};

#endif
