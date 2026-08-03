// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef DRAGONFLY_H
#define DRAGONFLY_H
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "config.h"
#include "dragonfly_switch.h"
#include "eventlist.h"
#include "logfile.h"
#include "loggers.h"
#include "main.h"
#include "network.h"
#include "pipe.h"
#include "randomqueue.h"
#include "switch.h"
#include "topology.h"

// Dragonfly parameters
//  p = number of hosts per switch
//  a = number of switches per group
//  h = number of links used to connect to other groups
//
//  Dragonfly parameters a, p, and h can have any value
//  To balance channel load on load-balanced traffic: a = 2p = 2h; p = h;

#ifndef QT
#define QT
typedef enum {UNDEFINED, RANDOM, ECN, COMPOSITE, PRIORITY,
              CTRL_PRIO, FAIR_PRIO, LOSSLESS, LOSSLESS_INPUT, LOSSLESS_INPUT_ECN,
              COMPOSITE_ECN, COMPOSITE_ECN_LB, SWIFT_SCHEDULER, ECN_PRIO, AEOLUS, AEOLUS_ECN} queue_type;
#endif

class DragonflyTopology : public Topology {
public:
    std::vector<Switch*> switches;

    std::vector<std::vector<Pipe*>> pipes_host_switch;
    std::vector<std::vector<Pipe*>> pipes_switch_switch;
    std::vector<std::vector<Pipe*>> pipes_switch_host;

    std::vector<std::vector<Queue*>> queues_host_switch;
    std::vector<std::vector<Queue*>> queues_switch_switch;
    std::vector<std::vector<Queue*>> queues_switch_host;

    DragonflyTopology(queue_type queue_type,
                      mem_b queue_size,
                      QueueLoggerFactory* logger_factory,
                      EventList* event_list);

    void construct_topology(uint32_t p, uint32_t a, uint32_t h);
    void load_topology(std::string path);

    virtual std::vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse);
    virtual std::vector<uint32_t>* get_neighbours(uint32_t src) { return NULL; };

    uint32_t no_of_nodes() const override { return _no_hosts; }

    uint32_t get_host_switch(uint32_t src) { return (src / _p); }
    uint32_t get_p() { return _p; }
    uint32_t get_a() { return _a; }
    uint32_t get_h() { return _h; }
    uint32_t get_no_groups() { return _no_groups; }
    uint32_t get_no_hosts() { return _no_hosts; }

    uint32_t get_group_switch(uint32_t src_group, uint32_t dst_group);
    uint32_t get_target_switch(uint32_t src_switch, uint32_t global_link);

    simtime_picosec get_max_rtt(DragonflySwitch::RoutingStrategy routing_strategy,
                                uint32_t data_packet_size,
                                uint32_t ack_packet_size);
    linkspeed_bps get_linkspeed();
    std::string get_link_latencies();

    static void set_ecn(bool enable_ecn, mem_b ecn_low, mem_b ecn_high) {
        _enable_ecn = enable_ecn;
        _ecn_low = ecn_low;
        _ecn_high = ecn_high;
    }

    static double _fail_percentage;
    static int _fail_packet_rate;

private:
    uint32_t _p, _a, _h;
    mem_b _queue_size;
    QueueLoggerFactory* _logger_factory;
    EventList* _event_list;
    queue_type _queue_type;
    uint32_t _no_groups, _no_switches, _no_hosts;

    void init_link_latencies();
    void init_pipes_queues();
    void init_network();

    inline void set_link_latency(uint32_t src_switch, uint32_t dst_switch, simtime_picosec latency);
    inline simtime_picosec get_link_latency(uint32_t src_switch, uint32_t dst_switch);

    Queue* alloc_host_queue(QueueLogger* queue_logger, linkspeed_bps linkspeed);
    Queue* alloc_switch_queue(QueueLogger* queue_logger,
                              linkspeed_bps linkspeed,
                              mem_b queue_size,
                              bool is_failing);

    std::vector<std::unordered_map<uint32_t, simtime_picosec>> _link_latencies;

    static bool _enable_ecn;
    static mem_b _ecn_low;
    static mem_b _ecn_high;

    static linkspeed_bps _link_speed_global;
    static linkspeed_bps _link_speed_local;
    static linkspeed_bps _link_speed_host;

    static simtime_picosec _link_latency_global;
    static simtime_picosec _link_latency_local;
    static simtime_picosec _link_latency_host;

    static simtime_picosec _switch_latency;
};

#endif
