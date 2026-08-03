// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "dragonfly_topology.h"
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "compositequeue.h"
#include "connection_matrix.h"
#include "dragonfly_switch.h"
#include "main.h"
#include "switch.h"

// Use tokenize from connection matrix
extern void tokenize(std::string const& str, const char delim, std::vector<std::string>& out);

// Default: ECN disabled
bool DragonflyTopology::_enable_ecn = false;
mem_b DragonflyTopology::_ecn_low = 0;
mem_b DragonflyTopology::_ecn_high = 0;

// Default: all links 200 Gbps
linkspeed_bps DragonflyTopology::_link_speed_global = 200ULL * 1000000000ULL;
linkspeed_bps DragonflyTopology::_link_speed_local = 200ULL * 1000000000ULL;
linkspeed_bps DragonflyTopology::_link_speed_host = 200ULL * 1000000000ULL;

// Default: global links 1000 ns, local links 500 ns, host links 200 ns
simtime_picosec DragonflyTopology::_link_latency_global = 1000 * 1000;
simtime_picosec DragonflyTopology::_link_latency_local = 500 * 1000;
simtime_picosec DragonflyTopology::_link_latency_host = 200 * 1000;

// Default: all switches 0 ns
simtime_picosec DragonflyTopology::_switch_latency = 0 * 1000;

// Default: no link failures
double DragonflyTopology::_fail_percentage = 0.0;
int DragonflyTopology::_fail_packet_rate = 0;

std::string ntoa(double n);
std::string itoa(uint64_t n);

DragonflyTopology::DragonflyTopology(queue_type queue_type,
                                     mem_b queue_size,
                                     QueueLoggerFactory* logger_factory,
                                     EventList* event_list) {
    _queue_type = queue_type;
    _queue_size = queue_size;
    _logger_factory = logger_factory;
    _event_list = event_list;

    _p = 0;
    _a = 0;
    _h = 0;
}

void DragonflyTopology::construct_topology(uint32_t p, uint32_t a, uint32_t h) {
    assert(p > 0 && a > 0 && h > 0);

    _p = p;
    _a = a;
    _h = h;

    _no_groups = a * h + 1;
    _no_switches = a * _no_groups;
    _no_hosts = p * _no_switches;

    init_link_latencies();
    init_pipes_queues();
    init_network();
}

void DragonflyTopology::load_topology(std::string path) {
    ifstream file(path + "/dragonfly.topo");
    if (!file.is_open()) {
        cerr << "ERROR: Could not open: " << path << endl;
        exit(-1);
    }
    uint32_t count = 0;
    std::string line;
    while (count < 10 && std::getline(file, line)) {
        std::vector<std::string> tokens;
        tokenize(line, ' ', tokens);
        if (tokens.size() == 0 || tokens[0][0] == '#')
            continue;

        count++;

        if (tokens[0] == "p")
            _p = stoi(tokens[1]);
        else if (tokens[0] == "a")
            _a = stoi(tokens[1]);
        else if (tokens[0] == "h")
            _h = stoi(tokens[1]);
        else if (tokens[0] == "Switch_latency_ns")
            _switch_latency = timeFromNs(stoi(tokens[1]));
        else if (tokens[0] == "Link_speed_global_Gbps")
            _link_speed_global = speedFromGbps(stoi(tokens[1]));
        else if (tokens[0] == "Link_speed_local_Gbps")
            _link_speed_local = speedFromGbps(stoi(tokens[1]));
        else if (tokens[0] == "Link_speed_host_Gbps")
            _link_speed_host = speedFromGbps(stoi(tokens[1]));
        else if (tokens[0] == "Link_latency_global_ns")
            _link_latency_global = timeFromNs(stoi(tokens[1]));
        else if (tokens[0] == "Link_latency_local_ns")
            _link_latency_local = timeFromNs(stoi(tokens[1]));
        else if (tokens[0] == "Link_latency_host_ns")
            _link_latency_host = timeFromNs(stoi(tokens[1]));
        else
            throw std::logic_error("Unexpected case");
    }

    assert(_p > 0 && _a > 0 && _h > 0);
    _no_groups = _a * _h + 1;
    _no_switches = _a * _no_groups;
    _no_hosts = _p * _no_switches;
    init_link_latencies();

    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        tokenize(line, ' ', tokens);
        if (tokens.size() == 0 || tokens[0][0] == '#')
            continue;

        if (tokens[0] == "Link_latency_ns") {
            uint32_t i = stoi(tokens[1]);
            uint32_t j = stoi(tokens[2]);
            simtime_picosec latency = timeFromNs(stoi(tokens[3]));
            set_link_latency(i, j, latency);
        } else
            throw std::logic_error("Unexpected case");
    }

    init_pipes_queues();
    init_network();
}

void DragonflyTopology::init_link_latencies() {
    _link_latencies.resize(_no_switches, std::unordered_map<uint32_t, simtime_picosec>());

    // i = each switch
    for (uint32_t i = 0; i < _no_switches; i++) {
        uint32_t group_id = i / _a;

        // Within group (full mesh)
        // j = each switch within same group with higher ID
        for (uint32_t j = i + 1; j < (group_id + 1) * _a; j++) {
            set_link_latency(i, j, _link_latency_local);
        }

        // Between groups
        // j = each switch to form cross-group connection with
        for (uint32_t ih = 0; ih < _h; ih++) {
            uint32_t right_steps = ((i - (group_id * _a)) * _h) + 1 + ih;
            uint32_t dst_group = (group_id + right_steps) % _no_groups;
            uint32_t j = (dst_group * _a) + (_no_groups - right_steps - 1) / _h;
            set_link_latency(i, j, _link_latency_global);
        }
    }
}

void DragonflyTopology::init_pipes_queues() {
    switches.resize(_no_switches, NULL);

    queues_host_switch.resize(_no_hosts, std::vector<Queue*>(_no_switches));
    queues_switch_switch.resize(_no_switches, std::vector<Queue*>(_no_switches));
    queues_switch_host.resize(_no_switches, std::vector<Queue*>(_no_hosts));

    pipes_host_switch.resize(_no_hosts, std::vector<Pipe*>(_no_switches));
    pipes_switch_switch.resize(_no_switches, std::vector<Pipe*>(_no_switches));
    pipes_switch_host.resize(_no_switches, std::vector<Pipe*>(_no_hosts));

    // Initializes all queues and pipes to NULL
    for (uint32_t i = 0; i < _no_switches; i++) {
        for (uint32_t j = 0; j < _no_switches; j++) {
            queues_switch_switch[i][j] = NULL;
            pipes_switch_switch[i][j] = NULL;
        }

        for (uint32_t j = 0; j < _no_hosts; j++) {
            queues_switch_host[i][j] = NULL;
            queues_host_switch[j][i] = NULL;
            pipes_switch_host[i][j] = NULL;
            pipes_host_switch[j][i] = NULL;
        }
    }
}

void DragonflyTopology::init_network() {
    QueueLogger* queue_logger;

    // Create the switches
    for (uint32_t i = 0; i < _no_switches; i++)
        switches[i] = new DragonflySwitch(*_event_list, "Switch_" + ntoa(i),
                                          DragonflySwitch::GENERAL, i, _switch_latency, this);

    // Create all links between Switches and Hosts
    // i = each switch
    // j = each host per switch
    for (uint32_t i = 0; i < _no_switches; i++) {
        for (uint32_t ip = 0; ip < _p; ip++) {
            uint32_t j = i * _p + ip;

            // Switch to Host
            queue_logger = _logger_factory == NULL ? NULL : _logger_factory->createQueueLogger();

            queues_switch_host[i][j] =
                alloc_switch_queue(queue_logger, _link_speed_host, _queue_size, false);
            queues_switch_host[i][j]->setName("SW" + ntoa(i) + "->DST" + ntoa(j));

            pipes_switch_host[i][j] = new Pipe(_link_latency_host, *_event_list);
            pipes_switch_host[i][j]->setName("Pipe-SW" + ntoa(i) + "->DST" + ntoa(j));

            // Host to Switch
            queue_logger = _logger_factory == NULL ? NULL : _logger_factory->createQueueLogger();

            queues_host_switch[j][i] = alloc_host_queue(queue_logger, _link_speed_host);
            queues_host_switch[j][i]->setName("SRC" + ntoa(j) + "->SW" + ntoa(i));

            pipes_host_switch[j][i] = new Pipe(_link_latency_host, *_event_list);
            pipes_host_switch[j][i]->setName("Pipe-SRC" + ntoa(j) + "->SW" + ntoa(i));

            // Add ports and set remote endpoints
            switches[i]->addPort(queues_switch_host[i][j]);
            switches[i]->addPort(queues_host_switch[j][i]);
            queues_host_switch[j][i]->setRemoteEndpoint(switches[i]);
        }
    }

    int total_queues = _no_switches * ((_a - 1) + _h);
    int no_failures = total_queues * _fail_percentage;
    std::unordered_set<int> failed_queues;

    if (no_failures > 0) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, total_queues - 1);
        while (failed_queues.size() < static_cast<size_t>(no_failures))
            failed_queues.insert(dist(gen));
    }

    int queue_index = 0;

    // Create all links between Switches and Switches
    // i = each switch
    for (uint32_t i = 0; i < _no_switches; i++) {
        uint32_t group_id = i / _a;
        bool fail_queue = false;

        // Within group (full mesh)
        // j = each switch within same group with higher ID
        for (uint32_t j = i + 1; j < (group_id + 1) * _a; j++) {
            // i → j
            queue_logger = _logger_factory == NULL ? NULL : _logger_factory->createQueueLogger();

            fail_queue = failed_queues.find(queue_index) != failed_queues.end();
            queues_switch_switch[i][j] =
                alloc_switch_queue(queue_logger, _link_speed_local, _queue_size, fail_queue);
            queues_switch_switch[i][j]->setName("SW" + ntoa(i) + "-I->SW" + ntoa(j));

            pipes_switch_switch[i][j] = new Pipe(get_link_latency(i, j), *_event_list);
            pipes_switch_switch[i][j]->setName("Pipe-SW" + ntoa(i) + "-I->SW" + ntoa(j));
            queue_index++;

            // j → i
            queue_logger = _logger_factory == NULL ? NULL : _logger_factory->createQueueLogger();

            fail_queue = failed_queues.find(queue_index) != failed_queues.end();
            queues_switch_switch[j][i] =
                alloc_switch_queue(queue_logger, _link_speed_local, _queue_size, fail_queue);
            queues_switch_switch[j][i]->setName("SW" + ntoa(j) + "-I->SW" + ntoa(i));

            pipes_switch_switch[j][i] = new Pipe(get_link_latency(i, j), *_event_list);
            pipes_switch_switch[j][i]->setName("Pipe-SW" + ntoa(j) + "-I->SW" + ntoa(i));
            queue_index++;

            // Add ports and set remote endpoints
            switches[i]->addPort(queues_switch_switch[i][j]);
            switches[j]->addPort(queues_switch_switch[j][i]);
            queues_switch_switch[i][j]->setRemoteEndpoint(switches[j]);
            queues_switch_switch[j][i]->setRemoteEndpoint(switches[i]);
        }

        // Between groups
        // j = each switch to form cross-group connection with
        for (uint32_t ih = 0; ih < _h; ih++) {
            uint32_t right_steps = ((i - (group_id * _a)) * _h) + 1 + ih;
            uint32_t dst_group = (group_id + right_steps) % _no_groups;
            uint32_t j = (dst_group * _a) + (_no_groups - right_steps - 1) / _h;

            // i → j
            queue_logger = _logger_factory == NULL ? NULL : _logger_factory->createQueueLogger();

            fail_queue = failed_queues.find(queue_index) != failed_queues.end();
            queues_switch_switch[i][j] =
                alloc_switch_queue(queue_logger, _link_speed_global, _queue_size, fail_queue);
            queues_switch_switch[i][j]->setName("SW" + ntoa(i) + "-G->SW" + ntoa(j));

            pipes_switch_switch[i][j] = new Pipe(get_link_latency(i, j), *_event_list);
            pipes_switch_switch[i][j]->setName("Pipe-SW" + ntoa(i) + "-G->SW" + ntoa(j));
            queue_index++;

            // Add ports and set remote endpoints
            switches[i]->addPort(queues_switch_switch[i][j]);
            queues_switch_switch[i][j]->setRemoteEndpoint(switches[j]);
        }
    }
}

std::vector<const Route*>* DragonflyTopology::get_bidir_paths(uint32_t src,
                                                              uint32_t dest,
                                                              bool reverse) {
    // TODO: implement (if needed)
    throw std::logic_error("Not implemented");
}

uint32_t DragonflyTopology::get_group_switch(uint32_t src_group, uint32_t dst_group) {
    uint32_t right_steps;
    if (src_group < dst_group)
        right_steps = dst_group - src_group;
    else
        right_steps = (_no_groups + dst_group) - src_group;
    return (src_group * _a) + (right_steps - 1) / _h;
}

uint32_t DragonflyTopology::get_target_switch(uint32_t src_switch, uint32_t global_link) {
    uint32_t src_group = src_switch / _a;
    uint32_t src_switch_index = src_switch - (src_group * _a);
    uint32_t right_steps = (src_switch_index * _h) + global_link + 1;
    uint32_t dst_group = (src_group + right_steps) % _no_groups;
    return (dst_group * _a) + (_a - 1) - src_switch_index;
}

simtime_picosec DragonflyTopology::get_max_rtt(DragonflySwitch::RoutingStrategy routing_strategy,
                                               uint32_t data_packet_size,
                                               uint32_t ack_packet_size) {
    // WARNING: does not support arbitrary specific link latencies
    // Transmission delays
    simtime_picosec t_delay_data_global =
        timeFromNs(((double)data_packet_size * 8) / speedAsGbps(_link_speed_global));
    simtime_picosec t_delay_data_local =
        timeFromNs(((double)data_packet_size * 8) / speedAsGbps(_link_speed_local));
    simtime_picosec t_delay_data_host =
        timeFromNs(((double)data_packet_size * 8) / speedAsGbps(_link_speed_host));

    simtime_picosec t_delay_ack_global =
        timeFromNs(((double)ack_packet_size * 8) / speedAsGbps(_link_speed_global));
    simtime_picosec t_delay_ack_local =
        timeFromNs(((double)ack_packet_size * 8) / speedAsGbps(_link_speed_local));
    simtime_picosec t_delay_ack_host =
        timeFromNs(((double)ack_packet_size * 8) / speedAsGbps(_link_speed_host));

    simtime_picosec data, ack;

    switch (routing_strategy) {
        case DragonflySwitch::MINIMAL:
        case DragonflySwitch::SOURCE:
        case DragonflySwitch::VALIANT:
        case DragonflySwitch::UGAL_L: {
            // Worst case routing
            // Data: (3,2), Ack: (2,1)
            data = (1 * (t_delay_data_host + _link_latency_host)) +
                   (3 * (_switch_latency + t_delay_data_local + _link_latency_local)) +
                   (2 * (_switch_latency + t_delay_data_global + _link_latency_global)) +
                   (1 * (_switch_latency + t_delay_data_host + _link_latency_host));

            ack = (1 * (t_delay_ack_host + _link_latency_host)) +
                  (2 * (_switch_latency + t_delay_ack_local + _link_latency_local)) +
                  (1 * (_switch_latency + t_delay_ack_global + _link_latency_global)) +
                  (1 * (_switch_latency + t_delay_ack_host + _link_latency_host));

            return data + ack;
        }

        default:
            throw std::logic_error("Unexpected case");
    }
}

linkspeed_bps DragonflyTopology::get_linkspeed() {
    // WARNING: same linkspeed assumed for now for CC
    assert(_link_speed_global == _link_speed_local);
    assert(_link_speed_local == _link_speed_host);
    return _link_speed_global;
}

std::string DragonflyTopology::get_link_latencies() {
    return std::to_string(_link_latency_global) + ":" + std::to_string(_link_latency_local) + ":" +
           std::to_string(_link_latency_host);
}

inline void DragonflyTopology::set_link_latency(uint32_t src_switch,
                                                uint32_t dst_switch,
                                                simtime_picosec latency) {
    _link_latencies[std::min(src_switch, dst_switch)][std::max(src_switch, dst_switch)] = latency;
}

inline simtime_picosec DragonflyTopology::get_link_latency(uint32_t src_switch,
                                                           uint32_t dst_switch) {
    return _link_latencies[std::min(src_switch, dst_switch)][std::max(src_switch, dst_switch)];
}

Queue* DragonflyTopology::alloc_host_queue(QueueLogger* queue_logger, linkspeed_bps linkspeed) {
    mem_b max_size = memFromPkt(FEEDER_BUFFER);
    return new FairPriorityQueue(linkspeed, max_size, *_event_list, queue_logger);
}

Queue* DragonflyTopology::alloc_switch_queue(QueueLogger* queue_logger,
                                             linkspeed_bps linkspeed,
                                             mem_b queue_size,
                                             bool is_failing) {
    switch (_queue_type) {
        case RANDOM: {
            mem_b max_size = _queue_size;
            mem_b drop = memFromPkt(RANDOM_BUFFER);
            return new RandomQueue(linkspeed, max_size, *_event_list, queue_logger, drop);
        }

        case COMPOSITE: {
            mem_b max_size = _queue_size;
            bool trim_disable = DragonflySwitch::get_trim_disable();
            uint16_t trim_size = DragonflySwitch::get_trim_size();

            CompositeQueue* q = new CompositeQueue(linkspeed, max_size, *_event_list, queue_logger,
                                                   trim_size, trim_disable);
            if (_enable_ecn)
                q->set_ecn_thresholds(_ecn_low, _ecn_high);

            if (is_failing)
                q->set_fail_rate(_fail_packet_rate);

            return q;
        }

        default:
            throw std::logic_error("Not implemented");
    }
}
