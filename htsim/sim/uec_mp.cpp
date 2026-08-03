
#include "uec_mp.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
void tokenize(const std::string& str, char delim, std::vector<std::string>& out) {
    std::stringstream ss(str);
    std::string s;
    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }
}
}  // namespace


UecMpOblivious::UecMpOblivious(uint16_t no_of_paths,
                               bool debug)
    : UecMultipath(debug),
      _no_of_paths(no_of_paths),
      _current_ev_index(0)
      {

    _path_random = rand() % UINT16_MAX;  // random upper bits of EV
    _path_xor = rand() % _no_of_paths;

    if (_debug)
        cout << "Multipath"
            << " Oblivious"
            << " _no_of_paths " << _no_of_paths
            << " _path_random " << _path_random
            << " _path_xor " << _path_xor
            << endl;
}

void UecMpOblivious::processEv(uint32_t path_id, PathFeedback feedback) {
    return;
}

uint32_t UecMpOblivious::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    // _no_of_paths must be a power of 2
    uint16_t mask = _no_of_paths - 1;
    uint16_t entropy = (_current_ev_index ^ _path_xor) & mask;

    // set things for next time
    _current_ev_index++;
    if (_current_ev_index == _no_of_paths) {
        _current_ev_index = 0;
        _path_xor = rand() & mask;
    }

    entropy |= _path_random ^ (_path_random & mask);  // set upper bits
    return entropy;
}


UecMpBitmap::UecMpBitmap(uint16_t no_of_paths, bool debug)
    : UecMultipath(debug),
      _no_of_paths(no_of_paths),
      _current_ev_index(0),
      _ev_skip_bitmap(),
      _ev_skip_count(0)
      {

    _max_penalty = 15;

    _path_random = rand() % 0xffff;  // random upper bits of EV
    _path_xor = rand() % _no_of_paths;

    _ev_skip_bitmap.resize(_no_of_paths);
    for (uint32_t i = 0; i < _no_of_paths; i++) {
        _ev_skip_bitmap[i] = 0;
    }

    if (_debug)
        cout << "Multipath"
            << " Bitmap"
            << " _no_of_paths " << _no_of_paths
            << " _path_random " << _path_random
            << " _path_xor " << _path_xor
            << " _max_penalty " << (uint32_t)_max_penalty
            << endl;
}

void UecMpBitmap::processEv(uint32_t path_id, PathFeedback feedback) {
    // _no_of_paths must be a power of 2
    uint16_t mask = _no_of_paths - 1;
    path_id &= mask;  // only take the relevant bits for an index

    if (feedback != PathFeedback::PATH_GOOD && !_ev_skip_bitmap[path_id])
        _ev_skip_count++;

    uint8_t penalty = 0;

    if (feedback == PathFeedback::PATH_ECN)
        penalty = 1;
    else if (feedback == PathFeedback::PATH_NACK)
        penalty = 4;
    else if (feedback == PathFeedback::PATH_TIMEOUT)
        penalty = _max_penalty;

    _ev_skip_bitmap[path_id] += penalty;
    if (_ev_skip_bitmap[path_id] > _max_penalty) {
        _ev_skip_bitmap[path_id] = _max_penalty;
    }
}

uint32_t UecMpBitmap::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    // _no_of_paths must be a power of 2
    uint16_t mask = _no_of_paths - 1;
    uint16_t entropy = (_current_ev_index ^ _path_xor) & mask;
    bool flag = false;
    int counter = 0;
    while (_ev_skip_bitmap[entropy] > 0) {
        if (flag == false){
            _ev_skip_bitmap[entropy]--;
            if (!_ev_skip_bitmap[entropy]){
                assert(_ev_skip_count>0);
                _ev_skip_count--;
            }
        }

        flag = true;
        counter ++;
        if (counter > _no_of_paths){
            break;
        }
        _current_ev_index++;
        if (_current_ev_index == _no_of_paths) {
            _current_ev_index = 0;
            _path_xor = rand() & mask;
        }
        entropy = (_current_ev_index ^ _path_xor) & mask;
    }

    // set things for next time
    _current_ev_index++;
    if (_current_ev_index == _no_of_paths) {
        _current_ev_index = 0;
        _path_xor = rand() & mask;
    }

    entropy |= _path_random ^ (_path_random & mask);  // set upper bits
    return entropy;
}

UecMpReps::UecMpReps(uint16_t no_of_paths, bool debug, bool is_trimming_enabled)
    : UecMultipath(debug),
      _no_of_paths(no_of_paths),
      _crt_path(0),
      _is_trimming_enabled(is_trimming_enabled) {

    circular_buffer_reps = new CircularBufferREPS<uint16_t>(CircularBufferREPS<uint16_t>::repsBufferSize);

    if (_debug)
        cout << "Multipath"
            << " REPS"
            << " _no_of_paths " << _no_of_paths
            << endl;
}

void UecMpReps::processEv(uint32_t path_id, PathFeedback feedback) {

    if ((feedback == PATH_TIMEOUT) && !circular_buffer_reps->isFrozenMode() && circular_buffer_reps->explore_counter == 0) {
        if (_is_trimming_enabled) { // If we have trimming enabled
            circular_buffer_reps->setFrozenMode(true);
            circular_buffer_reps->can_exit_frozen_mode = EventList::getTheEventList().now() +  circular_buffer_reps->exit_freeze_after;
        } else {
            cout << timeAsUs(EventList::getTheEventList().now()) << "REPS currently requires trimming in this implementation." << endl;
            exit(EXIT_FAILURE); // If we reach this point, it means we are trying to enter freezing mode without trimming enabled.
        } // In this version of REPS, we do not enter freezing mode without trimming enabled. Check the REPS paper to implement it also without trimming.
    }

    if (circular_buffer_reps->isFrozenMode() && EventList::getTheEventList().now() > circular_buffer_reps->can_exit_frozen_mode) {
        circular_buffer_reps->setFrozenMode(false);
        circular_buffer_reps->resetBuffer();
        circular_buffer_reps->explore_counter = 16;
    }

    if ((feedback == PATH_GOOD) && !circular_buffer_reps->isFrozenMode()) {
        circular_buffer_reps->add(path_id);
    } else if (circular_buffer_reps->isFrozenMode() && (feedback == PATH_GOOD)) {
        circular_buffer_reps->add(path_id);
    }
}

uint32_t UecMpReps::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    if (circular_buffer_reps->explore_counter > 0) {
        circular_buffer_reps->explore_counter--;
        return rand() % _no_of_paths;
    }

    if (circular_buffer_reps->isFrozenMode()) {
        if (circular_buffer_reps->isEmpty()) {
            return rand() % _no_of_paths;
        } else {
            return circular_buffer_reps->remove_frozen();
        }
    } else {
        if (circular_buffer_reps->isEmpty() || circular_buffer_reps->getNumberFreshEntropies() == 0) {
            return _crt_path = rand() % _no_of_paths;
        } else {
            return circular_buffer_reps->remove_earliest_fresh();
        }
    }
}


UecMpRepsLegacy::UecMpRepsLegacy(uint16_t no_of_paths, bool debug)
    : UecMultipath(debug),
      _no_of_paths(no_of_paths),
      _crt_path(0) {

    if (_debug)
        cout << "Multipath"
            << " REPS"
            << " _no_of_paths " << _no_of_paths
            << endl;
}

void UecMpRepsLegacy::processEv(uint32_t path_id, PathFeedback feedback) {
    if (feedback == PATH_GOOD){
        _next_pathid.push_back(path_id);
        if (_debug){
            cout << timeAsUs(EventList::getTheEventList().now()) << " " << _debug_tag << " REPS Add " << path_id << " " << _next_pathid.size() << endl;
        }
    }
}

uint32_t UecMpRepsLegacy::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    if (seq_sent < min(cur_cwnd_in_pkts, (uint64_t)_no_of_paths)) {
        _crt_path++;
        if (_crt_path == _no_of_paths) {
            _crt_path = 0;
        }

        if (_debug) 
            cout << timeAsUs(EventList::getTheEventList().now()) << " " << _debug_tag << " REPS FirstWindow " << _crt_path << endl;

    } else {
        if (_next_pathid.empty()) {
            assert(_no_of_paths > 0);
		    _crt_path = random() % _no_of_paths;

            if (_debug) 
                cout << timeAsUs(EventList::getTheEventList().now()) << " " << _debug_tag << " REPS Steady " << _crt_path << endl;

        } else {
            _crt_path = _next_pathid.front();
            _next_pathid.pop_front();

            if (_debug) 
                cout << timeAsUs(EventList::getTheEventList().now()) << " " << _debug_tag << " REPS Recycle " << _crt_path << " " << _next_pathid.size() << endl;

        }
    }
    return _crt_path;
}

optional<uint32_t> UecMpRepsLegacy::nextEntropyRecycle() {
    if (_next_pathid.empty()) {
        return {};
    } else {
        _crt_path = _next_pathid.front();
        _next_pathid.pop_front();

        if (_debug) 
            cout << timeAsUs(EventList::getTheEventList().now()) << " " << _debug_tag << " MIXED Recycle " << _crt_path << " " << _next_pathid.size() << endl;
        return { _crt_path };
    }
}


UecMpMixed::UecMpMixed(uint16_t no_of_paths, bool debug)
    : UecMultipath(debug),
      _bitmap(UecMpBitmap(no_of_paths, debug)),
      _reps_legacy(UecMpRepsLegacy(no_of_paths, debug))
      {
}

void UecMpMixed::set_debug_tag(string debug_tag) {
    _bitmap.set_debug_tag(debug_tag);
    _reps_legacy.set_debug_tag(debug_tag);
}

void UecMpMixed::processEv(uint32_t path_id, PathFeedback feedback) {
    _bitmap.processEv(path_id, feedback);
    _reps_legacy.processEv(path_id, feedback);
}

uint32_t UecMpMixed::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    auto reps_val = _reps_legacy.nextEntropyRecycle();
    if (reps_val.has_value()) {
        return reps_val.value();
    } else {
        return _bitmap.nextEntropy(seq_sent, cur_cwnd_in_pkts);
    }
}

UecMpEcmp::UecMpEcmp(uint16_t no_of_paths, bool debug)
    : UecMultipath(debug),
      _crt_path(0) {
    if (_debug)
        cout << "Multipath"
            << " ECMP"
            << " _no_of_paths " << no_of_paths
            << endl;
    _crt_path = rand() % no_of_paths;
}

void UecMpEcmp::processEv(uint32_t path_id, PathFeedback feedback) {
    // No OP in ECMP
    return;
}

uint32_t UecMpEcmp::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    // Always same path for a given flow in ECMP
    return _crt_path;
}

std::map<UecMpSource::FlowPair, simtime_picosec> UecMpSource::_last_ack_timestamp;
std::map<UecMpSource::FlowPair, std::deque<uint32_t>> UecMpSource::_good_paths;
std::map<UecMpSource::FlowPair, std::set<uint32_t>> UecMpSource::_bad_paths;

UecMpSource::UecMpSource(const string& base_host_table_path,
                         uint32_t src,
                         uint32_t dst,
                         uint32_t hosts_per_switch,
                         bool debug)
    : UecMpSource(base_host_table_path, src, dst, hosts_per_switch, Strategy::RANDOM,
                  SpritzConfig{}, 0, 0, debug) {
}

UecMpSource::UecMpSource(const string& base_host_table_path,
                         uint32_t src,
                         uint32_t dst,
                         uint32_t hosts_per_switch,
                         Strategy strategy,
                         const SpritzConfig& config,
                         simtime_picosec base_rtt,
                         mem_b flow_size,
                         bool debug)
    : UecMultipath(debug),
      _strategy(strategy),
      _config(config),
      _flow_pair(src, dst),
      _paths(),
      _path_index_map(),
      _weights(),
      _rng(src * 2654435761U + dst),
      _dist(0, 0),
      _dist_weighted(),
      _base_rtt(base_rtt),
      _flow_size(flow_size),
      _packet_count(0),
      _ecn_count(0),
      _first_rtt(true),
      _last_timestamp(EventList::getTheEventList().now()),
      _ecn_counts() {
    load_paths(base_host_table_path, src, dst, hosts_per_switch);

    if (_last_ack_timestamp.find(_flow_pair) == _last_ack_timestamp.end())
        _last_ack_timestamp[_flow_pair] = EventList::getTheEventList().now();
    if (_good_paths.find(_flow_pair) == _good_paths.end())
        _good_paths[_flow_pair] = std::deque<uint32_t>{};

    if (_base_rtt > 0 &&
        EventList::getTheEventList().now() > _last_ack_timestamp[_flow_pair] + _base_rtt) {
        _good_paths[_flow_pair].clear();
    }
}

void UecMpSource::load_paths(const string& base_host_table_path,
                             uint32_t src,
                             uint32_t dst,
                             uint32_t hosts_per_switch) {
    if (hosts_per_switch == 0) {
        throw std::logic_error("hosts_per_switch must be > 0");
    }

    uint32_t src_switch = src / hosts_per_switch;
    uint32_t dst_switch = dst / hosts_per_switch;
    if (src_switch == dst_switch) {
        _paths.push_back(0);
        _path_index_map[0] = 0;
        _weights.push_back(1.0);
        _dist = std::uniform_int_distribution<size_t>(0, _paths.size() - 1);
        update_weighted_dist();
        return;
    }

    std::string file_path = base_host_table_path + "/host_table/" + std::to_string(src_switch) + ".lt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open: " << file_path << std::endl;
        exit(-1);
    }

    std::string line;
    bool dst_entry_found = false;
    while (!dst_entry_found && std::getline(file, line)) {
        std::vector<std::string> tokens;
        tokenize(line, ' ', tokens);
        if (tokens.size() == 1 && std::stoul(tokens[0]) == dst_switch)
            dst_entry_found = true;
    }

    if (!dst_entry_found) {
        std::cerr << "ERROR: host table missing destination entry for switch " << dst_switch << std::endl;
        exit(-1);
    }

    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        tokenize(line, ' ', tokens);
        if (tokens.size() < 2)
            break;

        int local_hops = std::stoi(tokens[0]);
        int global_hops = std::stoi(tokens[1]);
        double latency = local_hops * 25.0 + global_hops * 500.0;
        if (latency <= 0.0)
            latency = 1.0;

        for (std::size_t i = 2; i + 1 < tokens.size(); i += 2) {
            uint32_t hop_one_switch = static_cast<uint32_t>(std::stoul(tokens[i]));
            uint32_t hop_two_switch = static_cast<uint32_t>(std::stoul(tokens[i + 1]));
            uint32_t encoded_path_id = ((hop_one_switch & 0xFFFF) << 16) | (hop_two_switch & 0xFFFF);
            _path_index_map[encoded_path_id] = _paths.size();
            _paths.push_back(encoded_path_id);
            _weights.push_back(latency);
        }
    }

    if (_paths.empty()) {
        std::cerr << "ERROR: no source paths loaded for " << src_switch
                  << " -> " << dst_switch << std::endl;
        exit(-1);
    }

    double max_latency = *std::max_element(_weights.begin(), _weights.end());
    for (size_t i = 0; i < _weights.size(); i++) {
        _weights[i] = max_latency / _weights[i];
        if (_weights[i] != 1.0)
            _weights[i] *= _config.weight_scaling;
    }

    if (_config.weight_scaling == 0) {
        for (size_t i = 0; i < _weights.size(); i++)
            _weights[i] = 1.0;
    }

    if ((_strategy == Strategy::FLOW_V1 || _strategy == Strategy::FLOW_V2) &&
        _config.small_flows_bias && _flow_size < _config.small_flows_threshold) {
        _weights[0] = _config.small_flows_weight;
        for (size_t i = 1; i < _weights.size(); i++)
            _weights[i] = 1.0;
    }

    _dist = std::uniform_int_distribution<size_t>(0, _paths.size() - 1);
    update_weighted_dist();
}

void UecMpSource::update_weighted_dist() {
    auto it = _bad_paths.find(_flow_pair);
    if (it != _bad_paths.end() && it->second.size() < _weights.size()) {
        for (uint32_t bad_path : it->second) {
            auto index = _path_index_map.find(bad_path);
            if (index != _path_index_map.end())
                _weights[index->second] = 0.0;
        }
    }

    _dist_weighted = std::discrete_distribution<size_t>(_weights.begin(), _weights.end());
}

uint32_t UecMpSource::stable_hash(uint32_t a, uint32_t b) const {
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    h ^= h >> 16;
    return h;
}

void UecMpSource::processEv(uint32_t path_id, PathFeedback feedback) {
    if (_strategy == Strategy::RANDOM || _strategy == Strategy::ECMP || _strategy == Strategy::OPS)
        return;

    if (feedback != PathFeedback::PATH_TIMEOUT)
        _last_ack_timestamp[_flow_pair] = EventList::getTheEventList().now();

    if (_strategy == Strategy::FLICR) {
        if (_first_rtt) {
            if (_base_rtt == 0 ||
                EventList::getTheEventList().now() > _last_timestamp + _base_rtt) {
                _first_rtt = false;
                _packet_count = 0;
                _ecn_count = 0;
                _last_timestamp = EventList::getTheEventList().now();
            }
            return;
        }

        _packet_count++;
        if (feedback == PathFeedback::PATH_ECN)
            _ecn_count++;

        auto& good_paths_dq = _good_paths[_flow_pair];
        if (good_paths_dq.empty())
            return;

        if (_base_rtt > 0 &&
            EventList::getTheEventList().now() > _last_timestamp + _base_rtt) {
            if (_packet_count != 0 && (double)_ecn_count / _packet_count > 0.1)
                good_paths_dq.pop_front();

            _packet_count = 0;
            _ecn_count = 0;
            _last_timestamp = EventList::getTheEventList().now();
            return;
        }

        if ((feedback == PathFeedback::PATH_TIMEOUT || feedback == PathFeedback::PATH_NACK) &&
            !good_paths_dq.empty()) {
            good_paths_dq.pop_front();
        }
    } else if (_strategy == Strategy::FLOW_V1) {
        auto& good_paths_dq = _good_paths[_flow_pair];
        if (feedback == PathFeedback::PATH_GOOD) {
            if (good_paths_dq.size() > 8)
                return;
            if (std::find(good_paths_dq.begin(), good_paths_dq.end(), path_id) != good_paths_dq.end())
                return;

            if (_config.sort_buffer_insert) {
                auto new_path_index = _path_index_map[path_id];
                auto insert_pos = std::find_if(
                    good_paths_dq.begin(),
                    good_paths_dq.end(),
                    [&](uint32_t other_path_id) {
                        return _path_index_map[other_path_id] > new_path_index;
                    });
                good_paths_dq.insert(insert_pos, path_id);
            } else {
                good_paths_dq.push_back(path_id);
            }
        } else if (feedback == PathFeedback::PATH_TIMEOUT ||
                   feedback == PathFeedback::PATH_NACK) {
            auto it = std::find(good_paths_dq.begin(), good_paths_dq.end(), path_id);
            if (it == good_paths_dq.end())
                return;

            _ecn_counts.erase(_flow_pair);
            good_paths_dq.erase(it);
            if (feedback == PathFeedback::PATH_TIMEOUT) {
                _bad_paths[_flow_pair].insert(path_id);
                update_weighted_dist();
            }
        } else if (feedback == PathFeedback::PATH_ECN) {
            auto it = std::find(good_paths_dq.begin(), good_paths_dq.end(), path_id);
            if (it == good_paths_dq.end())
                return;

            _ecn_counts[_flow_pair]++;
            if (_ecn_counts[_flow_pair] > _config.ecn_threshold) {
                _ecn_counts[_flow_pair] = 0;
                good_paths_dq.erase(it);
            }
        }
    } else if (_strategy == Strategy::FLOW_V2) {
        if (feedback == PathFeedback::PATH_GOOD) {
            auto& good_paths_dq = _good_paths[_flow_pair];
            if (good_paths_dq.size() > 8)
                return;
            good_paths_dq.push_back(path_id);
        } else if (feedback == PathFeedback::PATH_TIMEOUT) {
            _bad_paths[_flow_pair].insert(path_id);
            update_weighted_dist();
        }
    }
}

uint32_t UecMpSource::nextEntropy(uint64_t seq_sent, uint64_t cur_cwnd_in_pkts) {
    if (_paths.empty())
        return 0;

    if (_strategy == Strategy::RANDOM)
        return _paths[_dist(_rng)];

    if (_strategy == Strategy::ECMP)
        return _paths[stable_hash(_flow_pair.first, _flow_pair.second) % _paths.size()];

    if (_strategy == Strategy::OPS)
        return _paths[_dist_weighted(_rng)];

    if (_strategy == Strategy::FLICR) {
        auto& good_paths_dq = _good_paths[_flow_pair];
        if (good_paths_dq.empty()) {
            _first_rtt = true;
            _last_timestamp = EventList::getTheEventList().now();
            good_paths_dq.push_front(_paths[_dist_weighted(_rng)]);
        }
        return good_paths_dq.front();
    }

    if (_strategy == Strategy::FLOW_V1) {
        if (_config.explore_threshold > 0 && _packet_count > _config.explore_threshold) {
            _packet_count = 0;
            return _paths[_dist_weighted(_rng)];
        }

        auto& good_paths_dq = _good_paths[_flow_pair];
        if (good_paths_dq.empty())
            return _paths[_dist_weighted(_rng)];

        _packet_count++;
        return good_paths_dq.front();
    }

    if (_strategy == Strategy::FLOW_V2) {
        if (_config.explore_threshold > 0 && _packet_count > _config.explore_threshold) {
            _packet_count = 0;
            return _paths[_dist_weighted(_rng)];
        }

        _packet_count++;
        auto& good_paths_dq = _good_paths[_flow_pair];
        if (good_paths_dq.empty())
            return _paths[_dist_weighted(_rng)];

        uint32_t good_path = good_paths_dq.front();
        good_paths_dq.pop_front();
        return good_path;
    }

    return _paths[_dist(_rng)];
}
