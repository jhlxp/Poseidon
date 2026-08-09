// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_dag_manager.h"

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

int32_t parse_rank(const std::string& value, uint32_t node_count) {
    try {
        size_t consumed = 0;
        const long long rank = std::stoll(value, &consumed);
        if (consumed != value.size() || rank < 0
                || rank >= static_cast<long long>(node_count)) {
            throw std::invalid_argument("rank outside the configured node range");
        }
        return static_cast<int32_t>(rank);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid rank: " + value);
    }
}

std::string trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = value.find_last_not_of(" \t\r");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split_groups(const std::string& line) {
    std::vector<std::string> groups;
    std::istringstream input(line);
    std::string group;
    while (std::getline(input, group, '|')) {
        groups.push_back(trim(group));
    }
    return groups;
}

std::vector<std::string> split_words(const std::string& value) {
    std::vector<std::string> words;
    std::istringstream input(value);
    std::string word;
    while (input >> word) {
        words.push_back(word);
    }
    return words;
}

}  // namespace

OxcDagManager::OxcDagManager(EventList& eventlist, uint32_t node_count)
    : eventlist_(eventlist), node_count_(node_count) {}

OxcDagManager::ComputeDoneEvent::ComputeDoneEvent(
        EventList& eventlist, OxcDagManager& manager, uint32_t task_id)
    : EventSource(eventlist, "DagComputeDone"), manager_(manager), task_id_(task_id) {}

void OxcDagManager::ComputeDoneEvent::doNextEvent() {
    manager_.notify_compute_task_done(task_id_);
    delete this;
}

void OxcDagManager::load_from_file(const std::string& path) {
    if (loaded_) {
        throw std::logic_error("DAG has already been loaded");
    }

    std::ifstream input(path);
    if (!input.good()) {
        throw std::invalid_argument("cannot open DAG file: " + path);
    }

    std::string line;
    uint32_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }

        const size_t comment = line.find('#', first);
        const std::string task_line = line.substr(
                first, comment == std::string::npos ? std::string::npos : comment - first);
        OxcDagTask task = parse_task_line(
                task_line, path + ":" + std::to_string(line_number));
        if (tasks_.count(task.id)) {
            throw std::invalid_argument("DAG task IDs must be unique positive values");
        }
        tasks_.emplace(task.id, TaskRecord{std::move(task)});
    }

    if (tasks_.empty()) {
        throw std::invalid_argument("DAG file contains no tasks: " + path);
    }
    build_barriers();
    validate_acyclic();
    loaded_ = true;
    std::cout << "DAG_LOADED tasks=" << tasks_.size()
              << " barriers=" << barriers_.size() << std::endl;
}

OxcDagTask OxcDagManager::parse_task_line(
        const std::string& task_line, const std::string& context) const {
    const std::vector<std::string> groups = split_groups(task_line);
    if (groups.size() != 4 && groups.size() != 5) {
        throw std::invalid_argument(
                "malformed DAG task at " + context
                + ": expected four groups plus an optional route group");
    }

    OxcDagTask task;
    std::string src_rank;
    std::string dst_rank;
    std::string extra;
    std::istringstream identity(groups[0]);
    std::istringstream endpoints(groups[1]);
    std::istringstream operation(groups[2]);
    if (!(identity >> task.id >> task.barrier_id) || (identity >> extra)
            || !(endpoints >> src_rank >> dst_rank) || (endpoints >> extra)
            || !(operation >> task.transfer_bytes >> task.compute_us)
            || (operation >> extra)) {
        throw std::invalid_argument(
                "malformed DAG task at " + context);
    }
    if (task.id == 0) {
        throw std::invalid_argument("DAG task IDs must be positive values");
    }
    task.src_rank = parse_rank(src_rank, node_count_);
    task.dst_rank = parse_rank(dst_rank, node_count_);
    if (task.compute_us < 0.0) {
        throw std::invalid_argument("DAG compute_us must be non-negative");
    }
    if (task.has_network() == task.has_compute()) {
        throw std::invalid_argument(
                "DAG task must contain exactly one of transfer_bytes or compute_us");
    }
    if (task.has_network() && task.src_rank == task.dst_rank) {
        throw std::invalid_argument(
                "DAG network task requires src_rank != dst_rank");
    }
    if (task.has_compute() && task.src_rank != task.dst_rank) {
        throw std::invalid_argument(
                "DAG compute task requires src_rank == dst_rank");
    }

    std::set<int> dependencies;
    bool saw_predecessor = false;
    bool saw_no_predecessor = false;
    std::istringstream predecessor_fields(groups[3]);
    std::string dependency;
    while (predecessor_fields >> dependency) {
        saw_predecessor = true;
        if (dependency == "-") {
            if (saw_no_predecessor || !dependencies.empty()) {
                throw std::invalid_argument(
                        "DAG '-' predecessor cannot be combined with barrier IDs");
            }
            saw_no_predecessor = true;
            continue;
        }
        if (saw_no_predecessor) {
            throw std::invalid_argument(
                    "DAG '-' predecessor cannot be combined with barrier IDs");
        }
        try {
            size_t consumed = 0;
            const int predecessor = std::stoi(dependency, &consumed);
            if (consumed != dependency.size()) {
                throw std::invalid_argument("invalid barrier ID");
            }
            if (predecessor == task.barrier_id) {
                throw std::invalid_argument("DAG task cannot depend on its own barrier");
            }
            dependencies.insert(predecessor);
        } catch (const std::exception&) {
            throw std::invalid_argument(
                    "invalid DAG predecessor at " + context);
        }
    }
    if (!saw_predecessor) {
        throw std::invalid_argument(
                "DAG task requires '-' or at least one predecessor barrier");
    }
    task.predecessor_barriers.assign(dependencies.begin(), dependencies.end());

    if (groups.size() == 5) {
        if (!task.has_network()) {
            throw std::invalid_argument(
                    "DAG compute task cannot carry a route at " + context);
        }
        task.route = parse_mprail_route_spec(
                split_words(groups[4]),
                "DAG route at " + context);
    }
    return task;
}

void OxcDagManager::enable_dynamic() {
    if (loaded_ || started_) {
        throw std::logic_error("dynamic DAG mode must be enabled before loading tasks");
    }
    dynamic_ = true;
    closed_ = false;
}

void OxcDagManager::append_batch(
        const std::string& batch_id,
        const std::vector<std::string>& task_lines,
        const std::vector<OxcDagObservation>& observations) {
    if (!dynamic_) {
        throw std::logic_error("DAG append requires dynamic mode");
    }
    if (closed_ || finished_) {
        throw std::logic_error("cannot append to a closed DAG");
    }
    if (batch_id.empty()) {
        throw std::invalid_argument("DAG append batch ID must not be empty");
    }
    if (appended_batch_ids_.count(batch_id)) {
        throw std::invalid_argument(
                "DAG append batch IDs must be globally unique: " + batch_id);
    }
    if (task_lines.empty()) {
        throw std::invalid_argument("DAG append batch must contain at least one task");
    }

    std::map<uint32_t, OxcDagTask> new_tasks;
    std::map<int, std::vector<uint32_t>> barrier_tasks;
    std::map<int, std::vector<int>> barrier_predecessors;
    for (size_t index = 0; index < task_lines.size(); ++index) {
        OxcDagTask task = parse_task_line(
                task_lines[index], "dynamic batch " + batch_id + " task "
                        + std::to_string(index));
        if (tasks_.count(task.id) || new_tasks.count(task.id)) {
            throw std::invalid_argument(
                    "DAG task IDs must be globally unique: " + std::to_string(task.id));
        }
        if (barriers_.count(task.barrier_id)) {
            throw std::invalid_argument(
                    "cannot append tasks to an existing DAG barrier: "
                    + std::to_string(task.barrier_id));
        }
        auto predecessors = barrier_predecessors.find(task.barrier_id);
        if (predecessors == barrier_predecessors.end()) {
            barrier_predecessors.emplace(
                    task.barrier_id, task.predecessor_barriers);
        } else if (predecessors->second != task.predecessor_barriers) {
            throw std::invalid_argument(
                    "all tasks in an appended DAG barrier must declare the same predecessors");
        }
        if (task.has_network() && network_validator_) {
            network_validator_(task);
        }
        barrier_tasks[task.barrier_id].push_back(task.id);
        new_tasks.emplace(task.id, std::move(task));
    }

    for (const auto& entry : barrier_predecessors) {
        for (int predecessor : entry.second) {
            if (!barriers_.count(predecessor)
                    && !barrier_predecessors.count(predecessor)) {
                throw std::invalid_argument(
                        "appended DAG task depends on a missing barrier: "
                        + std::to_string(predecessor));
            }
        }
    }

    std::map<int, int> local_indegree;
    std::map<int, std::vector<int>> local_successors;
    for (const auto& entry : barrier_predecessors) {
        local_indegree[entry.first] = 0;
    }
    for (const auto& entry : barrier_predecessors) {
        for (int predecessor : entry.second) {
            if (barrier_predecessors.count(predecessor)) {
                ++local_indegree[entry.first];
                local_successors[predecessor].push_back(entry.first);
            }
        }
    }
    std::vector<int> ready;
    for (const auto& entry : local_indegree) {
        if (entry.second == 0) {
            ready.push_back(entry.first);
        }
    }
    size_t visited = 0;
    while (!ready.empty()) {
        const int barrier_id = ready.back();
        ready.pop_back();
        ++visited;
        for (int successor : local_successors[barrier_id]) {
            if (--local_indegree[successor] == 0) {
                ready.push_back(successor);
            }
        }
    }
    if (visited != barrier_predecessors.size()) {
        throw std::invalid_argument("appended DAG batch contains a cycle");
    }

    std::set<uint32_t> new_observation_ids;
    for (const OxcDagObservation& observation : observations) {
        if (observations_.count(observation.id)
                || !new_observation_ids.insert(observation.id).second) {
            throw std::invalid_argument(
                    "DAG observation IDs must be globally unique: "
                    + std::to_string(observation.id));
        }
        if (observation.predecessor_barriers.empty()) {
            throw std::invalid_argument("DAG observation requires predecessor barriers");
        }
        std::set<int> unique;
        bool has_unfinished = false;
        for (int predecessor : observation.predecessor_barriers) {
            if (!unique.insert(predecessor).second) {
                throw std::invalid_argument("DAG observation predecessors must be unique");
            }
            const auto existing = barriers_.find(predecessor);
            if (existing == barriers_.end()
                    && !barrier_predecessors.count(predecessor)) {
                throw std::invalid_argument(
                        "DAG observation references a missing barrier: "
                        + std::to_string(predecessor));
            }
            if (existing == barriers_.end() || !existing->second.finished) {
                has_unfinished = true;
            }
        }
        if (!has_unfinished) {
            throw std::invalid_argument(
                    "DAG observation must include at least one unfinished barrier");
        }
    }

    for (auto& entry : new_tasks) {
        tasks_.emplace(entry.first, TaskRecord{std::move(entry.second)});
    }
    for (const auto& entry : barrier_tasks) {
        Barrier barrier;
        barrier.id = entry.first;
        barrier.task_ids = entry.second;
        barrier.remaining_tasks = static_cast<int>(entry.second.size());
        barriers_.emplace(entry.first, std::move(barrier));
    }
    for (const auto& entry : barrier_tasks) {
        Barrier& barrier = barriers_.at(entry.first);
        for (int predecessor : barrier_predecessors.at(entry.first)) {
            Barrier& previous = barriers_.at(predecessor);
            if (!previous.finished) {
                ++barrier.indegree;
                previous.successors.push_back(entry.first);
            }
        }
    }
    for (const OxcDagObservation& observation : observations) {
        ObservationRecord record;
        record.observation = observation;
        for (int predecessor : observation.predecessor_barriers) {
            if (!barriers_.at(predecessor).finished) {
                record.pending_barriers.insert(predecessor);
                barrier_observers_[predecessor].push_back(observation.id);
            }
        }
        observations_.emplace(observation.id, std::move(record));
    }

    loaded_ = true;
    appended_batch_ids_.insert(batch_id);
    ++appended_batches_;
    std::cout << "DAG_APPEND_ACK batch=" << batch_id
              << " batch_tasks=" << task_lines.size()
              << " batch_barriers=" << barrier_tasks.size()
              << " batch_observations=" << observations.size()
              << " total_tasks=" << tasks_.size()
              << " total_barriers=" << barriers_.size()
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    if (started_ && !in_observation_handler_) {
        start_ready_barriers();
    }
}

void OxcDagManager::close_dynamic() {
    if (!dynamic_) {
        throw std::logic_error("DAG close requires dynamic mode");
    }
    if (closed_) {
        throw std::logic_error("dynamic DAG has already been closed");
    }
    closed_ = true;
    std::cout << "DAG_CONTROL_CLOSED batches=" << appended_batches_
              << " tasks=" << tasks_.size()
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    maybe_finish();
}

void OxcDagManager::build_barriers() {
    std::map<int, std::vector<int>> barrier_predecessors;
    for (const auto& entry : tasks_) {
        const OxcDagTask& task = entry.second.task;
        Barrier& barrier = barriers_[task.barrier_id];
        barrier.id = task.barrier_id;
        barrier.task_ids.push_back(task.id);

        const auto existing = barrier_predecessors.find(task.barrier_id);
        if (existing == barrier_predecessors.end()) {
            barrier_predecessors.emplace(task.barrier_id, task.predecessor_barriers);
        } else if (existing->second != task.predecessor_barriers) {
            throw std::invalid_argument(
                    "all tasks in a DAG barrier must declare the same predecessor barriers");
        }
    }

    for (auto& entry : barriers_) {
        Barrier& barrier = entry.second;
        barrier.remaining_tasks = static_cast<int>(barrier.task_ids.size());
        const std::vector<int>& predecessors = barrier_predecessors.at(barrier.id);
        barrier.indegree = static_cast<int>(predecessors.size());
        for (int predecessor : predecessors) {
            const auto predecessor_barrier = barriers_.find(predecessor);
            if (predecessor_barrier == barriers_.end()) {
                throw std::invalid_argument(
                        "DAG task depends on a barrier that has no tasks: "
                        + std::to_string(predecessor));
            }
            predecessor_barrier->second.successors.push_back(barrier.id);
        }
    }
}

void OxcDagManager::validate_acyclic() const {
    std::map<int, int> indegree;
    for (const auto& entry : barriers_) {
        indegree.emplace(entry.first, entry.second.indegree);
    }

    std::vector<int> ready;
    for (const auto& entry : indegree) {
        if (entry.second == 0) {
            ready.push_back(entry.first);
        }
    }

    uint32_t visited = 0;
    while (!ready.empty()) {
        const int barrier_id = ready.back();
        ready.pop_back();
        ++visited;
        for (int successor : barriers_.at(barrier_id).successors) {
            int& successor_indegree = indegree.at(successor);
            --successor_indegree;
            if (successor_indegree == 0) {
                ready.push_back(successor);
            }
        }
    }
    if (visited != barriers_.size()) {
        throw std::invalid_argument("DAG contains a cycle");
    }
}

void OxcDagManager::set_network_launcher(NetworkLauncher launcher) {
    network_launcher_ = std::move(launcher);
}

void OxcDagManager::set_network_validator(NetworkValidator validator) {
    network_validator_ = std::move(validator);
}

void OxcDagManager::set_observation_handler(ObservationHandler handler) {
    observation_handler_ = std::move(handler);
}

void OxcDagManager::validate_network_tasks(
        const NetworkValidator& validator) const {
    if (!loaded_) {
        throw std::logic_error("cannot validate an unloaded DAG");
    }
    for (const auto& entry : tasks_) {
        if (entry.second.task.has_network()) {
            validator(entry.second.task);
        }
    }
}

uint32_t OxcDagManager::network_task_count() const {
    uint32_t count = 0;
    for (const auto& entry : tasks_) {
        if (entry.second.task.has_network()) {
            ++count;
        }
    }
    return count;
}

void OxcDagManager::start() {
    if (!loaded_) {
        throw std::logic_error("cannot start an unloaded DAG");
    }
    if (started_) {
        throw std::logic_error("DAG has already started");
    }
    if (network_task_count() > 0 && !network_launcher_) {
        throw std::logic_error("DAG network launcher is not configured");
    }

    started_ = true;
    start_time_ = eventlist_.now();
    start_ready_barriers();
    maybe_finish();
}

void OxcDagManager::start_ready_barriers() {
    std::vector<int> ready;
    for (const auto& entry : barriers_) {
        const Barrier& barrier = entry.second;
        if (barrier.indegree == 0 && !barrier.started) {
            ready.push_back(barrier.id);
        }
    }
    for (int barrier_id : ready) {
        start_barrier(barrier_id);
    }
}

void OxcDagManager::start_barrier(int barrier_id) {
    Barrier& barrier = barriers_.at(barrier_id);
    if (barrier.started || barrier.indegree != 0) {
        return;
    }
    barrier.started = true;
    std::cout << "DAG_BARRIER_START barrier=" << barrier.id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    for (uint32_t task_id : barrier.task_ids) {
        launch_task(task_id);
    }
}

void OxcDagManager::launch_task(uint32_t task_id) {
    TaskRecord& record = tasks_.at(task_id);
    if (record.state != TaskState::PENDING) {
        throw std::logic_error("DAG task launched more than once");
    }
    record.state = TaskState::RUNNING;
    const OxcDagTask& task = record.task;
    std::cout << "DAG_TASK_START task=" << task.id
              << " barrier=" << task.barrier_id
              << " src_rank=" << task.src_rank
              << " dst_rank=" << task.dst_rank
              << " transfer_bytes=" << task.transfer_bytes
              << " compute_us=" << task.compute_us
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;

    if (task.has_network()) {
        network_launcher_(task);
    } else {
        auto* event = new ComputeDoneEvent(eventlist_, *this, task.id);
        eventlist_.sourceIsPending(*event, eventlist_.now() + timeFromUs(task.compute_us));
    }
}

void OxcDagManager::notify_network_task_done(uint32_t task_id) {
    const auto task_it = tasks_.find(task_id);
    if (task_it == tasks_.end() || !task_it->second.task.has_network()) {
        throw std::invalid_argument("network completion received for an unknown DAG task");
    }
    TaskRecord& record = task_it->second;
    if (record.state != TaskState::RUNNING) {
        throw std::logic_error("DAG network task completed before start or more than once");
    }
    std::cout << "DAG_NETWORK_DONE task=" << task_id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    finish_task(task_id);
}

void OxcDagManager::notify_compute_task_done(uint32_t task_id) {
    const auto task_it = tasks_.find(task_id);
    if (task_it == tasks_.end() || !task_it->second.task.has_compute()) {
        throw std::invalid_argument("compute completion received for an unknown DAG task");
    }
    TaskRecord& record = task_it->second;
    if (record.state != TaskState::RUNNING) {
        throw std::logic_error("DAG compute task completed before start or more than once");
    }
    std::cout << "DAG_COMPUTE_DONE task=" << task_id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    finish_task(task_id);
}

void OxcDagManager::finish_task(uint32_t task_id) {
    TaskRecord& record = tasks_.at(task_id);
    if (record.state != TaskState::RUNNING) {
        throw std::logic_error("DAG task completed before start or more than once");
    }
    record.state = TaskState::DONE;
    ++completed_tasks_;
    const OxcDagTask& task = record.task;
    std::cout << "DAG_TASK_DONE task=" << task.id
              << " barrier=" << task.barrier_id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;

    Barrier& barrier = barriers_.at(task.barrier_id);
    --barrier.remaining_tasks;
    if (barrier.remaining_tasks < 0) {
        throw std::logic_error("DAG barrier task accounting underflow");
    }
    if (barrier.remaining_tasks != 0) {
        return;
    }

    barrier.finished = true;
    std::cout << "DAG_BARRIER_DONE barrier=" << barrier.id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    for (int successor : barrier.successors) {
        Barrier& next = barriers_.at(successor);
        --next.indegree;
        if (next.indegree < 0) {
            throw std::logic_error("DAG barrier indegree underflow");
        }
    }
    notify_observations(barrier.id);
    start_ready_barriers();
    maybe_finish();
}

void OxcDagManager::notify_observations(int barrier_id) {
    const auto watchers = barrier_observers_.find(barrier_id);
    if (watchers == barrier_observers_.end()) {
        return;
    }
    const std::vector<uint32_t> observation_ids = watchers->second;
    barrier_observers_.erase(watchers);
    for (uint32_t observation_id : observation_ids) {
        ObservationRecord& record = observations_.at(observation_id);
        record.pending_barriers.erase(barrier_id);
        if (record.emitted || !record.pending_barriers.empty()) {
            continue;
        }
        record.emitted = true;
        if (!observation_handler_) {
            throw std::logic_error("DAG observation handler is not configured");
        }
        in_observation_handler_ = true;
        try {
            observation_handler_(
                    observation_id, timeAsUs(eventlist_.now()));
        } catch (...) {
            in_observation_handler_ = false;
            throw;
        }
        in_observation_handler_ = false;
    }
}

void OxcDagManager::maybe_finish() {
    if (finished_ || !started_ || !closed_
            || completed_tasks_ != tasks_.size()) {
        return;
    }
    finished_ = true;
    finish_time_ = eventlist_.now();
    std::cout << "DAG_SUMMARY tasks=" << tasks_.size()
              << " barriers=" << barriers_.size();
    if (dynamic_) {
        std::cout << " observations=" << observations_.size()
                  << " batches=" << appended_batches_;
    }
    std::cout << " makespan_us=" << makespan_us() << std::endl;
}
