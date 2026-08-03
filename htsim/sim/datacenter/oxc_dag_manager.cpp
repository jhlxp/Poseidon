// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "oxc_dag_manager.h"

#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

int32_t parse_optional_rank(const std::string& value, uint32_t node_count) {
    if (value == "-" || value == "-1") {
        return -1;
    }

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

        std::istringstream fields(line.substr(first));
        OxcDagTask task;
        std::string src_rank;
        std::string dst_rank;
        std::string compute_rank;
        if (!(fields >> task.id >> task.stage_id >> src_rank >> dst_rank >> compute_rank
              >> task.bytes >> task.compute_us)) {
            throw std::invalid_argument(
                    "malformed DAG task at " + path + ":" + std::to_string(line_number));
        }
        if (task.id == 0 || tasks_.count(task.id)) {
            throw std::invalid_argument("DAG task IDs must be unique positive values");
        }
        task.src_rank = parse_optional_rank(src_rank, node_count_);
        task.dst_rank = parse_optional_rank(dst_rank, node_count_);
        task.compute_rank = parse_optional_rank(compute_rank, node_count_);
        if (task.compute_us < 0.0) {
            throw std::invalid_argument("DAG compute_us must be non-negative");
        }
        if (task.has_network() == task.has_compute()) {
            throw std::invalid_argument(
                    "DAG task must contain exactly one of network bytes or compute time");
        }
        if (task.has_network() && (task.src_rank < 0 || task.dst_rank < 0)) {
            throw std::invalid_argument("DAG network task requires src_rank and dst_rank");
        }
        if (!task.has_network() && (task.src_rank >= 0 || task.dst_rank >= 0)) {
            throw std::invalid_argument("DAG compute-only task must use '-' for src_rank and dst_rank");
        }
        if (task.has_compute() && task.compute_rank < 0) {
            throw std::invalid_argument("DAG compute task requires compute_rank");
        }
        if (!task.has_compute() && task.compute_rank >= 0) {
            throw std::invalid_argument("DAG network-only task must use '-' for compute_rank");
        }

        std::set<int> dependencies;
        std::string dependency;
        while (fields >> dependency) {
            if (dependency == "-" || dependency[0] == '#') {
                break;
            }
            try {
                size_t consumed = 0;
                const int predecessor = std::stoi(dependency, &consumed);
                if (consumed != dependency.size()) {
                    throw std::invalid_argument("invalid stage ID");
                }
                if (predecessor == task.stage_id) {
                    throw std::invalid_argument("DAG task cannot depend on its own stage");
                }
                dependencies.insert(predecessor);
            } catch (const std::exception&) {
                throw std::invalid_argument(
                        "invalid DAG predecessor at " + path + ":"
                        + std::to_string(line_number));
            }
        }
        task.predecessor_stages.assign(dependencies.begin(), dependencies.end());
        tasks_.emplace(task.id, TaskRecord{std::move(task)});
    }

    if (tasks_.empty()) {
        throw std::invalid_argument("DAG file contains no tasks: " + path);
    }
    build_stages();
    validate_acyclic();
    loaded_ = true;
    std::cout << "DAG_LOADED tasks=" << tasks_.size()
              << " stages=" << stages_.size() << std::endl;
}

void OxcDagManager::build_stages() {
    std::map<int, std::vector<int>> stage_predecessors;
    for (const auto& entry : tasks_) {
        const OxcDagTask& task = entry.second.task;
        Stage& stage = stages_[task.stage_id];
        stage.id = task.stage_id;
        stage.task_ids.push_back(task.id);

        const auto existing = stage_predecessors.find(task.stage_id);
        if (existing == stage_predecessors.end()) {
            stage_predecessors.emplace(task.stage_id, task.predecessor_stages);
        } else if (existing->second != task.predecessor_stages) {
            throw std::invalid_argument(
                    "all tasks in a DAG stage must declare the same predecessor stages");
        }
    }

    for (auto& entry : stages_) {
        Stage& stage = entry.second;
        stage.remaining_tasks = static_cast<int>(stage.task_ids.size());
        const std::vector<int>& predecessors = stage_predecessors.at(stage.id);
        stage.indegree = static_cast<int>(predecessors.size());
        for (int predecessor : predecessors) {
            const auto predecessor_stage = stages_.find(predecessor);
            if (predecessor_stage == stages_.end()) {
                throw std::invalid_argument(
                        "DAG task depends on a stage that has no tasks: "
                        + std::to_string(predecessor));
            }
            predecessor_stage->second.successors.push_back(stage.id);
        }
    }
}

void OxcDagManager::validate_acyclic() const {
    std::map<int, int> indegree;
    for (const auto& entry : stages_) {
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
        const int stage_id = ready.back();
        ready.pop_back();
        ++visited;
        for (int successor : stages_.at(stage_id).successors) {
            int& successor_indegree = indegree.at(successor);
            --successor_indegree;
            if (successor_indegree == 0) {
                ready.push_back(successor);
            }
        }
    }
    if (visited != stages_.size()) {
        throw std::invalid_argument("DAG contains a cycle");
    }
}

void OxcDagManager::set_network_launcher(NetworkLauncher launcher) {
    network_launcher_ = std::move(launcher);
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
    start_ready_stages();
}

void OxcDagManager::start_ready_stages() {
    std::vector<int> ready;
    for (const auto& entry : stages_) {
        const Stage& stage = entry.second;
        if (stage.indegree == 0 && !stage.started) {
            ready.push_back(stage.id);
        }
    }
    for (int stage_id : ready) {
        start_stage(stage_id);
    }
}

void OxcDagManager::start_stage(int stage_id) {
    Stage& stage = stages_.at(stage_id);
    if (stage.started || stage.indegree != 0) {
        return;
    }
    stage.started = true;
    std::cout << "DAG_STAGE_START stage=" << stage.id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    for (uint32_t task_id : stage.task_ids) {
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
              << " stage=" << task.stage_id
              << " src_rank=" << task.src_rank
              << " dst_rank=" << task.dst_rank
              << " compute_rank=" << task.compute_rank
              << " bytes=" << task.bytes
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
              << " stage=" << task.stage_id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;

    Stage& stage = stages_.at(task.stage_id);
    --stage.remaining_tasks;
    if (stage.remaining_tasks < 0) {
        throw std::logic_error("DAG stage task accounting underflow");
    }
    if (stage.remaining_tasks != 0) {
        return;
    }

    stage.finished = true;
    std::cout << "DAG_STAGE_DONE stage=" << stage.id
              << " time_us=" << timeAsUs(eventlist_.now()) << std::endl;
    for (int successor : stage.successors) {
        Stage& next = stages_.at(successor);
        --next.indegree;
        if (next.indegree < 0) {
            throw std::logic_error("DAG stage indegree underflow");
        }
    }
    start_ready_stages();

    if (completed_tasks_ == tasks_.size()) {
        finished_ = true;
        finish_time_ = eventlist_.now();
        std::cout << "DAG_SUMMARY tasks=" << tasks_.size()
                  << " stages=" << stages_.size()
                  << " makespan_us=" << makespan_us() << std::endl;
    }
}
