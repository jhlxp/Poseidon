// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef OXC_DAG_MANAGER_H
#define OXC_DAG_MANAGER_H

#include "eventlist.h"
#include "mprail_route_spec.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

struct OxcDagTask {
    uint32_t id = 0;
    int barrier_id = 0;
    int32_t src_rank = -1;
    int32_t dst_rank = -1;
    uint64_t transfer_bytes = 0;
    double compute_us = 0.0;
    std::vector<int> predecessor_barriers;
    std::optional<MpRailRouteSpec> route;

    bool has_network() const { return transfer_bytes > 0; }
    bool has_compute() const { return compute_us > 0.0; }
};

// A barrier DAG above the selected data plane. Each task owns exactly one
// network or compute operation. Tasks assigned to a ready barrier run in
// parallel, and the barrier is released after its last task finishes.
class OxcDagManager {
public:
    using NetworkLauncher = std::function<void(const OxcDagTask&)>;
    using NetworkValidator = std::function<void(const OxcDagTask&)>;

    OxcDagManager(EventList& eventlist, uint32_t node_count);

    // One non-comment task per line:
    // task_id barrier_id | src_rank dst_rank | transfer_bytes compute_us |
    // predecessor_barrier ... | -
    void load_from_file(const std::string& path);
    void validate_network_tasks(const NetworkValidator& validator) const;
    void set_network_launcher(NetworkLauncher launcher);
    void start();
    void notify_network_task_done(uint32_t task_id);
    void notify_compute_task_done(uint32_t task_id);

    uint32_t network_task_count() const;
    bool finished() const { return finished_; }
    double makespan_us() const { return timeAsUs(finish_time_ - start_time_); }

private:
    enum class TaskState {
        PENDING,
        RUNNING,
        DONE,
    };

    struct TaskRecord {
        OxcDagTask task;
        TaskState state = TaskState::PENDING;
    };

    struct Barrier {
        int id = 0;
        int indegree = 0;
        int remaining_tasks = 0;
        bool started = false;
        bool finished = false;
        std::vector<uint32_t> task_ids;
        std::vector<int> successors;
    };

    class ComputeDoneEvent : public EventSource {
    public:
        ComputeDoneEvent(EventList& eventlist, OxcDagManager& manager, uint32_t task_id);
        void doNextEvent() override;
        bool isTraffic() override { return false; }

    private:
        OxcDagManager& manager_;
        uint32_t task_id_;
    };

    EventList& eventlist_;
    uint32_t node_count_;
    NetworkLauncher network_launcher_;
    std::map<uint32_t, TaskRecord> tasks_;
    std::map<int, Barrier> barriers_;
    bool loaded_ = false;
    bool started_ = false;
    bool finished_ = false;
    uint32_t completed_tasks_ = 0;
    simtime_picosec start_time_ = 0;
    simtime_picosec finish_time_ = 0;

    void build_barriers();
    void validate_acyclic() const;
    void start_ready_barriers();
    void start_barrier(int barrier_id);
    void launch_task(uint32_t task_id);
    void finish_task(uint32_t task_id);
};

#endif
