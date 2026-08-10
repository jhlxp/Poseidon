# Fig_02 Mechanism Analysis

## 实验目标

这一类实验沿同一个 Base Case 解释 ProbeEP 为什么快：Gate 如何形成 rank/server skew，
MoonEP 为什么受 server-local ceiling 限制，ProbeEP 的 global plan 与 NIC admission 分别
消除了多少 padded compute，以及跨服务器 Weight movement 是否形成不可忽略的网络债务。

## 执行方式

| 项目 | 配置 |
|---|---|
| 数据源 | `test_logs/run_20260809_234100_h20_h100_2layer_5algo_full` |
| 新 HTSim 仿真 | 0 |
| CPU worker pool | 不占用；仅做结构化 artifact analysis |
| 输入 ZIP | H20/H100 各一个五算法 visualization ZIP |
| 证据类型 | `trace_analysis_of_packet_simulation` |

这一实验不会重新生成 Gate 或 DAG。它必须复用 Fig_01 的 assignment digest、makespan、
timeline 和 link-load 数据，保证性能结果与机制解释来自同一批 case。

## 被分析的 Base Case

| 类别 | 参数 | 值 |
|---|---|---|
| Hardware | Compute profiles | H20、H100 schema-v2 |
| Hardware | NIC / local fabric | 400 / 7200 Gbps |
| Topology | EP / servers / GPUs per server | 32 / 4 / 8 |
| Topology | planes / rails / spines | 1 / 8 / 4 |
| Topology | MTU / queue | 4150 bytes / 128 packets |
| Workload | layers / microbatches | 2 / 2 |
| Workload | tokens/rank/microbatch | 4096 |
| Workload | hidden / FFN hidden | 7168 / 2048 |
| Workload | experts / top-k | 256 / 8 |
| Workload | padding / chunk | 128 / 4096 |
| Gate | provider / seed / raw layers | `raw_receive_cdf` / 17 / `0,1` |
| Algorithms | compared cases | NCCL、DeepEP、EPLB、MoonEP、ProbeEP |

## 输入 Artifact

| ZIP member | 读取内容 |
|---|---|
| `gate_load_profile_summary.json` | Gate digest、before/after rank/server/instance load |
| `dag_timeline_summary.json` | makespan、compute/network active time、overlap、payload bytes |
| `mprail_endpoint_load_summary.csv` | endpoint mean/peak utilization |
| `mprail_link_load_summary.csv` | link throughput、queue 和 active samples |
| `dag_gpu_timeline.html` | 关键 invocation 的 Weight→Dispatch 时序审计 |

## 分析维度

| 分析 | 对比对象 | 主要指标 |
|---|---|---|
| Gate skew | 相同 assignment 的 rank 与 server 聚合 | max/mean imbalance |
| Local ceiling | NCCL/DeepEP/EPLB/MoonEP/ProbeEP after profile | rank 与 server max/mean |
| Planner opportunity | ProbeEP baseline→planned | server padded routes |
| Admission effect | ProbeEP planned→admitted | admitted/deferred intents、padded routes |
| Migration efficiency | 每个 invocation | moved routes / remote replica |
| Network cost | ProbeEP Weight legs | scale-out RDMA bytes、server-local bytes |
| Overlap | per-rank interval union | compute active、network active、intersection |

## 数据口径

| 字段 | 正确语义 |
|---|---|
| `expert_weight_rdma_bytes` | 跨服务器 Weight wire bytes，只统计 RDMA leg |
| `expert_weight_local_bytes` | gather/scatter/prefetch local legs |
| `logical_transfer_bytes` | 所有逻辑传输 leg 的总 work，不等于 wire bytes |
| `overlap_us` | 每 rank compute/network interval intersection 后求和 |
| `task FCT sum` | 工作量统计，不能作为 critical path breakdown |
| `max(TX,RX)` | 单 endpoint full-duplex footprint，不能跨 rank 求和为 wire bytes |

## 输出

| 子结果 | 数据文件 | 回答的问题 |
|---|---|---|
| `fig_02a_gate_skew` | `load_balance.csv` | Gate 是否同时形成 rank/server skew |
| `fig_02b_local_ceiling` | `load_balance.csv` | 本地复制是否留下 server lower bound |
| `fig_02c_padding` | `planning.csv` | plan 与 admission 分别降低多少 padding |
| `fig_02d_network_cost` | `network_cost.csv` | 额外 Weight traffic 在哪里发生 |
| `fig_02e_migration_efficiency` | `planning.csv` | 一个完整副本承载多少 moved routes |

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MODE=full bash bash/run.sh
```

该类型只分析既有 full 结果；`MODE=quick` 只会改变输出 metadata 的 eligibility，不会把
full Base Case 降格为 smoke workload。
