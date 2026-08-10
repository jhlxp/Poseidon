# Fig_07 Design Ablation

## 实验目标

这一类实验在完全相同的动态 DAG、Gate 和 H20/EP32 Base Case 上，逐项改变 migration
admission、feedback target 和 Weight chunk granularity，区分 ProbeEP 各模块的贡献。

所有 case 使用 `run_probeep_2layer_ratio_full.py`，保持 observation/update/append 路径
一致；不能把 static ProbeEP 或 MoonEP 结果作为某个动态消融的替代品。

## 固定硬件、拓扑与 Workload

| 类别 | 参数 | 值 |
|---|---|---:|
| Compute | Profile | `H20_DSV3_EP32_compute_4096tpr.json` |
| Network | NIC / local fabric | 400 / 7200 Gbps |
| EP | ranks / servers / GPUs per server | 32 / 4 / 8 |
| Fabric | planes / rails / spines | 1 / 8 / 4 |
| Packet | MTU / queue | 4150 bytes / 128 packets |
| Model | layers / microbatches | 2 / 2 |
| Model | tokens/rank/microbatch | 4096 |
| Model | hidden / FFN hidden | 7168 / 2048 |
| MoE | experts / top-k | 256 / 8 |
| MoE | padding / route chunk | 128 / 4096 |
| Gate | provider / layers / seed | raw receive / `0,1` / 17 |
| Weight | expert scale | 1.0 |

## Case Matrix

| Case ID | Variant | Controller | Budget | Weight chunk | Target `N/C` |
|---|---|---|---:|---:|---:|
| `no_remote` | No-Remote | fixed | 0 MiB | 4 MiB | 0.90 |
| `fixed_8` | Fixed-Conservative | fixed | 8 MiB | 4 MiB | 0.90 |
| `fixed_64` | Fixed-Aggressive | fixed | 64 MiB | 4 MiB | 0.90 |
| `fine_1` | Fine Weight chunks | feedback | 16 MiB initial | 1 MiB | 0.90 |
| `full` | Full ProbeEP | feedback | 16 MiB initial | 4 MiB | 0.90 |
| `coarse_128` | Monolithic Weight | feedback | 16 MiB initial | 128 MiB | 0.90 |
| `target_05` | Conservative target | feedback | 16 MiB initial | 4 MiB | 0.50 |

完整 BF16 expert state 约 84 MiB；128 MiB chunk 因此把一个 expert 的 Weight 作为单块
调度，用于观察 coarse-grained head-of-line blocking。1 MiB 与 4 MiB case 分别观察更细
pipeline 和默认设计。

## 各 Variant 改变了什么

| Variant | 保留 | 移除/改变 |
|---|---|---|
| No-Remote | global opportunity planning、dynamic runner | admission budget 设为 0 |
| Fixed-Conservative | planning、Weight pipeline | feedback 不应用，固定 8 MiB |
| Fixed-Aggressive | planning、endpoint hard cap、Weight pipeline | feedback 不应用，固定 64 MiB |
| Fine Weight chunks | planning、feedback、rail scheduling | chunk 从 4 MiB 改为 1 MiB |
| Full ProbeEP | 全部当前实现 | 无 |
| Monolithic Weight | planning、feedback | chunk 改为 128 MiB |
| Conservative target | planning、feedback、4 MiB chunk | target 从 0.90 改为 0.50 |

fixed mode 仍测量 feedback observation，但后续层只消费固定 budget。这样 fixed/full case 的
执行路径相同，区别只在 controller update 是否应用。

## 当前不可执行的消融

| Variant | 缺失的 config switch | 当前处理 |
|---|---|---|
| No-Per-Rail-Release | 全局 Weight barrier 与 per-source-rail release 切换 | 显式 guard |
| No-Pair-Aware-Placement | rail selection policy 切换 | 显式 guard |
| True Compute-Only | 关闭 endpoint hard cap 的 admission switch | 尚未实现，不用 `fixed_64` 冒充 |

```bash
bash bash/run_pipeline_ablation.sh
```

该命令会失败并解释缺失项。实现正式开关前，不使用手工修改 DAG 或后处理数字代替。

## 仿真与资源配置

| 项目 | 值 |
|---|---:|
| HTSim cases | 7 |
| 单 case HTSim PID | 1 |
| 单 case CPU | 1 core |
| 本实验最大并发占用 | 7 workers |
| 全局 worker pool cap | 100 workers |
| Simulation end / timeout | 1000000 us / 1200 s |
| Link sample interval | 100 us |

## 采集指标与输出

| 子结果 | 指标 | 目的 |
|---|---|---|
| `fig_07a_makespan` | normalized makespan to Full | 每项设计对端到端路径的影响 |
| `fig_07b_network_cost` | migration RDMA GiB | 性能变化对应的网络代价 |
| `fig_07c_admission` | admitted/deferred intents | budget/controller 的直接行为 |
| `fig_07d_chunk_granularity` | 1/4/128 MiB vs makespan | Weight pipeline 粒度影响 |

`data/observations.csv` 还保留每个 invocation 的 `N/C`、budget、moved routes、replicas 和
TX/RX bytes，避免只根据最终 makespan 猜测模块作用。

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MAX_HTSIM_PROCESSES=100 MODE=full bash bash/run.sh
```
