# Fig_03 Feedback and Adaptivity

## 实验目标

这一类实验验证 ProbeEP 的在线 NIC feedback：相同 Gate 序列下，固定 migration budget
是否会过于保守或激进；feedback 是否能根据实际 overlap window 改变预算；Attention 与
MoE 两条 controller state 是否保持独立。

## 相对 Base Case 的改动

| 参数 | Base Case | 本实验 |
|---|---:|---:|
| MoE layers | 2 | 6，默认 `LAYERS=6` |
| Raw layer map | `0,1` | `0,1,2,3,4,5` |
| Controller mode | feedback | fixed 与 feedback 对照 |
| Initial/fixed budget | 16 MiB | 0、16、64 MiB |
| 其余模型、Gate、硬件、拓扑 | Base | 不变 |

增加 layer 只为得到更长的 observation/update 序列。每层仍包含两个 microbatch，MB0 更新
Attention chain，MB1 更新 MoE chain；Combine 不更新 controller。

## 硬件与拓扑

| 类别 | 参数 | 值 |
|---|---|---:|
| Compute | Profile | `H20_DSV3_EP32_compute_4096tpr.json` |
| Network | NIC / local fabric | 400 / 7200 Gbps |
| EP | ranks / servers / GPUs per server | 32 / 4 / 8 |
| Fabric | planes / rails / spines | 1 / 8 / 4 |
| Fabric | links/spine | 1 |
| Packet | MTU / queue | 4150 bytes / 128 packets |
| Routing | strategy / CC | `ecmp_host` / `nscc` |

## 模型、Gate 与 ProbeEP 默认参数

| 参数 | 值 |
|---|---:|
| Layers / microbatches | 6 / 2 |
| Tokens/rank/microbatch | 4096 |
| Hidden / FFN hidden | 7168 / 2048 |
| Experts / top-k | 256 / 8 |
| Expert placement | 8/rank，contiguous |
| Token padding / chunk | 128 / 4096 |
| Gate provider | `raw_receive_cdf` |
| Gate seed | 17 |
| Gate raw layers | 0--5 |
| Target overlap ratio | 0.90 |
| Weight chunk | 4 MiB |
| Expert weight scale | 1.0 |

## Case Matrix

| Case ID | Controller | Budget 语义 | Initial/fixed budget | 预期用途 |
|---|---|---|---:|---|
| `fixed_0` | fixed | 所有层使用零 migration budget | 0 MiB | No-Remote lower bound |
| `fixed_16` | fixed | 所有层复用同一预算 | 16 MiB | 固定保守/中等点 |
| `fixed_64` | fixed | 所有层复用同一预算 | 64 MiB | 固定激进点 |
| `feedback` | feedback | 后续层消费上一同类窗口更新 | 16 MiB | 完整 ProbeEP |

四个 case 都使用动态 DAG runner，不能用 static ProbeEP 替代 fixed-budget case。controller
即使在 fixed mode 下仍记录 observation 和建议更新，但这些更新不会应用到下一层。

## HTSim 与 CPU 配置

| 参数 | 值 |
|---|---:|
| HTSim cases | 4 |
| 单 case HTSim PID | 1 |
| 单 case CPU | 1 core |
| 本实验最大并发占用 | 4 workers |
| 全局 worker pool cap | 100 |
| Simulation end | 1000000 us |
| Timeout/case | 1200 s |
| Link sample interval | 100 us |

## 每个 Observation 采集的字段

| 类别 | 字段 |
|---|---|
| 时序 | observation、layer、microbatch、simulation time、compute kind |
| 窗口 | compute reference、Weight+Dispatch duration、global `N/C` |
| 控制 | action、adjustment factor、budget before/after、bottleneck ranks |
| 迁移 | planned/admitted/deferred intents、remote replicas、moved routes |
| 字节 | remote Weight RDMA、local Weight、migration TX/RX |
| 审计 | controller mode、controller applied、runtime validation |

## 输出

| 子结果 | 指标 |
|---|---|
| `fig_03a_ratio` | Attention/MoE `N/C` 时序与 0.90 target |
| `fig_03b_budget` | mean budget before/after |
| `fig_03c_admission` | planned/admitted/deferred intents |
| `fig_03d_fixed_vs_feedback` | makespan 与 migration RDMA Pareto |

结构化数据保存在 `data/results.csv`、`data/observations.csv` 和
`data/metadata.json`。

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MAX_HTSIM_PROCESSES=100 MODE=full bash bash/run.sh

# 改变 feedback 序列长度时显式记录该值
LAYERS=8 MODE=full bash bash/run.sh
```
