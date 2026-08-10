# Fig_04 Workload Sensitivity

## 实验目标

这一类实验固定 H20 与 EP32/1Plane，改变 Gate family、倾斜强度、seed 和 token volume，
验证 ProbeEP 的收益是否由实际 server-level skew 和每个副本承载的 useful routes 决定，
而不是只对当前 raw distribution 有效。

每个 workload point 成对运行 MoonEP 与动态 ProbeEP。绘图横轴使用实际 assignment 得到的
rank/server max-to-mean，不用 provider 名称代替负载强度。

## 固定硬件与拓扑

| 类别 | 参数 | 值 |
|---|---|---:|
| Compute | Profile | `H20_DSV3_EP32_compute_4096tpr.json` |
| Network | NIC / local fabric | 400 / 7200 Gbps |
| EP | ranks / servers / GPUs per server | 32 / 4 / 8 |
| Fabric | planes / rails / spines | 1 / 8 / 4 |
| Fabric | L0-L1 links/spine | 1 |
| Packet | MTU / queue | 4150 bytes / 128 packets |
| Routing | strategy / CC | `ecmp_host` / `nscc` |

## 固定模型与算法参数

| 参数 | 值 |
|---|---:|
| Layers / microbatches | 2 / 2 |
| Hidden / FFN hidden | 7168 / 2048 |
| Experts / top-k | 256 / 8 |
| Expert placement | 8/rank，contiguous |
| Token padding | 128 |
| Dispatch / Combine | FP8 / BF16 |
| MoonEP replicas/rank | 256 |
| ProbeEP controller | feedback，initial budget 16 MiB |
| ProbeEP target ratio | 0.90 |
| ProbeEP Weight chunk / scale | 4 MiB / 1.0 |

## Gate Family Matrix

| Case ID | Provider | Provider 参数 | Seed | Tokens/rank | Chunk |
|---|---|---|---:|---:|---:|
| `balanced` | `balanced_permuted` | exact balanced permutation | 17 | 4096 | 4096 |
| `uniform` | `uniform_random` | uniform without replacement | 17 | 4096 | 4096 |
| `ultra_1_25` | `ultra_rank_zipf` | target rank imbalance 1.25 | 17 | 4096 | 4096 |
| `ultra_2` | `ultra_rank_zipf` | target rank imbalance 2 | 17 | 4096 | 4096 |
| `ultra_4` | `ultra_rank_zipf` | target rank imbalance 4 | 17 | 4096 | 4096 |
| `fast_05` | `fast_matrix_zipf` | skew 0.5 | 17 | 4096 | 4096 |
| `fast_08` | `fast_matrix_zipf` | skew 0.8 | 17 | 4096 | 4096 |
| `fast_095` | `fast_matrix_zipf` | skew 0.95 | 17 | 4096 | 4096 |
| `raw` | `raw_receive_cdf` | raw layer map `0,1` | 17 | 4096 | 4096 |

## Token Volume 与 Seed Matrix

| Case ID | 目的 | Provider | Seed | Tokens/rank | Chunk |
|---|---|---|---:|---:|---:|
| `raw_t1024` | 小 token volume | raw receive | 17 | 1024 | 1024 |
| `raw` | Base volume | raw receive | 17 | 4096 | 4096 |
| `raw_t8192` | 大 token volume | raw receive | 17 | 8192 | 8192 |
| `ultra_2` | Seed baseline | Ultra target 2 | 17 | 4096 | 4096 |
| `ultra_seed23` | Seed variation | Ultra target 2 | 23 | 4096 | 4096 |
| `ultra_seed41` | Seed variation | Ultra target 2 | 41 | 4096 | 4096 |

token sweep 只改变 tokens/rank 和 matching chunk；compute profile 文件、硬件和 Gate shape
保持不变。raw workload 保持同一个经验 receive histogram shape，并按新的总 route budget
重新做整数 quota。

## Gate Fidelity

| Provider | Fidelity |
|---|---|
| balanced/uniform/Ultra | `synthetic_assignments` |
| FAST-style | `sampled_from_source_expert_matrix` |
| raw receive | `quota_matched_global_receive_histogram` |

FAST-style 是 source-expert matrix 形状适配，不是复现 FAST 的可变 row-sum traffic；raw
receive 是 decode-derived distribution，不是训练逐 token trace。

## 仿真规模

| 项目 | 数量/配置 |
|---|---:|
| Distinct workload points | 13 |
| Algorithms/point | 2（MoonEP、ProbeEP） |
| HTSim cases | 26 |
| 单 case CPU | 1 core |
| 本实验最大并发占用 | 26 workers |
| 全局 worker pool cap | 100 |
| Simulation end / timeout | 1000000 us / 1200 s |
| Link sample interval | 100 us |

## 采集与输出

| 子结果 | 主要数据 | 解释 |
|---|---|---|
| `fig_04a_skew_speedup` | measured server imbalance、MoonEP/ProbeEP makespan | 收益是否随 server skew 增长 |
| `fig_04b_migration` | server imbalance、Weight RDMA | controller 为不同 skew 支付多少通信 |
| `fig_04c_token_volume` | 1024/4096/8192 token speedup | expert copy 是否被更多 routes 摊销 |
| `fig_04d_seed_variation` | Ultra target 2 的三个 seeds | 结论是否依赖单个 seed |
| `fig_04_legend` | Gate family legend | 与主图分离 |

`collect.py` 从每个实际 Gate profile 提取 mean/max rank/server imbalance 和 routing fidelity；
不使用 generator 的目标参数冒充 realized skew。

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MAX_HTSIM_PROCESSES=100 MODE=full bash bash/run.sh
```

`quick` 不适合该矩阵的 token-volume 结论；论文结果必须使用 `MODE=full`。
