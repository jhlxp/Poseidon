# Fig_05 Hardware Sensitivity

## 实验目标

这一类实验保持 workload、Gate assignment 和 EP32 topology 不变，分别改变 GPU compute
profile、scale-out NIC、server-local fabric 和 expert state bytes，定位“通信换计算”的
硬件适用区间。

每个硬件点成对运行 MoonEP 与动态 ProbeEP。除 compute×NIC 必要交叉外，每组 sweep 只
改变一个硬件因素。

## 固定 Workload

| 参数 | 值 |
|---|---:|
| Model | DSV3 representative MoE forward |
| Layers / microbatches | 2 / 2 |
| Tokens/rank/microbatch | 4096 |
| Hidden / FFN hidden | 7168 / 2048 |
| Experts / top-k | 256 / 8 |
| Expert placement | 8/rank，contiguous |
| Token padding / chunk | 128 / 4096 |
| Gate provider | `raw_receive_cdf` |
| Gate layers / seed | `0,1` / 17 |
| Dispatch / Combine / Weight dtype | FP8 / BF16 / BF16 |

## 固定拓扑

| 参数 | 值 |
|---|---:|
| EP ranks / servers / GPUs per server | 32 / 4 / 8 |
| Planes / rails / spines | 1 / 8 / 4 |
| L0-L1 links/spine | 1 |
| MTU / queue | 4150 bytes / 128 packets |
| Routing / CC | `ecmp_host` / `nscc` |

## 算法配置

| Algorithm | 配置 |
|---|---|
| MoonEP | server-local replication，`replicas_per_rank=256` |
| ProbeEP | dynamic feedback，initial budget 16 MiB，target 0.90 |
| ProbeEP Weight | default chunk 4 MiB，default expert weight scale 1.0 |

## Sweep A：Compute × NIC

| Case prefix | Compute profile | NIC | Local fabric | Weight scale |
|---|---|---:|---:|---:|
| `nic_H20_100` | H20 | 100 Gbps | 7200 Gbps | 1.0 |
| `nic_H20_200` | H20 | 200 Gbps | 7200 Gbps | 1.0 |
| `nic_H20_400` | H20 | 400 Gbps | 7200 Gbps | 1.0 |
| `nic_H20_800` | H20 | 800 Gbps | 7200 Gbps | 1.0 |
| `nic_H100_100` | H100 | 100 Gbps | 7200 Gbps | 1.0 |
| `nic_H100_200` | H100 | 200 Gbps | 7200 Gbps | 1.0 |
| `nic_H100_400` | H100 | 400 Gbps | 7200 Gbps | 1.0 |
| `nic_H100_800` | H100 | 800 Gbps | 7200 Gbps | 1.0 |

H20/H100 分别使用对应 schema-v2 profile。NIC rate 同时进入 HTSim `-linkspeed` 和
ProbeEP controller 的 theoretical line-rate cap。

## Sweep B：Expert State Size

| Case prefix | Compute | NIC | Local fabric | `expert_weight_scale` |
|---|---|---:|---:|---:|
| `weight_0.25` | H20 | 400 Gbps | 7200 Gbps | 0.25 |
| `weight_0.5` | H20 | 400 Gbps | 7200 Gbps | 0.5 |
| `weight_1` | H20 | 400 Gbps | 7200 Gbps | 1.0 |
| `weight_2` | H20 | 400 Gbps | 7200 Gbps | 2.0 |
| `weight_4` | H20 | 400 Gbps | 7200 Gbps | 4.0 |

`expert_weight_scale` 只缩放 ProbeEP 完整专家的 gather/RDMA/scatter/prefetch bytes，用于
代表不同 expert state size 或传输精度；token payload、expert compute 和 MoonEP 不变。

## Sweep C：Server-Local Fabric

| Case prefix | Compute | NIC | Local fabric | Weight scale |
|---|---|---:|---:|---:|
| `local_900` | H20 | 400 Gbps | 900 Gbps | 1.0 |
| `local_1800` | H20 | 400 Gbps | 1800 Gbps | 1.0 |
| `local_3600` | H20 | 400 Gbps | 3600 Gbps | 1.0 |
| `local_7200` | H20 | 400 Gbps | 7200 Gbps | 1.0 |

该 sweep 改变 HTSim `-local_linkspeed`，用于观察 Weight gather/scatter/local-prefetch 与
scale-out RDMA 之间的瓶颈转移。

## Case 与资源数量

| 项目 | 数量 |
|---|---:|
| Hardware points | 8 + 5 + 4 = 17 |
| Algorithms/point | 2 |
| HTSim cases | 34 |
| 单 case CPU | 1 core |
| 本实验最大并发占用 | 34 workers |
| 全局 worker pool cap | 100 workers |
| Simulation end / timeout | 1000000 us / 1200 s |
| Link sample interval | 100 us |

每个 prefix 实际产生 `${prefix}_moonep` 和 `${prefix}_probeep` 两个独立 source runs。

## 输出

| 子结果 | 变量 | 指标 |
|---|---|---|
| `fig_05a_compute_nic` | H20/H100 × NIC | speedup over MoonEP |
| `fig_05b_expert_state` | expert weight scale | speedup、收益边界 |
| `fig_05c_local_fabric` | local fabric rate | speedup |
| `fig_05d_admission` | NIC rate | admitted migration RDMA |

`data/results.csv` 保存 makespan 和所有 case 参数；`data/observations.csv` 保存 ProbeEP
admission/controller 明细；`data/metadata.json` 保存 CPU pool 与 paper eligibility。

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MAX_HTSIM_PROCESSES=100 MODE=full bash bash/run.sh
```
