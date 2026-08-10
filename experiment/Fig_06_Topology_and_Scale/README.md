# Fig_06 Topology and Scale Sensitivity

## 实验目标

这一类实验固定 EP32 workload，改变 server boundary、plane、Spine 和 L0-L1 bundle，观察
MoonEP 的 server-local ceiling、ProbeEP speedup、endpoint utilization、queue pressure 和
path diversity。

当前动态 runner、raw Gate 和 H20/H100 profile 都是 EP32 口径，因此本脚本只改变 EP32
内部的物理组织，不把不兼容的 EP16/EP64 数据混入论文。

## 固定硬件、模型与 Gate

| 类别 | 参数 | 值 |
|---|---|---:|
| Compute | Profile | `H20_DSV3_EP32_compute_4096tpr.json` |
| Network | NIC / local rate | 400 / 7200 Gbps |
| Model | layers / microbatches | 2 / 2 |
| Model | tokens/rank/microbatch | 4096 |
| Model | hidden / FFN hidden | 7168 / 2048 |
| MoE | experts / top-k | 256 / 8 |
| MoE | padding / chunk | 128 / 4096 |
| Gate | provider / layers / seed | raw receive / `0,1` / 17 |
| Algorithms | comparison | MoonEP、dynamic ProbeEP |
| MoonEP | replicas/rank | 256 |
| ProbeEP | controller | feedback，16 MiB initial，target 0.90 |

所有 case 保持 32 ranks；`gpus_per_server` 改变时 server 数和 rail 数随之改变，expert master
仍按 256 experts 在 32 ranks 上连续放置。

## Base Topology

| 参数 | Base 值 |
|---|---:|
| Ranks | 32 |
| Servers / GPUs per server | 4 / 8 |
| Planes / rails | 1 / 8 |
| Spines/plane | 4 |
| L0-L1 links/spine | 1 |
| MTU / queue | 4150 bytes / 128 packets |
| Routing / CC | `ecmp_host` / `nscc` |

## Sweep A：Server Boundary

| Case prefix | GPUs/server | Servers | Rails | Planes | Spines | Links/spine |
|---|---:|---:|---:|---:|---:|---:|
| `boundary_4` | 4 | 8 | 4 | 1 | 4 | 1 |
| `boundary_8` | 8 | 4 | 8 | 1 | 4 | 1 |
| `boundary_16` | 16 | 2 | 16 | 1 | 4 | 1 |

该 sweep 改变 MoonEP 可以利用的 server-local domain 大小，也改变 ProbeEP global
server-first planning 中的 server 数量。

## Sweep B：Spine Path Diversity

| Case prefix | GPUs/server | Planes | Spines/plane | Links/spine | Path multiplicity |
|---|---:|---:|---:|---:|---:|
| `spines_1` | 8 | 1 | 1 | 1 | 1 |
| `spines_2` | 8 | 1 | 2 | 1 | 2 |
| `spines_4` | 8 | 1 | 4 | 1 | 4 |
| `spines_8` | 8 | 1 | 8 | 1 | 8 |

## Sweep C：L0-L1 Bundle

| Case prefix | GPUs/server | Planes | Spines/plane | Links/spine | Path multiplicity |
|---|---:|---:|---:|---:|---:|
| `bundles_1` | 8 | 1 | 4 | 1 | 4 |
| `bundles_2` | 8 | 1 | 4 | 2 | 8 |
| `bundles_4` | 8 | 1 | 4 | 4 | 16 |

Spine 与 bundle sweep 的横轴统一使用 `spines × links_per_spine`。主机 NIC 仍为
400 Gbps，因此更多 path 主要改变 fabric contention，不提高单 rank line rate。

## Sweep D：Plane Capacity

| Case prefix | GPUs/server | Planes | Spines/plane | Links/spine |
|---|---:|---:|---:|---:|
| `planes_1` | 8 | 1 | 4 | 1 |
| `planes_2` | 8 | 2 | 4 | 1 |

当前 ProbeEP Weight route metadata 固定 `plane=0`。因此 `planes_2` 用来暴露当前实现没有
利用的额外 plane capacity，不能包装成 multi-plane-aware scheduling 结果。

## Case 与资源数量

| 项目 | 数量 |
|---|---:|
| Topology points | 3 + 4 + 3 + 2 = 12 |
| Algorithms/point | 2 |
| HTSim cases | 24 |
| 单 case CPU | 1 core |
| 本实验最大并发占用 | 24 workers |
| 全局 worker pool cap | 100 workers |
| Simulation end / timeout | 1000000 us / 1200 s |
| Link sample interval | 100 us |

`boundary_8`、`spines_4`、`bundles_1` 和 `planes_1` 都对应 Base topology，但保留独立
case ID，便于每组 sensitivity 拥有自己的 anchor 和 artifact。

## 采集与输出

| 子结果 | 主要字段 | 回答的问题 |
|---|---|---|
| `fig_06a_server_boundary` | servers、makespan | local ceiling 如何随 server domain 变化 |
| `fig_06b_endpoint_load` | endpoint peak utilization | ProbeEP 是否制造 endpoint hotspot |
| `fig_06c_path_diversity` | path multiplicity、speedup | fabric path 是否限制收益 |
| `fig_06d_queue_pressure` | max queue bytes | extra Weight 是否形成拥塞 |
| `fig_06e_plane_capacity` | configured planes、speedup | 当前 plane-0 pinning 的边界 |

`collect.py` 从每个 MoonEP/ProbeEP source run 读取 endpoint 与 link-load CSV，生成
`data/network_metrics.csv`；端到端结果保存在 `data/results.csv`。

## EP Scale Guard

```bash
bash bash/run_ep_scale.sh
```

该入口会明确失败。启用 EP16/EP64 前必须同时泛化 dynamic runner、compute profile 和 Gate
assignment，不能只修改 HTSim node 数。

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MAX_HTSIM_PROCESSES=100 MODE=full bash bash/run.sh
```
