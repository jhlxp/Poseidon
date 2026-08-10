# Fig_01 End-to-End Performance

## 实验目标

这一类实验回答两个问题：ProbeEP 是否优于 NCCL、DeepEP、EPLB 和 MoonEP；该结论是否
同时存在于 H20/H100 两种 compute/network balance。所有算法使用相同 Gate assignments，
主指标是完整两层 forward DAG 的 makespan 和有效 token throughput。

## 数据来源与执行规模

| 项目 | 配置 |
|---|---|
| 默认数据源 | `test_logs/run_20260809_234100_h20_h100_2layer_5algo_full` |
| 默认行为 | `REUSE_BASE=1`，只采集既有结果，不启动 HTSim |
| 完整重跑 | `REUSE_BASE=0` |
| 完整重跑 case 数 | 2 hardware × 5 algorithms = 10 HTSim cases |
| 单 case CPU | 1 core，`--workers 1` |
| worker pool | 全局最多 100；本实验最多同时占用 10 个 workers |
| 证据类型 | packet-level simulation |

## 硬件配置

| 参数 | H20 case | H100 case |
|---|---|---|
| Compute profile | `H20_DSV3_EP32_compute_4096tpr.json` | `H100_DSV3_EP32_compute_4096tpr.json` |
| Scale-out NIC | 400 Gbps/rank | 400 Gbps/rank |
| Server-local fabric | 7200 Gbps/rank aggregate | 7200 Gbps/rank aggregate |
| Dispatch dtype | FP8 | FP8 |
| Combine dtype | BF16 | BF16 |
| Expert weight dtype | BF16 | BF16 |

H20/H100 case 只切换 compute profile；Gate、拓扑、链路速率、DAG 和算法参数保持一致。

## 拓扑配置

| 参数 | 值 |
|---|---:|
| EP ranks / GPUs | 32 |
| Servers | 4 |
| GPUs/server | 8 |
| Planes | 1 |
| Rails / L0 Leafs | 8 |
| L1 Spines/plane | 4 |
| L0-L1 links/spine | 1 |
| Host-L0 / L0-L1 rate | 400 Gbps |
| Local injection | 7200 Gbps |
| MTU | 4150 bytes |
| Queue | 128 packets = 531200 bytes |
| ECN low | 16600 bytes |
| ECN high | 53950 bytes，仅解析/校验 |
| Routing | `ecmp_host` |
| Sender CC | `sender_cc_only + nscc` |

rank 到 server/rail 的映射为 `server=rank//8`、`rail=rank%8`。同服务器流量走 local
FullMesh；跨服务器同 rail 走对应 L0；跨 rail 经过一个 L1 Spine。

## 模型与负载配置

| 参数 | Full 配置 |
|---|---:|
| Model | DeepSeek-V3 representative MoE forward |
| MoE layers | 2 |
| Microbatches | 2 |
| Tokens/rank/microbatch | 4096 |
| Global tokens/microbatch | 131072 |
| Unique input tokens | 262144 |
| Hidden | 7168 |
| Expert FFN hidden | 2048 |
| Logical experts | 256 |
| Master experts/rank | 8，contiguous placement |
| Top-k | 8 |
| Routes/microbatch | 1048576 |
| Total expert route executions | 4194304 |
| Attention heads | 128 |
| Sequence length | 4096 |
| Token padding quantum | 128 |
| Token/route chunk | 4096 |

## Gate 配置

| 参数 | 值 |
|---|---|
| Provider | `raw_receive_cdf` |
| Raw layers | `0,1` |
| Seed | 17 |
| Fidelity | `quota_matched_global_receive_histogram` |
| Raw source | 32 个 `decode_{rank}.csv` + placement JSON |
| 公平性 | 五算法按 `(layer,microbatch)` 复用相同 assignment digest |

该输入是 decode-derived empirical receive distribution，不是完整训练逐 token trace。

## 算法与 Case Matrix

| Case | Hardware | Algorithm | 关键配置 |
|---|---|---|---|
| `H20_nccl` | H20 | NCCL | direct route-multiplicity transport |
| `H20_deepep` | H20 | DeepEP | hierarchical token transport |
| `H20_eplb` | H20 | EPLB | persistent physical placement |
| `H20_moonep` | H20 | MoonEP | server-local replication，`replicas_per_rank=256` |
| `H20_probeep` | H20 | ProbeEP | dynamic feedback，initial budget 16 MiB/rank |
| `H100_nccl` | H100 | NCCL | 与 H20 case 相同，仅 compute profile 改变 |
| `H100_deepep` | H100 | DeepEP | 与 H20 case 相同，仅 compute profile 改变 |
| `H100_eplb` | H100 | EPLB | 与 H20 case 相同，仅 compute profile 改变 |
| `H100_moonep` | H100 | MoonEP | `replicas_per_rank=256` |
| `H100_probeep` | H100 | ProbeEP | dynamic feedback，initial budget 16 MiB/rank |

ProbeEP 其余默认参数：target overlap ratio 0.90、Weight chunk 4 MiB、expert weight scale
1.0、padding 128。每个 ProbeEP case 在一个 HTSim PID 内完成 observation/update/append。

## HTSim 配置

| 参数 | Full 值 |
|---|---:|
| Simulation end | 1000000 us |
| Timeout/case | 1200 s |
| Link-load sample | 100 us |
| Local latency | 50 ns |
| Hop latency | 0.1（HTSim CLI units） |
| Switch latency | 0.02（HTSim CLI units） |
| HTSim PID/case | 1 |
| CPU affinity | 从全局 worker pool 获取 CPU 0--99 中的一个 core |

## 采集指标与输出

| 子结果 | 数据 | 作用 |
|---|---|---|
| `fig_01a_makespan` | absolute makespan (ms) | 展示绝对量级 |
| `fig_01b_normalized` | normalized to best non-ProbeEP baseline | 比较五算法 |
| `fig_01c_speedup` | best baseline / ProbeEP | 展示创新算法收益 |
| `fig_01d_throughput` | unique input token throughput | 避免只报告 latency |
| `fig_01_legend` | standalone legend | 保证主图尺寸稳定 |

结构化结果写入 `data/results.csv` 和 `data/metadata.json`。源运行、命令日志和 HTML 分别
记录在 `artifact/source_runs.csv`、`artifact/logs.zip` 和 `artifact/html.zip`。

## 运行命令

```bash
# 只查看将要执行的 10 个 case
REUSE_BASE=0 PLAN_ONLY=1 MODE=full bash bash/run.sh

# 复用当前 Base Case，仅重新采集和绘图
MODE=full bash bash/run.sh

# 重新运行全部 H20/H100 × 五算法
REUSE_BASE=0 MAX_HTSIM_PROCESSES=100 MODE=full bash bash/run.sh
```

`quick` 使用 2 tokens/rank/microbatch、chunk 32 和理论 smoke compute model，只检查执行链，
不能进入论文结果。
