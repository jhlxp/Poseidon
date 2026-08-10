# Fig_08 Overheads and Operating Boundary

## 实验目标与证据边界

这一类实验包含两类证据：Python reference planner scaling 和 analytical break-even model。
它不运行 HTSim，不能替代 packet-level end-to-end 结果，也不能作为 C++/CUDA prototype 的
在线 latency claim。

| Evidence | 回答的问题 | 不能证明什么 |
|---|---|---|
| `reference_planner` | 当前算法复杂度趋势、内存和 intent 数量 | 生产 runtime 的绝对延迟 |
| `analytical_model` | compute saving 与 exposed Weight cost 的边界 | 真实网络拥塞下的 makespan |

## Reference Planner 固定配置

| 参数 | 值 |
|---|---:|
| Planner API | `ProbeEPBuilder.plan()` |
| Compute model | `H100CostModel` theoretical |
| Hidden / FFN hidden | 7168 / 2048 |
| Top-k | 8 |
| Dispatch / Combine / Weight | FP8 / BF16 / BF16 |
| Gate provider | `ultra_rank_zipf` |
| Gate target rank imbalance | 2.0 |
| Gate seed | 17 |
| GPUs/server | 8（EP≥8） |
| Default EP / experts / tokens | 32 / 256 / 32 |
| Default Weight chunk | 4 MiB |
| Full repeats/config | 7 |
| Quick repeats/config | 2 |
| Reported latency | median、empirical P95 |
| Memory | 独立 Python `tracemalloc` run 的 peak MiB |

Gate assignment 在计时前生成，因此 `runtime_*` 只覆盖 `ProbeEPBuilder.plan()`。latency
repeats 不开启 `tracemalloc`；peak memory 由同配置的额外一次 plan 单独测量。每个 repeat
创建新的 builder，避免 controller/planner 状态跨 repeat 污染。

## Planner Scaling Matrix

| Sweep | EP | Experts | Tokens/rank | Logical routes | Chunk | Configs |
|---|---:|---:|---:|---:|---:|---:|
| EP | 8、16、32、64 | 256 | 32 | `EP×32×8` | 4 MiB | 4 |
| Experts | 32 | 64、128、256、512 | 32 | 8192 | 4 MiB | 4 |
| Routes | 32 | 256 | 8、16、32、64、128 | 2048--32768 | 4 MiB | 5 |
| Chunks | 32 | 256 | 32 | 8192 | 1、2、4、8、16 MiB | 5 |

总计 18 个 planner configurations。每个点保存 median/P95、peak memory、planned/admitted
intents 和 remote replicas。当前图使用前三组；chunk sweep 数据保留给 scheduler overhead
补充分析。

## Analytical Model

模型使用以下定义：

```text
expert_weight_bytes = 3 * hidden * ffn_hidden * BF16_bytes * weight_scale
compute_flops/route = 6 * hidden * ffn_hidden
transfer_us         = expert_weight_bytes * 8 / effective_rate
exposed_cost_us     = transfer_us * (1 - target_overlap_ratio)
net_saving_us       = moved_routes * compute_us_per_route - exposed_cost_us
```

## Break-Even Matrix

| 维度 | 扫描值 |
|---|---|
| Expert state scale | 0.25、0.5、1、2、4 |
| Moved routes/replica | 128、512、2048、8192、32768、131072 |
| Effective migration rate | 100、200、400、800 Gbps |
| Target overlap ratio | 0.90 |
| Total analytical points | 5 × 6 × 4 = 120 |

`fig_08d_break_even` 固定展示 400 Gbps slice，并标出 `net_saving_us=0` 等值线。其他 rate
完整保留在 `data/break_even.csv`，不能只保留最终图片。

## CPU 与执行配置

| 项目 | 值 |
|---|---|
| HTSim processes | 0 |
| HTSim worker pool | 不占用 |
| Python benchmark process | 1 |
| CPU affinity | CPU 100（control/analysis partition） |
| OpenMP/BLAS threads | 1 |

该 benchmark 应与大规模 HTSim sweep 分开运行，避免 CPU 100 的 sibling contention 影响
尾部数据。

## 输出

| 子结果 | 数据 | 作用 |
|---|---|---|
| `fig_08a_ep_scaling` | EP vs median/P95 plan time | EP scaling |
| `fig_08b_expert_scaling` | experts vs median/P95 | expert-count scaling |
| `fig_08c_route_scaling` | logical routes vs median/P95 | histogram/route scaling |
| `fig_08d_break_even` | moved routes × state scale | operating boundary |

结构化数据：`data/planner_scaling.csv`、`data/break_even.csv`、`data/metadata.json`。命令日志
进入 `artifact/logs.zip`；本实验不生成 HTML。

## 运行命令

```bash
PLAN_ONLY=1 MODE=full bash bash/run.sh
MODE=full bash bash/run.sh
```
