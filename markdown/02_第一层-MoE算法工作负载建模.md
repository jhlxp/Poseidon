# 第一层：MoE 算法工作负载建模

## 1. 第一层的职责

第一层接收一次已经确定形状和 router 结果的 MoE invocation，将其转换为算法专属的计算通信任务图。

它不负责构造 attention，也不负责决定一共有多少 transformer layer。它回答的是：

```text
给定这些 source tokens、top-k experts、expert placement 和硬件 rank，
DeepEP 或 MoonEP 实际需要哪些计算、复制、转发、归并和同步？
```

## 2. 共享输入模型

建议使用不可变的数据结构表达一次 invocation：

```text
MoEInvocation
  invocation_id
  mode: train_fwd | train_bwd | prefill | decode
  ranks: [Rank]
  rank_to_server: rank -> server
  experts: E
  expert_placement: expert -> home_rank
  tokens_per_source_rank: S_r
  hidden: H
  ffn_hidden: H_ff
  topk: K
  dispatch_dtype
  combine_dtype
  weight_dtype
  routing_assignments:
    (src_rank, token_id, topk_slot, expert_id, route_weight)
```

`routing_assignments` 是比较算法时必须共享的原始事实。算法可以改变 token 最终在哪个 replica rank 上计算，但不能修改 token 选择了哪个逻辑 expert。

## 3. 共享 IR

### 3.1 Task

生成器内部至少需要三类节点：

```text
ComputeTask
  rank
  op_kind
  shape_key
  operation_flops
  duration_us
  overlaps_communication
  metadata

TransferTask
  src_rank
  dst_rank
  bytes
  payload_kind
  route_spec | none
  communication_phase_id
  metadata

BarrierGroup (optional)
  task_ids
  reason
```

HTSim emitter 最终只输出 compute 和 network task。默认每个 task 分配独立 barrier；`BarrierGroup` 只用于显式声明多个 task 必须共同完成后才能释放后继。

每个 task 使用 task-level predecessor 集合。task ID 和 barrier ID 只在最终 lowering 时分配。

### 3.2 PhaseGraph

算法插件返回一个带命名边界的 graph fragment：

```text
entry -> planning/layout -> dispatch -> expert_compute -> combine -> exit
```

命名边界用于第二层把 shared expert、attention 或下一个 microbatch 接进来。算法可以在边界内部加入任意专属 task。

## 4. 为什么需要共享 base

共享 base 负责以下不应重复实现的逻辑：

- rank/server/expert 坐标校验；
- router assignment 的加载和确定性排序；
- tensor 字节数、dtype、alignment 和 padding；
- point-to-point flow 聚合；
- local bypass 判定；
- 理论 FLOP 数和固定 `compute_us` 占位；
- task graph、ID、barrier 映射和输出校验。

但 base 不定义固定的通信 phase。算法插件至少实现：

```text
validate(invocation, topology, config)
plan(invocation, placement, routing) -> AlgorithmPlan
build_forward(plan, cost_provider) -> PhaseGraph
build_backward(plan, cost_provider) -> PhaseGraph
summarize(plan) -> AlgorithmReport
```

DeepEP 的 `AlgorithmPlan` 保存 domain 去重、relay 和 reduction 位置；MoonEP 的 plan 保存动态 replica、`experts_to_copy`、padding 和 duplicate groups。两者不应塞进一个充满可选字段的通用 plan。

## 5. Token 路由与字节核算

### 5.1 不允许直接使用 `S * K * H`

朴素 expanded dispatch 的 payload 是：

```text
num_routes * aligned_token_bytes
num_routes = sum_r(S_r * K)
```

但真实算法可能按 rank 或 node 去重。如果一个 token 的多个 expert 位于同一 destination rank，hidden state 可能只发送一次，expert ID 和 weight 作为 metadata 携带。因此需要从 route assignment 计算：

```text
U_rank(t) = unique destination ranks selected by token t
U_node(t) = unique destination servers selected by token t
```

不同算法分别选择 `K`、`U_rank` 或 `U_node` 作为某一 hop 的复制数。

### 5.2 Payload 分类

每个 TransferTask 必须标记 payload：

| payload_kind | 内容 |
|---|---|
| `token_hidden` | dispatch hidden state |
| `expert_output` | combine 返回值或 partial reduction |
| `route_metadata` | expert ID、slot、weight、count、offset |
| `expert_weight` | 动态冗余 expert 权重预取 |
| `expert_grad` | 冗余 expert 梯度归并 |
| `control_sync` | barrier、tail、counter、notification 的近似流量 |

不能使用一个 `redundancy_factor` 同时放大所有通信。token 冗余、expert 权重冗余、padding 和 metadata 是不同来源，生命周期和路径也不同。

### 5.3 Token 字节

基础公式：

```text
hidden_bytes = H * bytes(dispatch_dtype)
scale_bytes  = quantization_scale_bytes(H, quant_config)
token_bytes  = align(hidden_bytes + scale_bytes + per_token_metadata, alignment)
```

combine 通常使用不同 dtype：

```text
output_bytes = H * bytes(combine_dtype)
```

例如 FP8 dispatch 与 BF16 combine 不能共用同一个 `token_bytes`。

### 5.4 Flow 聚合

IR 不为每个 token 创建一个 HTSim flow。经过算法去重之后，按以下 key 聚合：

```text
(phase, src_rank, dst_rank, payload_kind, route_spec, chunk_id)
```

聚合后 `bytes` 是该 key 下所有逻辑 payload 的总和。报告中同时保存 logical message 数和 aggregated flow 数。

同 rank bypass 不生成 `src_rank == dst_rank` 的 network task，因为当前 DAG 禁止这种输入。本地读写、copy 或 reduce 计入对应 ComputeTask。

## 6. 多级转发与 one-to-many

DeepEP hierarchical dispatch 可能先把一个 token 按 destination node 去重，只通过 fabric 发送一份，然后在目标服务器内向多个 expert rank 扇出。

这不能展开为多条完整 `server_forward`：

```text
错误：每个 destination expert rank 各发一份 fabric payload
正确：src -> destination relay 只发一份，再 relay -> target ranks 本地扇出
```

第一层 IR 应表达：

```text
fabric_transfer(token, src, relay, once_per_destination_node)
local_fanout(token, relay, each_unique_destination_rank)
```

barrier DAG 既能表达 full-message 转发，也能通过显式 chunk task 表达流水转发。当前实现使用 `chunk_tokens` 把逻辑 token 集合拆成多个 task：

```text
fabric_chunk[i] -> local_fanout_chunk[i]
```

每个 chunk 是一条完整 flow，后继监听该 flow 的完成 barrier。它能表达 chunk 间流水，
但不监听单条 flow 内第 N 个 packet 到达，因此不能描述成 packet-progress 模型。

## 7. 冗余专家建模

冗余 expert 可以建模，而且必须拆成四部分：

1. `replica planning`：决定逻辑 expert 的临时执行 rank；
2. `expert weight prefetch`：home rank 到 replica rank 的权重通信；
3. `token reassignment`：dispatch 目的地改为 home 或 replica；
4. `gradient reduction`：训练时 replica grad 回到 home rank。

动态 replica 还会改变：

- 每 rank 的 token 数和 padding；
- expert GEMM 的 shape；
- dispatch/combine 的 src-dst matrix；
- 权重和梯度通信与 token 通信之间的依赖。

因此冗余专家属于算法 planner，而不是 emitter 的额外 flag。

## 8. Compute 固定时长占位

第一版不实现 profiling 或动态 GPU 资源模型。所有 compute task 都在生成 DAG 时得到固定 `duration_us`：

```text
H100_SXM_BF16_DENSE_FLOPS = 989e12
H100_SXM_SMS = 132
COMMUNICATION_SMS = 20

normal_flops  = 989e12
overlap_flops = 989e12 * (132 - 20) / 132

duration_us = operation_flops / selected_flops * 1e6
```

统一接口只需要返回：

```text
operation_flops
duration_us
overlaps_communication
available_sms: 132 | 112
source: h100_sxm_bf16_dense_peak | h100_sxm_fixed_comm_sm_partition
```

FMA 按 2 FLOPs 计数。`overlaps_communication=true` 时，整条 compute task 固定使用 112 SM 对应的约 `839.15 TFLOP/s`；否则使用完整 `989 TFLOP/s`。不根据网络实际结束时刻动态改变计算速度。

SM 预留按 `(rank, communication_phase_id)` 计算。同一个 dispatch/combine phase 的多条 flow、多个 destination 和多个 chunk 共同对应一个通信 kernel，只预留一次 20 SM。`communication_phase_id` 进入 IR/manifest，不增加 `.dag` 字段。

这个结果不包含 HBM、launch、融合、padding、warp、抢占、cache 和实际利用率，是理论下界。后续 profiling 或更完整的资源模型替换该接口时，不改变 DAG 格式。

## 9. 与 HTSim DAG 的 lowering

每个 TransferTask 输出为：

```text
task_id barrier_id | src_rank dst_rank | transfer_bytes 0 | predecessor_barriers [| route]
```

每个 ComputeTask 输出为：

```text
task_id barrier_id | rank rank | 0 compute_us | predecessor_barriers
```

lowering 需要验证：

- 所有 rank 在 `Nodes` 范围内；
- task ID 唯一；
- 每个 barrier 的前驱集合一致；
- 默认一 task 一 barrier；任何多 task barrier 都有显式 join 原因；
- route 与 rank/server mapping 一致；
- 每个算法 phase 都可追溯到 task；
- 总字节等于 AlgorithmPlan 的分类汇总；
- 无意外的同 rank network task；
- compute metadata 记录理论 FLOP 数、是否 overlap、可用 SM 和所用峰值；
- 同一 `(rank, communication_phase_id)` 只预留一次 20 SM；
- 没有把 network latency 重复写入 compute task。

## 10. 首版验收

- 同一个 routing trace 在 DeepEP/MoonEP 下共享逻辑 expert 选择。
- 算法输出的 token、weight、grad 字节分项可手算复核。
- DeepEP 按 destination node 去重，fabric 不重复发送。
- MoonEP 每个 rank 的 planned real token 数为 `S * K`，padding 单独统计。
- 输出 `.dag` 能被当前 HTSim 加载并完成。
- manifest 能恢复每个 task 对应的算法 phase。
- 固定随机种子时输出逐字节稳定。
