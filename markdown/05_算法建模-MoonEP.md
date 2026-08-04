# 算法建模：MoonEP

## 1. 算法边界

MoonEP 通过在线规划动态冗余 expert，使每个 EP rank 接收完全相同数量的 real token routes。

本文以 MoonshotAI/MoonEP 官方仓库 commit `86f3574` 为代码阅读基线。当前公开实现和测试面向单个 NVLink domain，README 的测试入口使用 8 GPU NVLink。首版模型因此限制：

```text
MoonEP EP group 必须位于同一服务器
```

不能把 MoonEP 的节点内权重访问直接扩展成未经验证的跨服务器 RDMA 算法。

当前 `pysrc/moe_dag/algorithms/moonep.py` 是第一版确定性 workload planner，
不是 MoonEP 官方 planner 的 Python 复刻。它按 hot expert 优先、rank 剩余容量优先
选择 replica；如果 `replicas_per_rank` 无法让每个 rank 达到相同 real route 数，
生成阶段直接失败。manifest 固定记录
`planner=deterministic_capacity_balancer_v1`，避免把结果误认为官方 `CommPlan`。

## 2. 符号

```text
S   每个 source rank 的 input tokens
K   每 token 的 routed top-k
E   EP group 的逻辑 experts
R   EP ranks
B   每 rank 的 weight prefetch slots
H   hidden size
H'  expert FFN intermediate size
NvS 每 rank 的物理 dispatch slots，包含 S*K real routes 和分组 padding
```

核心不变量：

```text
每个 rank 的 real routed token count = S * K
```

`NvS` 可以大于 `S*K`，因为每个 VM group 的 token segment 按 `token_padding` 对齐。expert GEMM 的理论 FLOP 数应使用实际 `cu_seqlens`/padded segment，而不是只看 real token 数。

## 3. Forward phase

### 3.1 Online planning

输入：

```text
topk expert IDs
tokens_per_expert
当前 expert home placement
B 和 token_padding
```

planner 输出 `MoonEPCommPlan`，核心字段包括：

```text
dst
cu_seqlens
experts_to_copy[R, B]
zero_fill_ranges
remote_stats
dup_groups / dup_loffs / dup_counts
```

建模为：

- 每 rank 的 planning ComputeTask；
- 必要的 rank sync/control metadata；
- 全 rank planning barrier。

当前不估算 planning kernel 的真实性能；每 rank 生成一个明确标为
`theoretical_placeholder` 的 ComputeTask，并通过共同 planning barrier 保留依赖语义。
后续可根据 `R*S*K`、`E`、`B` 和 imbalance 指标建立 profiling 或资源模型。

### 3.2 Dynamic redundant experts

planner 为 overloaded expert 选择临时 replica rank，并把 expert ID 写入目标 rank 的 `experts_to_copy`。

训练模式官方要求：

```text
B = E / R
```

推理允许更小的 B，官方建议值为 3 到 4；超出 prefetch slot 的 remote expert 可以通过 symmetric mapping 直接读取，但性能路径不同。

算法 plan 必须记录：

```text
logical_expert
home_rank
execution_rank
prefetch_slot | remote_direct
assigned_token_routes
```

### 3.3 Weight prefetch

对每个 `experts_to_copy[dst_rank, slot] = expert`：

```text
src = home_rank(expert)
dst = dst_rank
```

每个 projection 的 payload：

```text
gate_weight_bytes = H * H' * bytes(weight_dtype)
up_weight_bytes   = H * H' * bytes(weight_dtype)
down_weight_bytes = H' * H * bytes(weight_dtype)
```

若 gate/up 在实际实现中 fused，逻辑总 FLOP 数不变，但 task 数可以合并。

权重预取是服务器内部 rank-to-rank 通信，在 MpRail 上走高速 local FullMesh。不能只把它写成 planning compute；否则 MoonEP 的额外通信成本会消失。

该映射只保留 rank、payload 和本地带宽竞争。MoonEP 实际使用 NVLink symmetric mapping、remote read/write 和 TMA，MpRail/UEC local flow 的 ACK/拥塞控制语义并不等价；首版不声称节点内传输和 kernel 的联合延迟准确。

prefetch kernel 本身还有 HBM/TMA copy 开销。首版只保留 rank 间 payload 的 TransferTask，不额外估算这部分本地 compute 时间，以避免和 local flow 重复：

- rank 间 payload 作为 TransferTask；
- kernel setup/local staging 暂不计时；
- 后续引入 end-to-end prefetch profile 时必须先扣除或校准传输部分。

### 3.4 Token dispatch

planner 把每个 `(token, topk_slot)` 分配到 home 或 replica rank，使每 rank real routes 恰为 `S*K`。

MoonEP 的 dispatch 会把 token 直接写入目标 rank 最终 expert-grouped 位置。对同一 rank 上属于 duplicate group 的多份 route，跨 rank dispatch 只发送 primary row，随后 dispatch epilogue 在目标 rank 本地展开 duplicate slots。

因此实际 local network payload 从 `plan.dst` 和 duplicate encoding 计算：

```text
只统计 primary cross-rank rows
local source/destination bypass 不产生 network task
duplicate expansion 计入本地 ComputeTask
```

不能简单使用 `R * S * K * H` 作为跨 rank字节。该值是全局 real route 数，不是实际跨 rank primary copy 数。

### 3.5 Dispatch epilogue

官方实现中的 dispatch epilogue：

- 完全在目标 rank 的本地 NVL shard 上运行；
- 从 primary row 向 duplicate slots fan-out；
- 不产生 cross-rank 通信；
- zero-copy 模式直接把 shard view 交给 expert FFN。

建模为 memory-bound ComputeTask，cost key 至少包含：

```text
H, duplicate_groups, duplicate_rows, num_sms_dedup, zero_copy
```

### 3.6 Expert compute

每 rank 的 real route 数完全相等，但 compute shape 仍受 VM group padding 影响：

```text
real_tokens_per_rank = S * K
compute_slots_per_rank = NvS or sum(padded expert segments)
```

应根据 `cu_seqlens[E+B]` 为每个 VM group 建立 grouped GEMM shape。不能仅把所有 token 合成一个平均 GEMM，否则无法反映 padding、空 expert 和 replica group。

### 3.7 Combine prologue 与 combine

combine 前，MoonEP 在本地 shard 中把 duplicate rows 以 FP32 累加回 primary row。该 prologue：

- 是本地 ComputeTask；
- 不产生 cross-rank 通信；
- cost 取决于 duplicate groups、rows 和 H。

随后 combine 根据 forward plan 把 primary expert output 发回原 source rank，并恢复 token-major `[S,H]` 输出。TransferTask 必须复用同一个 plan，不能重新运行 planner。

zero-copy 为 false 时，还存在用户 tensor 与 NVL shard 的 boundary copy，应作为本地 compute/memory task计入。

## 4. Training backward

MoonEP backward 包含：

```text
combine backward:
  使用 saved plan 重新 dispatch grad_output，不重新 planning，不重新 prefetch

expert backward:
  对 home 和 replica VM groups 计算 activation/weight grad

dispatch backward:
  combine grad_hidden 回原 token

redundant expert grad reduce:
  replica rank 的 fp32 weight grad -> expert home rank
```

### 4.1 Gradient reduce 字节

对每个实际使用的 replica expert 和每个 projection：

```text
grad_bytes = parameter_count * bytes(fp32)
```

目标是 home rank。官方实现让 owner 读取各 rank reduce buffer 并累加到 local parameter grad，随后清理本地 slot。生成器按逻辑数据方向统一表示为：

```text
replica_rank -> home_rank, payload_kind=expert_grad
```

本地 accumulation/clear 是 ComputeTask。

首版只实现 forward；训练配置在实现 grad reduce 前必须报 unsupported。

## 5. 与 DeepEP 的本质差异

| 维度 | DeepEP | MoonEP |
|---|---|---|
| expert placement | 通常固定 home placement | 每步规划临时 redundant experts |
| token balance | 受 router skew 影响 | 每 rank real routes 固定为 `S*K` |
| 额外通信 | hierarchical forwarding/metadata | expert weight prefetch；训练还有 grad reduce |
| dispatch shape | rank token 数可变 | 静态 real count，另有 VM group padding |
| 专属本地 kernel | layout/copy/reduce | planning、duplicate epilogue/prologue、prefetch、grad reduce |
| 初始拓扑范围 | NVLink + RDMA 多节点 | 官方当前实现按单 NVLink domain 建模 |

因此不能把 MoonEP 实现成“DeepEP 加一个 balance flag”。

## 6. 配置草案

```yaml
algorithm: moonep
mode: forward
B: 32                 # training 通常 E/R；inference 可更小
token_padding: 128
zero_copy: true
weight_dtype: bf16
grad_dtype: fp32
metadata_mode: data_plus_metadata
```

配置校验：

- `E % R == 0`；
- 每个 source rank 的输入 token 数 S 相同；
- 所有 EP ranks 位于同一 server；
- `B > 0`；
- training 时 `B == E/R`；
- `S*K`、NvS 和 buffer shape 不溢出；
- `H/H'` 满足所选 kernel variant 的 alignment；
- plan 中每 rank real routes 恰为 `S*K`。

## 7. Imbalance 指标

报告中至少输出：

```text
T_e = expert e 的逻辑 routed token 数
mean_T = total_routes / E
maxvio = max_e(T_e / mean_T) - 1
```

同时对比规划前后：

- 每 home rank 逻辑 token 数；
- 每 execution rank 分配 token 数；
- duplicate expert 数；
- weight prefetch bytes；
- padding slots；
- primary cross-rank token copies；
- duplicate expansion rows。

## 8. 测试

- balanced routing：不应产生不必要的 expert prefetch。
- single hot expert：仍满足每 rank `S*K` real routes。
- `B` 边界：训练容量足够；推理 overflow 标记 remote-direct。
- token 守恒：全局 routes 等于 `R*S*K`。
- plan 重放：combine backward 不重新 planning/prefetch。
- duplicate group：跨 rank只发送 primary，本地展开数量正确。
- weight bytes：每个实际复制 expert 的三组 projection 字节精确。
- combine 使用 forward plan，输出回原 source rank。
- 非同服务器 EP group 被明确拒绝。
- 固定 routing 和 seed 时 plan/DAG 稳定。

## 9. 代码依据

- [MoonEP 官方仓库](https://github.com/MoonshotAI/MoonEP)。
- MoonEP 官方 `README.md`：perfect balance、B、weight/grad buffer、zero-copy 和 API 顺序。
- `moonep/planning.py`：`MoonEPCommPlan`、`experts_to_copy`、padding 和 duplicate plan。
- `moonep/api.py`：planning、dispatch、dispatch epilogue、prefetch、combine prologue、combine、grad reduce 调用顺序。
- `moonep/dispatch_epilogue.py`：primary row 的本地 duplicate expansion。
- `moonep/combine_prologue.py`：duplicate rows 的本地 FP32 reduction。
- `moonep/prefetch.py`：remote expert weight prefetch。
- `moonep/grad_reduce.py`：replica gradient 到 home expert 的归并。
