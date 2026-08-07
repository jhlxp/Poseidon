# 算法建模：DeepEP 核心路径

## 1. 项目范围

本项目只建模 DeepEP 面向训练和 prefill 的核心通信行为：

```text
按 destination rank 去除重复 token payload
  +
跨服务器时先进入目标服务器的同 local-index GPU
  +
目标服务器内部转发到真实 destination rank
```

阅读 `/home/xuheng/DeepEP` 源码是为了确认上述数据移动本质，不在 HTSim 中复刻
DeepEP 的 CUDA/RDMA kernel。首版明确不实现：

- low-latency/decode 路径；
- V1/V2、normal/hybrid/direct 等版本开关；
- notify/layout、TMA、RDMA queue pair 和内存一致性细节；
- packet/chunk 在单条 flow 内部的流水消费；
- backward、profiling 和真实 kernel 性能。

训练场景当前先使用与 prefill 相同的 forward dispatch/expert/combine 图。以后增加
backward 时复用同一套去冗余和目标端转发策略，不重新定义传输协议。

NCCL 是对照算法：它关闭 payload 去冗余，也不使用目标端转发，直接把每条 top-k
route 发送到真实 destination rank。

## 2. 核心一：默认去冗余

### 2.1 原始 route 与 payload item

router 的原始输出保留每条 top-k route：

```text
(src_rank, token_id, topk_slot, expert_id, dst_rank)
```

DeepEP 的 hidden payload 去重 key 为：

```text
(src_rank, token_id, dst_rank)
```

同一 token 的多个 expert 位于同一 destination rank 时，只发送一份 hidden；expert
ID、top-k slot 和 weight 仍作为逻辑 metadata 保留，expert compute 的 route 数也
不能减少。

例如：

```text
token 7 -> expert 9  on rank 5
token 7 -> expert 41 on rank 5
```

DeepEP 产生一个发往 rank 5 的 payload item，NCCL 产生两个。

### 2.2 共享 base

去冗余不是 DeepEP builder 的私有代码。第一层 base 提供：

```text
TokenPayloadPolicy
  deduplicate: bool = true
  scope: none | destination_rank = destination_rank
```

- 除 NCCL 外，算法默认 `deduplicate=true`。
- NCCL 显式选择 `deduplicate=false, scope=none`。
- planner 使用策略生成 payload item；emitter 只聚合、分 chunk 和输出 DAG。
- flow 聚合不能再次改变 payload 数量。

DeepEP 经过 rank-level 去重后，再按 `(src_rank, dst_rank)` 聚合，并由
`chunk_tokens` 拆成多个逻辑 network task。

## 3. 核心二：目标端转发

### 3.1 Relay 选择

设每台服务器有 `G` 张 GPU：

```text
server(rank) = rank // G
local(rank)  = rank % G
```

对于跨服务器逻辑传输 `src_rank -> dst_rank`，DeepEP 选择目标服务器中与 source
GPU local index 相同的 relay：

```text
dst_relay = server(dst_rank) * G + local(src_rank)
```

因此 fabric flow 始终是：

```text
src_rank -> dst_relay
```

在 MpRail 中二者 local index 相同，也就是同 rail，fabric flow 不需要经过 Spine。
随后由目标服务器的高速 FullMesh 完成：

```text
dst_relay -> dst_rank
```

### 3.2 一个 DAG task，两个串行 flow

DAG 中仍只生成一个逻辑 network task，端点写真实 source 和 destination rank：

```text
task_id barrier_id | src_rank dst_rank | transfer_bytes 0 | predecessors |
server_forward src_relay:<src_rank> dst_relay:<dst_relay>
```

关键约束是：

```text
src_relay == src_rank
```

现有 `server_forward` 状态机会因此跳过 `src_local`，只执行：

```text
Phase 1 fabric:    src_rank -> dst_relay
等待整个 flow 完成

Phase 2 dst_local: dst_relay -> dst_rank
等待整个 flow 完成
```

如果 `dst_relay == dst_rank`，第二阶段也跳过，此时只有 fabric flow。DAG task 的
完成回调在最后一个实际 flow 完成后触发，所以 Expert/Combine 后继自然等待目标端
转发结束。

这不是把两个 flow 展开成两个 DAG task；两个 subflow 由 HTSim 的
`server_forward` 状态机管理，属于同一个逻辑 task 和同一个完成 barrier。

### 3.3 示例

在 `G=8` 的 EP32 专用拓扑中：

```text
逻辑传输：0 -> 9
dst_relay = 8
DAG route：server_forward src_relay:0 dst_relay:8
实际 flow 1：0 -> 8   # fabric，同 rail 0
实际 flow 2：8 -> 9   # 目标服务器内部 FullMesh
```

另一个例子：

```text
逻辑传输：2 -> 13
dst_relay = 10
DAG route：server_forward src_relay:2 dst_relay:10
实际 flow 1：2 -> 10  # fabric，同 rail 2
实际 flow 2：10 -> 13 # 目标服务器内部 FullMesh
```

同服务器的 `src_rank -> dst_rank` 不使用 `server_forward`，直接生成普通本地
network task；`src_rank == dst_rank` 时不生成 network task。

## 4. Dispatch

对每条 router assignment 先得到真实 expert home rank，再应用默认去冗余：

```text
dst_rank = placement.expert_rank(expert_id)
payloads[(src_rank, dst_rank)] = unique (src_rank, token_id)
```

每个 chunk 的字节数：

```text
dispatch_bytes = unique_token_count * H * bytes(dispatch_dtype)
```

- 同服务器：普通 local flow。
- 跨服务器：一个带 destination-side `server_forward` route 的逻辑 task。
- Expert rank 等待发往自己的全部 dispatch task 完成。
- 原始 route count 仍用于 Expert FFN FLOP，不使用去重后的 payload count。

## 5. Combine

同一 token 在同一 execution rank 上的多个 expert output 先在该 rank 形成一个返回
payload，再沿反方向返回 origin rank：

```text
logical src = expert execution rank
logical dst = original source rank
```

Combine 同样应用 destination-rank 去冗余，并使用目标端转发：

```text
dst_relay = server(origin_rank) * G + local(execution_rank)
server_forward src_relay:<execution_rank> dst_relay:<dst_relay>
```

每个 chunk 的字节数：

```text
combine_bytes = unique_token_count * H * bytes(combine_dtype)
```

origin rank 的 `combine_reduce` 等待所有返回 task 以及本地 expert 结果完成。

## 6. DAG 与 overlap

```text
Attention
  -> Router
  -> DeepEP dispatch task/chunk
  -> destination-rank Expert FFN
  -> DeepEP combine task/chunk
  -> origin-rank Combine Reduce
```

- 每个逻辑 task 默认使用独立 barrier。
- `chunk_tokens` 只把聚合消息拆成多个完整 flow task。
- Expert 只等待发往自身的 dispatch chunks，不等待其他 rank。
- Combine Reduce 只等待返回自身的 combine chunks，不使用全局 collective barrier。
- 不监听单条 flow 的第 N 个 packet；每个 chunk 在完整 flow 完成后释放依赖。
- 通信可以与没有依赖冲突的其他 microbatch compute task 重叠。

计算仍使用 H100 理论 `compute_us` 占位；通信重叠阶段固定预留 20 SM。这是本项目
成本模型，不是对 DeepEP kernel occupancy 的复刻。

## 7. 配置与 Manifest

主算法名统一为：

```text
--algorithm deepep
```

核心配置：

```yaml
algorithm: deepep
workload_scope: training_prefill_forward
token_payload_policy:
  deduplicate: true
  scope: destination_rank
forwarding:
  mode: destination
  relay_coordinate: source_local_index
  completion: full_message
chunk_tokens: 32
dispatch_dtype: fp8
combine_dtype: bf16
```

不提供 hybrid/direct/low-latency 子模式；direct/no-dedup 对照由 NCCL builder
表达。

报告至少记录：

```text
route_count
unique_token_payload_count
deduplicated_route_count
logical_transfer_task_count
server_forward_task_count
dispatch_bytes
combine_bytes
```

## 8. 验收标准

1. 同 token 的两个 expert 位于同一 dst rank 时，DeepEP 只计算一份 payload 字节，
   但 Expert FFN 仍计算两个 routes。
2. 同 token 的 experts 位于不同 dst ranks 时，每个 dst rank 各保留一份 payload。
3. 跨服务器 `0 -> 9` 的 DAG task 写真实端点 `0 9`，route 必须是
   `server_forward src_relay:0 dst_relay:8`。
4. HTSim 日志中该 task 严格执行 `fabric 0 -> 8`、`dst_local 8 -> 9`，且逻辑
   task 只在第二个 flow 完成后释放 barrier。
5. 跨服务器同 index 目标，例如 `0 -> 8`，只执行 fabric flow。
6. 同服务器目标只走 FullMesh，不进入 fabric。
7. dispatch 使用 dispatch dtype，combine 使用 combine dtype。
8. NCCL 与 DeepEP 使用相同 assignments 和 placement 时，NCCL payload 字节不少于
   DeepEP，并且 NCCL 不出现 `server_forward`。

## 9. 代码依据与边界

核心行为的理解来自 `/home/xuheng/DeepEP` 中训练/prefill 数据路径；源码用于确认
同 local-index RDMA 与目标端 NVLink forwarding 的设计目的。本文定义的是本项目
可验证的 workload 模型，不声明与某个 DeepEP commit 的 kernel 时序或字节布局完全
一致。
