# 算法建模：NCCL MoE All-to-All

## 1. 目标与边界

本文定义 NCCL 风格 MoE All-to-All 的 workload 语义，用于和 DeepEP、MoonEP
在同一个模型、router assignment、expert placement 和 MpRail 拓扑下比较。

首版只建模 collective 产生的逻辑数据移动和 DAG 依赖：

```text
dispatch all-to-all -> expert compute -> combine all-to-all -> local reduce
```

它不模拟 NCCL 内部 ring/tree、channel、protocol、CUDA kernel、LL/LL128/Simple
选择或实际集合通信调度。MoE 路由通常让各 rank 消息大小不同，因此本文的
All-to-All 是逻辑 All-to-All/All-to-Allv workload，而不是固定等长消息假设。

实现状态：`pysrc/moe_dag` 提供 `nccl` builder 和 CLI 选项，并按本文契约生成
rank-direct/no-dedup DAG。NCCL 内部 collective kernel 仍不在仿真范围。

## 2. 与 DeepEP 的两个本质差异

| 能力 | NCCL MoE All-to-All | DeepEP |
|---|---|---|
| token hidden payload 去冗余 | 关闭；每条 top-k expert route 都保留一份 payload | 开启；按 destination rank 去重 |
| 服务器内部 relay forwarding | 不使用 | 使用同 local-index relay，再执行目标端本地转发 |
| fabric 目的端 | 实际 expert 所在 `dst_rank` | 目标服务器的同 local-index relay rank |
| 非同 local-index 跨服务器传输 | 直接进入跨 rail Leaf-Spine 路径 | 先送同 index relay，避免该次 fabric 跨 rail |

NCCL 并不是不聚合 flow。相同 `(src_rank, dst_rank)` 的多条 route 可以合并为
一个或多个 chunk flow，以控制 HTSim task 数量；但合并时必须保留 route
multiplicity，不能把重复 token payload 从字节数中删除。

## 3. 共享去冗余基座

去冗余通信属于第一层 MoE 算法基座的公共能力，不应硬编码在 DeepEP builder
内部。完整公共契约见
[02_第一层-MoE算法工作负载建模.md](02_第一层-MoE算法工作负载建模.md)，
目标接口应显式记录：

```text
TokenPayloadPolicy
  deduplicate: bool
  scope: none | destination_rank
```

默认策略：

```text
deduplicate = true
```

- DeepEP、MoonEP 以及后续优化算法默认开启 destination-rank 去重。
- NCCL 是唯一默认关闭该能力的算法：`deduplicate=false`、scope 为 `none`。
- emitter 只负责把 planner 给出的 transfer item 降低为 DAG task，不能再次猜测
  或改变去重策略。

### 3.1 输入 route item

原始 router 输出的每个条目都必须保留 top-k multiplicity：

```text
(src_rank, token_id, topk_slot, expert_id, dst_rank)
```

例如同一个 token 的两个 top-k slot 都落到 rank 9：

```text
(0, token 7, slot 0, expert 9,  dst 9)
(0, token 7, slot 1, expert 41, dst 9)
```

NCCL dispatch 计算两份 hidden payload；开启 rank-level 去重的算法只计算一份
hidden payload，并把两个 expert 选择作为 metadata。

### 3.2 去重层次

```text
none:
  每条 top-k route 一份 payload

destination_rank:
  同一 src token 发往同一 dst rank 只保留一份 payload
```

去重只改变 payload item 和 `transfer_bytes`，不能改变 expert route 数、expert
compute FLOP、top-k combine 语义或 routing metadata。

## 4. NCCL Dispatch

### 4.1 目标 rank

每条 route 的网络目的端就是 expert home rank：

```text
dst_rank = placement.expert_rank(expert_id)
```

不创建 destination-side relay，也不附加 `server_forward` route。

### 4.2 字节数

单条 dispatch route 的 payload：

```text
dispatch_route_bytes = H * bytes(dispatch_dtype)
```

聚合后的 `(src_rank, dst_rank)` payload：

```text
dispatch_bytes[src, dst]
  = route_count[src, dst] * dispatch_route_bytes
```

这里使用 `route_count`，不是 unique token count。之后才按 `chunk_bytes` 或
`chunk_routes` 拆成多个 network task。

### 4.3 MpRail 路径

在 32-GPU 专用测试拓扑中：

```text
server(rank) = rank // 8
rail(rank) = rank % 8
```

NCCL 直接使用 `src_rank -> dst_rank`：

- 同服务器：走 server-local FullMesh。
- 跨服务器且 `src % 8 == dst % 8`：走同 rail L0，不经过 Spine。
- 跨服务器且 `src % 8 != dst % 8`：走 `src L0 -> L1 Spine -> dst L0`。

第三种路径正是 NCCL 相比 DeepEP relay forwarding 更容易形成 Spine incast 的
来源。NCCL builder 不应附加 `server_forward` route；普通 MpRail 路由根据真实
端点自然选择 same-rail 或 cross-rail 路径。

## 5. Expert Compute

每个 destination rank 的 expert compute 等待所有发往该 rank 的 dispatch chunk
完成。计算量仍按真实 top-k route 数计算：

```text
expert_flops[rank]
  = route_count[rank] * 6 * H * H_ff
```

本地 route 不产生 network task，但必须作为该 rank 的已到达输入计入 expert
compute。首版沿用 H100 理论 cost model 和固定 `compute_us`。

## 6. NCCL Combine

Combine 沿 dispatch route 反向返回，每条 expert route 保留一份输出 partial：

```text
combine src = expert execution/home rank
combine dst = original source rank
combine_route_bytes = H * bytes(combine_dtype)
```

相同 `(execution_rank, origin_rank)` 的 route 可以聚合成 chunk flow，但仍按
route count 计算字节。NCCL combine 不使用 destination-side `server_forward`；
目的端收到全部 top-k partial 后再执行 `combine_reduce` compute task。

## 7. DAG 依赖

```text
Attention
  -> Router
  -> NCCL dispatch chunks
  -> destination-rank Expert FFN
  -> NCCL combine chunks
  -> origin-rank Combine Reduce
```

依赖要求：

- dispatch chunk 只等待所属 microbatch 的 Router。
- Expert FFN 等待该 rank 的全部 remote dispatch chunk 和本地 route 输入就绪。
- combine chunk 只等待其 source/execution rank 的 Expert FFN。
- Combine Reduce 等待该 origin rank 的全部 remote combine chunk，以及本地
  expert partial。
- 不把不同 rank 的 All-to-All task 放入一个全局 barrier；每个 task 默认独立
  barrier，避免人为引入 collective-wide stage barrier。

## 8. 配置与 Manifest

建议 CLI：

```bash
--algorithm nccl
```

manifest 至少记录：

```yaml
algorithm: nccl_alltoall
collective_semantics: alltoallv
token_payload_policy:
  deduplicate: false
  scope: none
hierarchical_forwarding: false
transfer_aggregation: src_dst_chunk
```

统计必须同时输出：

```text
route_count
unique_token_count
dispatch_bytes
combine_bytes
same_server_bytes
same_rail_bytes
cross_rail_bytes
```

这样才能直接量化 NCCL 无去冗余和无 relay forwarding 带来的额外字节与 Spine
流量。

## 9. 验收测试

首版实现必须覆盖：

1. 一个 token 的两个 expert 在同一 dst rank：NCCL 发送两份 payload，DeepEP
   rank 去重只发送一份。
2. 一个 token 的 experts 分布在不同目标 rank：NCCL 按每条 route 发送，DeepEP
   按每个 destination rank 发送。
3. `src=0, dst=8`：跨服务器同 rail，不经过 Spine。
4. `src=0, dst=9`：跨服务器不同 rail，实际经过 L1 Spine。
5. NCCL task graph 中不存在 `server_forward` route。
6. dispatch/combine 总字节严格等于 route multiplicity 乘 dtype payload。
7. NCCL 与 DeepEP 使用完全相同的 assignments 和 placement 做对照。
8. 生成的完整 DAG 能在 EP32、plane=1、400 Gbps 专用拓扑上完成。

## 10. 不在首版范围

- NCCL ring/tree/channel 拓扑展开；
- NCCL protocol 和 chunk/slice 内部流水线；
- CUDA Graph、stream 排队和 NCCL kernel SM 占用；
- All-to-All 算法自动选择；
- backward、梯度通信和容错。
