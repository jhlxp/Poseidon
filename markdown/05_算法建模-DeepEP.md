# 算法建模：DeepEP 核心路径

## 1. 建模范围

本项目只抽象 DeepEP 面向训练和 prefill 的核心数据移动：

```text
destination-rank 去冗余
  -> destination-server 去冗余
  -> 跨机 RDMA
  -> 目标服务器 NVLink fanout
  -> Expert FFN
  -> 专家服务器 NVLink reduce/gather
  -> 跨机 RDMA 返回
```

不复刻 CUDA kernel、notify/layout、TMA、QP、low-latency decode 或单 flow 内
packet-progress 事件。`chunk_tokens` 通过多个完整 flow task 表达 chunk pipeline。

NCCL 是 direct/no-dedup 对照：每条 top-k route 都保留独立 payload，直接发往真实
execution rank，不进行 DeepEP 的服务器级合并。

## 2. 两级去冗余

Router 的原始 route 是：

```text
(src_rank, token_id, topk_slot, expert_id, execution_rank)
```

第一层按 rank 去冗余：

```text
rank payload key = (src_rank, token_id, execution_rank)
```

同一 token 的多个专家落在同一 execution rank 时，hidden 只传一份，但 expert
route、top-k slot 和 Expert FFN 计算量全部保留。

第二层只作用于 scale-out RDMA：

```text
server payload key = (src_rank, token_id, execution_server)
```

同一 token 在目标服务器内命中多个 rank 时，跨机只发送一份 hidden；目标服务器
收到后再用 NVLink 向各个唯一 execution rank fanout。

Manifest 同时记录：

```text
route_count
unique_token_payload_count       # rank payload
unique_server_payload_count      # scale-out payload
deduplicated_route_count         # route - rank payload
scaleout_deduplicated_route_count# route - server payload
```

因此必须满足：

```text
route_count >= unique_token_payload_count >= unique_server_payload_count
```

## 3. Relay 与 Rail

设每台服务器有 `G` 张 GPU：

```text
server(rank) = rank // G
local(rank)  = rank % G
```

源 rank 向目标服务器发送时，选择目标服务器中相同 local index 的 relay：

```text
relay = execution_server * G + local(src_rank)
```

因此 `src_rank -> relay` 始终位于同一 rail。在 MpRail 中它不经过 Spine；这正是
DeepEP 利用同 index NIC、再在目标服务器内部转发的抽象。

DeepEP 不再把两段通信封装为 `server_forward` route。RDMA 和 NVLink 都是显式
DAG transfer task，具有各自的 barrier、端点、字节和完成事件。这样一个 RDMA
task 可以成为多个 local fanout task 的共同前驱。

`server_forward` 仍是 MpRail 的通用源路由能力，文档见
[10_MpRail源路由与服务器转发.md](10_MpRail源路由与服务器转发.md)，但不是当前
DeepEP/EPLB/MoonEP builder 的 lowering 方式。

## 4. Dispatch

跨服务器 Dispatch 对每个 `(src_rank, token_id, execution_server)` 生成一份
server payload：

```text
dispatch_fabric:
src_rank -> same-index relay
bytes = server_payload_count * H * bytes(dispatch_dtype)
```

fabric task 完成后，对该 chunk 中每个唯一 execution rank 生成本地 fanout：

```text
dispatch_local:
relay -> execution_rank
bytes = rank_payload_count * H * bytes(dispatch_dtype)
predecessor = corresponding dispatch_fabric task
```

若 execution rank 就是 relay，不创建自环 local task，fabric task 直接作为该 rank
的 arrival。若 origin 与 execution rank 在同一服务器，只创建普通
`dispatch_local`；若二者相同，则不创建 network task。

Expert FFN 只等待送达本 rank 的 dispatch task。Expert FLOP 仍按原始 route count
计算，不按去冗余后的 payload 数计算。

## 5. Combine

Combine 与 Dispatch 不是简单反向复用。DeepEP 的 multiple reduction 语义先在专家
服务器内汇聚，再跨机返回 origin：

```text
combine_local_reduce:
execution_rank -> relay matching local(origin_rank)
bytes = rank_payload_count * H * bytes(combine_dtype)

combine_fabric:
relay -> origin_rank
bytes = server_payload_count * H * bytes(combine_dtype)
predecessors = all local partials in the corresponding server chunk
```

若 execution rank 就是 relay，`combine_fabric` 直接等待该 rank 的 Expert FFN。
若专家与 origin 在同一服务器，只创建本地 combine；同 rank 结果由最终 reduce 直接
消费，不创建自环 transfer。

origin rank 的 `combine_reduce` 等待所有远端 server payload、本地 partial 和本地
Expert FFN 完成。

## 6. 完整例子

EP32、`G=8`，token 来自 rank 0，三个 expert route 分别落在 rank 9、9、10：

```text
原始 expert routes: 3
rank payloads:       2   # rank 9、rank 10
server payloads:     1   # 都在 server 1

Dispatch:
0 -> 8       one RDMA payload
8 -> 9       one NVLink payload, carrying two expert routes as metadata
8 -> 10      one NVLink payload

Combine:
9 -> 8       one NVLink partial
10 -> 8      one NVLink partial
8 -> 0       one reduced RDMA payload
```

这个例子保留 3 条 Expert FFN routes，但每个方向只有 1 份跨机 payload。

## 7. DAG 与 overlap

```text
Attention -> Router -> dispatch_fabric -> dispatch_local -> Expert FFN
          -> other independent microbatch compute can overlap
Expert FFN -> combine_local_reduce -> combine_fabric -> Combine Reduce
```

- 每个 transfer task 默认一个独立 barrier。
- 同一 fabric chunk 的 local fanout 必须等待该 fabric task。
- 不同源、服务器或 chunk 的独立 flow 可以并发。
- 每个 chunk 在完整 flow FCT 后释放后继。
- communication stream 内 phase 顺序保持串行，compute stream 可与通信重叠。
- Compute 仍为固定 `compute_us`；重叠阶段静态预留 20/132 SM。

## 8. 专家布局与测试路由

标准 DSV3 EP32 使用连续 expert placement，符合每 rank 持有一段 expert index 的
口径：

```text
execution_rank = floor(expert_id * num_ranks / num_experts)
```

测试 router 使用固定种子 0 的 expert permutation，并按 TopK 分组循环。这样总
expert route 数严格均衡，同时真实产生同 rank、同服务器命中。旧的
`expert_id % num_ranks` 加连续 TopK 会让 8 个 expert 永远落到 8 个不同 rank，无法
验证去冗余，已不再使用。

## 9. 配置与 Manifest

```yaml
algorithm: deepep
workload_scope: training_prefill_forward
token_payload_policy:
  deduplicate: true
  scope: destination_rank_then_server
forwarding:
  mode: hierarchical_scaleout_scaleup
  relay_coordinate: source_local_index
  dispatch: fabric_then_local_fanout
  combine: local_reduce_then_fabric
```

`hierarchical_transfer` 按四类 leg 分别记录 task、payload 和字节：

```text
dispatch_fabric
dispatch_local
combine_local_reduce
combine_fabric
```

## 10. 验收标准

1. 同 token、同 execution rank 的多个 expert routes 只保留一份 rank payload。
2. 同 token、同 execution server 的多个 rank payload 只保留一份 RDMA payload。
3. Expert FFN route count 不因通信去冗余而减少。
4. Dispatch fabric 到达同 index relay，local fanout 显式依赖 fabric。
5. Combine 先等待专家服务器内 local partial，再由 relay 发一份 RDMA 回 origin。
6. 同服务器 leg 只走 7200 Gbps FullMesh；fabric leg 受 400 Gbps RDMA 限制。
7. DeepEP DAG 不含 `server_forward` route；四类 hierarchy leg 可在 task map 恢复。
8. NCCL 不含 hierarchy leg，并保留每条 top-k route 的 payload multiplicity。
9. EP32 smoke/full 中 `route > rank payload > server payload`，HTSim 完成全部 task。

源码依据是 `/home/xuheng/DeepEP/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
中的 scaleout destination 去重和 scaleup forwarding，以及
`deep_ep/buffers/elastic.py` 中默认开启的 multiple reduction。本文抽象数据移动
本质，不声明与某个 kernel 的指令级时序完全一致。
