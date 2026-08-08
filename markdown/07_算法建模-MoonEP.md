# 算法建模：MoonEP 核心抽象

## 1. 项目定义

MoonEP 官方实现的动态冗余 expert 和 perfect balance 作用于单个 NVLink domain。
本项目不把官方代码描述成多服务器算法，而是抽取其核心 planner，按服务器独立
执行，再与 DeepEP 的跨服务器 transport 组合：

```text
scale-out:  DeepEP hierarchical RDMA/NVLink transport
scale-up:   per-server MoonEP dynamic expert replication and balancing
```

因此本项目中的 `moonep` 表示：

```text
DeepEP 跨服务器去冗余通信
  +
每台目标服务器内部独立的 MoonEP 临时 expert replica
  +
按最终 execution rank 生成计算和 combine
```

这是用于网络/DAG 仿真的组合模型，不声明 MoonEP 官方仓库提供了 RDMA 或多节点
实现。

## 2. 两层 placement

每条 router assignment 先确定 logical expert 的 home rank 和 home server：

```text
home_rank   = placement.expert_rank(expert_id)
home_server = server(home_rank)
```

MoonEP planner 只能在 `home_server` 内选择 execution rank：

```text
server(execution_rank) == server(home_rank)
```

这个不变量意味着：

- logical expert 不跨服务器迁移；
- 跨服务器流量的目标 server 仍由原始 expert placement 决定；
- MoonEP 只解决每台服务器内部 GPU 之间的 expert 负载不均衡；
- 不同服务器的总 route 数可以不同；
- 同一服务器内部各 rank 的 real route 数应尽量相等。

“临时专家迁移”在本模型中准确表示为“临时专家复制”：home expert 保留，目标
rank 增加 replica，部分 token route 改到 replica 执行。

## 3. Per-server Planner

### 3.1 输入分组

planner 按 expert home server 对 route 分组：

```text
routes_by_server[server(home_rank(expert))]
```

每个服务器只处理自己的 logical experts，并独立生成：

```text
execution_rank[(src_rank, token_id, topk_slot)]
replicas_by_rank[rank]
real_routes_by_rank[rank]
```

### 3.2 均衡目标

设服务器 `s` 收到 `T_s` 条 logical expert routes，服务器内有 `G` 张 GPU：

```text
base  = T_s // G
extra = T_s % G
```

planner 确定性地让 `extra` 个 rank 接收 `base+1` 条 real routes，其余 rank 接收
`base` 条。若 `T_s` 可被 `G` 整除，则服务器内完全相等。

这不是跨服务器全局均衡：服务器 A 可以每 rank 处理 2 条 route，服务器 B 每 rank
处理 1 条 route。最终每个 rank 的 compute task 必须使用自己的实际 route/padding，
不能写成全局统一时间。

### 3.3 Replica 选择

对 hot expert：

1. home rank 首先是候选 execution rank；
2. 当 home rank 容量不足时，只在同一服务器选择有剩余容量的 replica rank；
3. 每个 rank 最多接收 `replicas_per_rank` 个临时 experts；
4. 若容量或 replica slots 不足，生成阶段确定性失败。

当前 planner 使用 hot-expert-first、remaining-capacity-first 的确定性近似，不复刻
官方 GPU planning kernel 的指令级行为。固定输入必须得到固定 plan。

## 4. Expert Weight Prefetch

每个临时 replica 产生一条服务器内部权重传输：

```text
home_rank(expert) -> replica_rank
payload_kind = expert_weight_prefetch
```

字节数包含 gate/up/down 三个矩阵：

```text
expert_weight_bytes = 3 * H * H_ff * bytes(weight_dtype)
```

由于 replica 与 home 位于同一服务器，该 flow 只走高速 FullMesh，不使用 RDMA
或 Spine。

Weight prefetch 与 token dispatch 都可以在 planning 完成后启动；Expert FFN 同时
等待该 rank 的 prefetch 和 dispatch 到达。因此 DAG 可以表达两者 overlap，而不
需要模拟 TMA/HBM 细节。

## 5. Token Dispatch

planner 先把每条 route 映射到最终 execution rank，再应用共享默认去冗余：

```text
dedup key = (src_rank, token_id, execution_rank)
```

同一个 token 的多个 expert route 若最终落在同一 execution rank，只发送一份
hidden payload；逻辑 expert route metadata 和 Expert FFN route count仍全部保留。

传输规则复用 DeepEP 分层 transport：跨服务器先按 execution server 合并 hidden，
由 `src_rank` 向同 index relay 发送一份 `dispatch_fabric`，再显式生成
`dispatch_local` fanout。同服务器只生成 local flow。

## 6. Expert Compute

每个 execution rank 按 planner 实际分配的 routes 统计各 expert group：

```text
real_routes[rank, expert]
padded_routes[rank, expert]
  = ceil(real_routes / token_padding) * token_padding
```

理论计算量：

```text
expert_flops[rank]
  = sum_expert(padded_routes[rank, expert]) * 6 * H * H_ff
```

因此 MoonEP 重新分配会直接改变 `.dag` 中：

- 哪个 rank 创建 Expert FFN task；
- 每个 Expert FFN 的 `compute_us`；
- dispatch/combine 的 src-dst matrix；
- Expert FFN 等待哪些 weight/token flows。

计算仍是 H100 理论固定时长占位，不模拟 grouped GEMM kernel、HBM、zero-copy 或
动态 SM 调度。

## 7. Combine

Combine 从实际 execution rank 返回原始 token rank，并复用 dispatch 的 planner：

```text
logical src = execution_rank
logical dst = original source rank
```

payload 同样先按 `(origin token, execution_rank)` 去重，再按 execution server 合并。
跨服务器 combine 先把 execution rank partial 通过 `combine_local_reduce` 汇聚到
与 origin 同 local-index 的 relay，再由一个 `combine_fabric` 返回 origin。origin
rank 的 Combine Reduce 等待全部 remote server payload 和本地 expert 结果。

planner 不在 combine 阶段重新运行，replica placement 也不能改变。

## 8. DAG 结构

```text
Router
  -> per-server MoonEP planning
       +-> local expert_weight_prefetch --------+
       +-> DeepEP-style token dispatch ---------+-> Expert FFN
                                                   -> DeepEP-style combine
                                                   -> Combine Reduce
```

依赖要求：

- 每台服务器有自己的 planning barrier，不建立 MoonEP 全局服务器 barrier；
- prefetch 只等待所在服务器的 planning；
- dispatch 只等待目标 execution server 的 planning；
- Expert FFN 等待本 rank 的 planning、prefetch 和 dispatch；
- combine chunk 只等待对应 execution rank 的 Expert FFN；
- 每个 transfer chunk 默认独立 barrier。

这里的 `per_server_planning_proxy` 是 MoonEP 现有的 GPU compute 占位 task，会进入
每张 GPU 的 compute stream；它没有 `logical_resource=cpu`，不是 CPU planner lane。
ProbeEP 不复用这条 task，其 migration planning 完全位于离线 workload 生成阶段。

## 9. 配置与 Manifest

```yaml
algorithm: moonep
scale_out_transport: deepep_hierarchical
replica_scope: home_server
planner: deterministic_per_server_capacity_balancer_v2
replicas_per_rank: 2
token_padding: 128
chunk_tokens: 32
token_payload_policy:
  deduplicate: true
  scope: destination_rank_then_server
```

manifest 至少记录：

```text
routes_by_server
target_routes_by_rank
real_routes_by_rank
padded_routes_by_rank
replicas: expert_id, home_rank, execution_rank
expert_weight_prefetch_bytes
hierarchical_transfer
unique_server_payload_count
scaleout_deduplicated_route_count
dispatch/combine bytes
```

## 10. 验收标准

1. replica 的 home/execution rank 必须位于同一服务器。
2. expert weight prefetch 只生成 local flow，且字节等于完整 expert 权重。
3. 每台服务器独立达到 floor/ceil 均衡，不要求不同服务器的 rank route 数相等。
4. token dispatch/combine 的 logical dst 是最终 execution/origin rank。
5. 所有跨服务器 token flow 使用 DeepEP 显式 fabric/local hierarchy legs。
6. 同 token、同 execution rank 的多 expert routes 只产生一份 payload。
7. Expert FFN route count 不受 payload 去冗余影响。
8. EP32 测试中可同时观察服务器内权重复制和跨服务器 token flow。
9. 生成的完整 DAG 能由 HTSim 加载并完成。

## 11. 当前边界

首版不实现：

- MoonEP 官方 GPU planner 的逐指令复刻；
- zero-copy buffer、dispatch epilogue/combine prologue 的本地内存时间；
- remote-direct expert weight 访问；
- backward、replica gradient reduce 和参数更新；
- replica cache 跨 invocation 复用。

这些不影响本轮核心问题：临时 expert replica 产生额外权重 flow，execution rank
重映射改变 token通信矩阵和每 rank 计算，跨服务器部分由 DeepEP transport 仿真。
