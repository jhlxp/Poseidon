# 算法建模：ProbeEP 两阶段专家迁移与反馈式 NIC 窗口

> 实现状态：全局 server-first quota + padding refinement、热点 expert 优先、
> server 内 padding-aware capacity packing、动态
> expert weight bytes、逐 NIC 双端 admission 和 Attention/MoE 独立反馈窗口已实现。
> 当前版本使用目标比例控制器，以 `0.90*Cmax` 为工程目标，不再使用固定
> `1.05/0.95` AIMD。

## 1. 定义

ProbeEP 是一个面向训练/prefill forward 的跨服务器临时 expert replica 算法。
它由三个职责严格分离的模块组成：

```text
Compute Migration Planner
  先在全 EP 域均衡 server compute，再在 server 内均衡 GPU compute

NIC Budget Controller
  分别维护 Attention/MoE 参考时间和每张 NIC 的迁移字节窗口

Weight Chunk Scheduler
  在窗口内把完整 expert 的权重 chunks 分给轻载 NIC
```

核心优化目标只有计算均衡，按层级顺序解决：

```text
1. minimize max_server(padded_expert_compute_proxy)
2. minimize max_rank(expert_compute_us) within every admitted server mapping
```

所有 server/GPU 在 MoE 后续同步边界上由最慢 participant 决定完成时间。
因此不能把四台 server 当成四个独立优化问题：先全局处理 server 间差距，
再在各 server 内使用高带宽 NVLink 做二次均衡。这是工程启发式，目标是尽量
降低最慢 GPU，不要求 raw routes 或时间数学上 100% 相等。

但“可被掩盖的通信时间”不能再使用一个无类型的 compute 变量。Attention 的 shape
和 token/rank 基本固定，时间相对静态；MoE Expert FFN 时间随 Gate 分布、迁移结果和
padding 动态变化。控制器必须同时维护两类参考时间，并由当前通信窗口明确选择其中
一种。

网络不是 expert source/destination 规划的目标或约束。网络反馈只控制“本轮允许接纳
多少迁移”，NIC 负载只影响已接纳 expert 的 weight chunks 使用哪些 rail。

ProbeEP 是独立算法。`ProbeEPBuilder` 不导入、继承或调用 `MoonEPBuilder`；它在自己
的代码中实现服务器间和服务器内两阶段 planner，二者可以独立演进。Token
dispatch/combine 继续使用共享的 DeepEP hierarchy 数据面。

## 2. Overlap 观测，不把整个 MoE invocation 叫作一次采样

一次完整 MoE invocation 是：

```text
Router/Gate -> [Weight Migration + Dispatch communication stage]
            -> Expert FFN -> Combine
```

它同时包含两个不同的通信 phase，不能把整个 invocation 的起止时间作为一个 ProbeEP
控制样本。正确的观测单位是 compute stream 与 communication stream 在相邻同步边界
之间形成的 overlap window。每条观测显式记录：

```text
communication_kind = dispatch | combine
compute_kind       = attention | moe
compute_us_by_rank
communication_us_by_rank
```

双 microbatch 在每层可观察为四个主要 compute stage：

```text
S0: A0 || previous-layer C1（第一层没有 previous C1）
S1: A1 || W0+D0       <- Attention control chain
S2: E0 || W1+D1       <- MoE control chain
S3: E1 || C0
tail: C1 || next-layer A0
```

`W` 是临时 expert weight migration。Remote Weight RDMA TX 在同一 source rail 的
Dispatch fabric TX 之前完成；不同 rail、local NVLink Weight 和 Dispatch 可按真实链路
独立推进。ProbeEP controller 只消费 S1/S2 两类 `Weight+Dispatch` 观测。Combine 仍由
HTSim 完整执行并进入
timeline/link-load telemetry，但它不包含专家权重、不更新迁移 budget，也不参与
`D_i`、`M_i` 或 `Bhard_i` 的字节计算。

### 2.1 Per-source-rail 立即推进，不设全局 Weight barrier

32 个 rank 的 scale-out TX 对应各自的 rail。对 source rank/rail `i`：

```text
完成由 rail i 发送的本 invocation remote Expert Weight RDMA tasks
  -> 立即启动由 rail i 发送的 Dispatch fabric tasks
```

rail `i` 不等待其他 31 个 rank、其他 rail、目的 RX 或无关 NVLink Weight tasks。ProbeEP
的 Weight 和 Dispatch 在 stream lowering 中属于同一个 communication stage；顺序由
`same_source_rail_remote_weight_tx` task edges 表达，不使用 `prefetch phase -> dispatch
phase` join。若 rail `i` 没有 remote Weight TX，它在 Router/planner ready 后即可发送
Dispatch。不得引入 per-rank all-prefetch join 或全局 `all_weight_done` barrier。

TX 和 RX 属于同一个 network stage。它们可以 full-duplex 并行，因此 stage 时间不是
`TX time + RX time`，而是从共同 release 到 TX/RX 都完成：

```text
release_i = 与该 Dispatch observation 配对的 compute task start
done_i    = max(last Weight/Dispatch TX done,
                last Weight/Dispatch RX done)
N_i       = done_i - release_i
```

这里的 Weight/Dispatch stage 包含完整数据面：NVLink scatter/gather/prefetch、Expert
Weight RDMA、Dispatch fabric 和 Dispatch local forwarding，也包含依赖等待和网络空隙。
这才回答“整个 Network TX/RX stage 是否隐藏在计算里”。

诊断必须把以下分量分开记录，不能只用一种 Expert Weight 颜色或一个总字节解释 stage：

```text
remote_weight_rdma_bytes/time
local_weight_nvlink_bytes/time
dispatch_fabric_bytes/time
dispatch_local_bytes/time
complete_weight_dispatch_stage_elapsed
```

NIC TX/RX active-time union 仍单独保留为利用率诊断：

```text
U_i = max(union_active(TX fabric intervals),
          union_active(RX fabric intervals))
```

`U_i` 不包含 idle gap 和 local forwarding，不能代替 `N_i` 回答 end-to-end overlap。

Combine telemetry 记录其 fabric active time 和在 phase 释放时与之配对的计算类型/时间。
一个较长 Combine 可能跨过 MoE tail，继续与下一层 Attention overlap，因此不把整个
Combine phase 强行除以单个计算块并生成 controller `N/C`；精确跨段关系由 GPU timeline
展示。

对于两个 microbatch 和 `L` 个 MoE layer：

```text
MoE invocations                 = 2 * L
Dispatch control observations   = 2 * L
Combine telemetry observations  = 2 * L
all communication observations  = 4 * L
```

因此两层完整测试有 4 个 Dispatch 控制观测和 4 个 Combine telemetry；58 层实验有
116 个 Dispatch 控制观测，而不是把 116 个完整 MoE invocation 各自粗略称为“一次
采样”。Router/Gate 的概率抽样仍可称为 Gate sample，那是 workload 输入语义，与
ProbeEP feedback observation 不是同一个概念。

观测原始数据必须保留 per-rank 数组。EP32 两层专项测试的扁平 CSV 固定有
`8 observations * 32 ranks = 256` 行，每行包含 `compute_us` 和 `communication_us`；
跨 rank 最大值只用于 Dashboard 总览，不能替代 per-GPU feedback。

不同 layer 的 logical experts 是不同参数，但它们共享同一组 GPU/NIC。expert
migration plan 每个 invocation 重新计算；per-rank NIC budget 跨 layer 保留，因为它
学习的是共享 endpoint 上 `Weight+Dispatch` 可被计算掩盖的窗口。

### 2.2 Attention/MoE 两条 Dispatch 状态链

双 microbatch 的 compute stream 核心顺序是：

```text
Attention 0 -> Attention 1 -> MoE 0 -> MoE 1
```

因此控制器不再维护单一 `compute_max_us`，而是维护：

```text
Amax = max_rank(attention_compute_us_by_rank)
Mmax = max_rank(moe_compute_us_by_rank)
dispatch_overlap_compute_kind = attention | moe
Cmax = Amax if dispatch_overlap_compute_kind == attention else Emax
```

- `Amax` 来自与 `Weight+Dispatch` overlap 的 Attention 时间，通常跨层较稳定；
- `Mmax` 来自与 `Weight+Dispatch` overlap 的 Expert FFN 时间，随 Gate 分布和迁移结果
  动态变化；
- 每个 Dispatch observation 必须显式记录 `dispatch_overlap_compute_kind`，不能从
  invocation 编号或 task 名字隐式猜测；
- 当前双 microbatch 每层固定产生一个 Attention-chain Dispatch observation 和一个
  MoE-chain Dispatch observation；按实际 layer wavefront 是 `A,M,A,M,...`；
- 若对应类型尚无已完成反馈，Attention 可使用配置中的静态时间；MoE 只能使用明确
  标记的 analytical proxy，不能伪装成实测 FCT。

这里的“上个阶段”指当前通信真正与之 overlap、且在决策时已经可用的计算窗口。不能
拿未来尚未完成的 MoE 实测时间作为 oracle。

当前控制器对两条链都采用 `latest valid observation`：

```text
State_attention <- 最新完成的 Weight+Dispatch || Attention 观测
State_moe       <- 最新完成的 Weight+Dispatch || MoE 观测
```

EWMA/smoothing 可以作为后续可选策略，但首版不增加额外参数；先保证观测窗口、状态链
和因果关系正确。

## 3. Planner 与 DAG 边界

当前 invocation 的因果关系固定为：

```text
Router/Gate
    |
    v
[Weight Migration + Token Dispatch communication stage]
    |
    v
Expert FFN
    |
    v
Combine
```

方括号内是同一个 communication stage，不是先完成全部 Weight 再统一
释放 Dispatch。对每条 source rail，只有该 source rank 发出的 remote Weight
RDMA TX 是其 `dispatch_fabric` TX 的前驱；该 TX 完成后即可启动 Dispatch。
目的端 Weight RX/gather、local NVLink prefetch、其他 source rail 和全局 Weight
完成都不是这条 Dispatch TX 的前驱。没有 remote Weight TX 的 source
rail 可在 Router/Gate 依赖满足后直接启动 Dispatch。

Expert FFN 仍必须等待该 execution rank 需要的 token 和 expert weight 到达；
这是消费端正确性依赖，不是 Weight 与 Dispatch 之间的全局 barrier。

当前契约：

- planner 只产生 migration intents、execution placement 和 NIC chunk assignment；
- `.dag` 不生成 CPU planner task，不增加 `compute_us`、barrier 或 timeline lane；
- Router/Gate 完成后同时释放已经确定的 Weight Migration 和可执行的
  Token Dispatch；只为同 source rail 的 remote Weight RDMA TX 到 `dispatch_fabric`
  TX 添加顺序边；
- manifest 标记 `planner_runtime_model=not_in_dag`；
- planner 的真实实现和运行开销由论文中的独立 CUDA/C++ 实现与复杂度实验给出。

这里的 `planner` 是算法层的迁移决策逻辑，不表示 DAG 中存在一条 CPU 执行流。当前
实现边界如下：

| 工作 | 发生位置 | 生成 DAG task | 计入 HTSim makespan |
|---|---|---:|---:|
| Gate/Router | GPU compute stream | 是 | 是 |
| migration planning | Python 离线 workload 生成 | 否 | 否 |
| route lowering/稳定序列化 | Python 离线 workload 生成 | 否 | 否 |
| expert weight scatter/RDMA/gather | HTSim communication stream | 是 | 是 |
| Dispatch/Expert FFN/Combine | GPU compute/communication streams | 是 | 是 |

因此当前 ProbeEP 的运行时计数必须始终满足：

```text
cpu_task_count = 0
cpu_streams_global = 0
```

代码中的通用 two-stream scheduler 仍能识别 `logical_resource=cpu`，这是框架兼容
能力；ProbeEP builder 不生成这种 task，该分支不会参与当前 workload、timeline 或
makespan。

Python 生成器仍必须把最终 placement 展开到具体 route，才能生成逐 flow DAG。对
4096tpr/EP32 来说，这一步会访问 1048576 条 route；它属于离线仿真输入 lowering，
不是在线 planner，也不进入 makespan。ProbeEP 决策本身不要求对百万 route 做全局
排序；若 emitter 为稳定输出执行排序，同样只属于离线生成器开销。

## 4. 第一阶段：服务器间计算迁移

### 4.1 输入

Gate sample 先聚合为：

```text
routes_by_expert
routes_by_home_server
compute_us_per_expert_route
total_routes
target_routes_per_rank/server
padded_routes_by_server
```

当前 DSV3 expert 同构时，route 数与 FFN 时间成正比，因此服务器间 quota 使用 route
数建立第一个全局锚点。但 Expert FFN 实际使用按 `(server,expert)` 向上取整的
padding block；raw routes 完全相等时，server padded compute 仍可不等。因此 quota
之后还需要 padding-aware refinement，并允许 raw route 数对 quota 有小幅工程偏离。
未来支持异构 expert 时，两步的单位都需要升级为等价 compute cost。

### 4.2 全局 Quota Negotiation

旧的“每轮只找最多 server，再搬到最少 server”不再作为算法定义。新 planner 在整个
EP 域一次性计算所有 server 的 surplus/deficit，让所有不平衡 server 参与同一个
quota negotiation。已经达到 quota 的 server 不需要为了形式上的“全部搬运”产生
无效通信。

对同构 expert，均衡单位是 expert route，不是去冗余后的 network payload：

```text
total_routes = num_ranks * tokens_per_rank * topk
target_routes_per_rank = total_routes / num_ranks
target_routes_per_server = target_routes_per_rank * gpus_per_server
```

EP32、`4096 tokens/rank`、`topk=8` 时恰好为：

```text
target_routes_per_rank   = 4096 * 8 = 32768
target_routes_per_server = 32768 * 8 = 262144
```

不能把 `4096 tokens/rank` 直接当成 Expert FFN 的目标，因为每个 token 有 8 条 expert
routes。若总量不能整除，按 floor/ceil quota 分配；额外 quota 优先给 baseline load
更高的 server，以减少无意义迁移。

全局 negotiation 过程：

1. 同时计算所有 server 的 `surplus=max(load-target, 0)` 和
   `deficit=max(target-load, 0)`。
2. 把所有 donor server 上的 `(source_server, expert)` group 按 route 数从大到小排序。
3. 优先选择最热点 expert，再选择 deficit 最大的 receiver server。
4. 一次移动不超过 donor surplus、receiver deficit、该 expert 剩余 routes 和
   `route_chunk_tokens`。
5. 对同一个 `(expert, destination_server)`，优先复用已经打开的 remote replica，
   填满该 receiver 后才为同一 expert 打开新的 destination。
6. 更新全体 surplus/deficit，直到所有 server 达到 floor/ceil raw-route quota。
7. 以 `sum_e ceil(routes[s,e]/token_padding)*token_padding` 作为 server compute proxy，
   只接受能以字典序降低 `(global padded max, global padded spread)` 的少量跨机
   refinement。
8. 全局 server mapping 确定后，才进入第二阶段 server 内 padding-aware
   capacity packing。

优先搬热点 expert 的原因不是让热点获得更高优先级，而是一个完整 expert 权重副本
可以承接更多 routes，从而倾向于用更少的跨机 replica 和权重字节完成相同计算均衡。
这是低复杂度启发式，不宣称得到全局最少 replica 数。
服务器间 planner 仍完全不读取 NIC load、NIC budget、RDMA bandwidth 或 weight
bytes；网络只在下一步决定 planned intents 中哪些本轮可接纳。

refinement 不会为了“把 RDMA 用满”而迁移无效 expert。当网络比计算快时，
它可以接纳更多能降低全局最慢计算的 intents；但若 raw/padded server load 已在
工程容差内，额外权重传输只会浪费窗口，不应强行生成。

一个 intent 至少包含：

```text
intent_id
priority
expert_id
source_server
destination_server
moved_route_keys / moved_route_count
compute_us_before
compute_us_after
```

同一个 `(expert_id, destination_server)` 的多个 route moves 合并为一个 remote
replica，只需要复制一次完整权重。

### 4.3 规划与接纳分离

第一阶段按热点 expert 优先级输出迁移 intents。此时不考虑网络。

NIC controller 随后按顺序尝试接纳 intents：

- 相对当前已接纳 mapping，candidate 必须严格降低
  `(global server padded max, global padded spread)`；网络有空闲不是发送无计算
  收益权重的理由；
- 若 remote replica 已存在/已在本轮接纳，新增 routes 不产生额外权重通信；
- 若需要新 replica，只有全部 weight chunks 能放入本轮 endpoint budgets 才接纳；
- 无法容纳完整权重时，整个 intent 延后，routes 本轮仍留在原 server；
- 不允许只传一部分权重就执行 remote Expert FFN。

当前研究关闭显存容量约束，并删除全局共享的 `max_remote_replicas` admission cap。
不同服务器拥有独立 NIC 资源，server 0->1 建立 replica 不能消耗 server 2->3 的共享
计数。intent 不会因人为 slot/replica-count cap 被拒绝；接纳只由 source server 各
rail 的 TX `Bhard`、destination server 各 rail 的 RX `Bhard`，以及完整权重能否在
本轮传完约束。

因此 admission 在保留 planner 计算优先级的同时，只接纳对当前 prefix 仍有计算
收益且网络可行的子集。`admitted == planned` 只说明网络未拒绝 planner 的
计算改善方案；是否已经尽量
均衡必须继续检查 admitted server padded load 和最终 `max_rank_compute_us`，不能看
intent 数判断。

## 5. 第二阶段：服务器内计算均衡

NIC admission 确定最终 route-to-server mapping 后，在每个 server 内对
`token_padding` 计算块运行 capacity-aware greedy packing：

```text
server 内的 (expert, padding block)
  -> 按 expert load 从大到小
  -> 先为每个 rank 设置 floor/ceil padded-block 目标容量
  -> 优先填充剩余容量最大的 rank，tie 时优先 expert home rank
  -> 一个 rank 一次承接尽可能多的同 expert blocks
  -> 必要时建立 server-local replica
```

目标是：

```text
minimize global max_rank_compute_us
subject to the admitted server mapping
```

实现上每个 server 可以独立执行这个 packing，因为 server mapping 已由第一阶段全局
确定；但算法目标仍是所有 server 中的最慢 GPU，不是四个互不相关的局部目标。
它不强制每张 GPU 的 real routes 完全相等；在 FFN padding 语义下，padded compute
更接近才是正确的工程均衡。同时它不会把每个 expert 的每个 padding block 独立
喷到所有 GPU；尽量连续填充同一 rank，避免为追求完全均衡产生上千个无效
NVLink 权重副本。

### 5.1 EP32 两层完整实验证据

`run_20260809_160653_947420_probeep_2layer_ratio_full` 的 Layer 1 MoE-chain 说明为什么
不能用 intent 数判断均衡程度。旧版同样接纳 4 个 remote replicas，但只以 raw route
quota 为终点；新版仍为 4 个 replicas/336 MiB RDMA weight，却重新调整了它们承接的
route 数：

| 指标 | 旧 raw-quota planner | 新 server-first padding-aware planner |
|---|---:|---:|
| server raw routes | 262144/262144/262144/262144 | 262174/262213/262276/261913 |
| server padded routes | 267264/267648/266752/268416 | 266112/266112/266112/266112 |
| server padded spread | 1664 | 0 |
| 32-GPU Expert FFN min-max | 3452.88-3600.67 us | 3479.75-3493.19 us |
| 32-GPU compute spread | 147.79 us | 13.44 us（1 padding block） |
| remote replicas / RDMA weight | 4 / 336 MiB | 4 / 336 MiB |
| local prefetch count | 145 | 226 |
| Weight+Dispatch N/C | 0.7164 | 0.7261 |

这是有意识的工程取舍：允许 raw routes 对理论 quota 有不超过约 0.1% 的小幅
偏移，换取全局 padded compute 更接近；利用高带宽 NVLink 多建立一部分本地副本，
但不增加跨机权重字节。最终 MoE Weight+Dispatch 仍只占计算窗口的 0.7261，
通信未变成新瓶颈。此时继续为“用满网络”增加 remote replicas 没有计算收益，
不属于 ProbeEP 的优化目标。

对于 home server，expert master rank 是原始 placement rank。对于 remote server，
第二阶段选择一个 seed rank 接收跨机完整权重；同一 remote expert 若需要在该 server
多个 rank 执行，先由 seed rank 通过 NVLink 复制到其余 local replicas。默认 NVLink
不是瓶颈，但这些 local flow 仍显式进入 DAG 和字节统计。

第二阶段输出：

```text
execution_rank[route]
remote_seed_rank[expert, server]
local_replica_source/destination
real_routes_by_rank
padded_routes_by_rank
compute_us_by_rank
```

## 6. Weight Chunks

算法不得编码固定 expert 大小。每次 invocation 必须从模型规格和 weight dtype 得到：

```text
expert_weight_bytes = invocation.expert_weight_bytes
weight_chunks_per_expert
  = ceil(expert_weight_bytes / weight_chunk_bytes)
last_chunk_bytes
  = expert_weight_bytes - (chunk_count - 1) * weight_chunk_bytes
```

当前 DSV3 BF16 profile 只是一个数值示例：

```text
expert_weight_bytes
  = 3 * hidden * ffn_hidden * bytes(BF16)
  = 3 * 7168 * 2048 * 2
  = 88080384 bytes
  = 84 MiB
```

默认：

```text
weight_chunk_bytes = 4 MiB = 4194304 bytes
weight_chunks_per_expert = ceil(84 MiB / 4 MiB) = 21
```

Weight chunk 只表示网络分片，不表示 expert tensor parallel。remote seed rank 必须
等待该 invocation 的全部 `weight_chunks_per_expert` 到齐，才能执行该 replica 的任何
routes。模型 shape、expert 结构或 weight dtype 改变时，weight bytes、chunk 数和最后
一个 chunk 大小必须同步变化；84 MiB/21 chunks 不能作为通用常量进入 planner。

## 7. 每 Rank 数据与全局 Barrier 控制状态

每个 rank 管理一张 GPU 和对应的 rail NIC。控制器为 rank `i` 维护：

```text
A_i(t) = Attention compute time
E_i(t) = MoE Expert FFN compute time
K(t)   = Dispatch observation t 的 dispatch_overlap_compute_kind
Cmax(t)= max_i(A_i) 或 max_i(E_i)，由 K(t) 选择
N_i(t) = observation t 中完整 Weight+Dispatch TX/RX stage elapsed time
Nmax(t)= max_i(N_i)
U_i(t) = Weight RDMA+Dispatch fabric 的 NIC active-time union，仅作诊断
Dtx_i(t), Drx_i(t) = Dispatch TX/RX bytes 基线
Mtx_i(t), Mrx_i(t) = 实际接纳的 Expert Weight TX/RX bytes
P_i(t) = max(Dtx_i+Mtx_i, Drx_i+Mrx_i)，实际 endpoint byte footprint
NICmax(t)   = 理论线速推导的总 NIC bytes 硬上限
Bhard_i(t)  = 扣除 Dispatch baseline 后的 migration bytes 硬上限
B_i(t)      = 当前实际生效并保存的 migration bytes budget
```

一个 Dispatch overlap observation 完成后 AllGather：

```text
(A_i, E_i, K, N_i, U_i, dispatch_baseline_i, migration_tx_i, migration_rx_i)
for every rank
```

EP32 只有几十个标量，payload 开销可忽略；同步 latency 不是严格为零，正式实验需要
作为 control-plane overhead 在论文实现中单独实测，不进入当前 DAG。

比较的 `Cmax` 和 `Nmax` 必须来自同一个 release/deadline 窗口，
不能拿整轮任意两个不对齐的区间直接比较。一个 rank 较早完成不能让
全局 barrier 提前释放；因此 `N_i` 仅用于定位最慢 endpoint，窗口增减
方向和比例只能由全局 `beta*Cmax/Nmax` 决定，不得为每个 rank
单独计算 `Cmax/N_i`。

## 8. 理论线速硬上限

### 8.1 使用单张 NIC 的速率

当前 EP32/1-plane 拓扑中，每个 rank 对应一张 400 Gbps RDMA NIC：

```text
Rnic = 400 Gbps = 50 GB/s = 50000 bytes/us
```

`B_i` 是每张 NIC 的预算，因此必须使用单 NIC 的 `400 Gbps`。一台服务器的 8 张 NIC
合计 3.2 Tbps 已经由 8 个独立预算和 chunk water-filling 并行体现；若给每个 `B_i`
使用 3.2 Tbps，会把服务器能力再重复计算 8 次。

当前不能因为拓扑支持多 plane 就直接把 `Rnic` 乘 plane 数。未来若一个 rank 确实有
多张独立 scale-out NIC，控制状态应扩展为 `(rank, plane/NIC)`；只有定义成一个可完全
stripe 的聚合 endpoint 后，才能使用聚合速率。

### 8.2 从上次计算时间得到最大迁移字节

硬上限直接使用上一个已完成、且与当前通信类型对应的跨 GPU 最大计算时间：

```text
Cmax(t) = previous Amax(t) or previous Emax(t)
NICmax(t) = floor(Rnic * Cmax(t))
Bhard_tx_i(t) = max(0, NICmax(t) - dispatch_tx_bytes_i)
Bhard_rx_i(t) = max(0, NICmax(t) - dispatch_rx_bytes_i)
Bhard_i(t) = min(Bhard_tx_i(t), Bhard_rx_i(t))
           = max(0, NICmax(t) - max(dispatch_tx_bytes_i, dispatch_rx_bytes_i))
```

`NICmax` 是该时间窗口内每张 NIC 的理论总传输字节上限。ProbeEP controller 只控制
Dispatch 之前的额外 expert-weight migration，因此必须先减去同一个窗口内不可避免
的 DeepEP Dispatch baseline；如果直接把 `NICmax` 全部给权重迁移，权重和 token
的总时间仍可能超过计算窗口。

Combine 位于 Expert FFN 之后，是另一段 communication phase。它必须在 timeline 和
链路负载中完整仿真、单独报告，但不能加入本次 `weight -> dispatch` 窗口的
`dispatch_baseline_i`，也不能把 Combine task duration 直接累加进 `N_i`。如果前一
Combine 尚占用同一 endpoint communication stream，导致本次 Weight+Dispatch 延迟释放，
这段真实 queue wait 会自然出现在完整 stage elapsed 中，并应作为 prior-phase wait 单独
诊断。当前闭环反馈的 `N_i` 固定为每 rank 从配对 compute start 到完整 Weight+Dispatch
TX/RX 都完成的 elapsed time；旧的
`expert_weight_rdma + dispatch_fabric` active union 改名为 `U_i`，只作辅助统计。

例如 `Cmax=2000 us` 时，单 NIC 理论硬上限为：

```text
50000 bytes/us * 2000 us
= 100000000 bytes
= 95.4 MiB
```

若该 NIC 已知 Dispatch baseline 为 10 MiB，则权重迁移上限约为 85.4 MiB，而不是使用
服务器聚合 3.2 Tbps 再计算一次。

对本轮实际接纳的迁移，定义：

```text
Tnetwork_theory = max_i(
  (dispatch_tx_i + assigned_migration_tx_i) / Rnic,
  (dispatch_rx_i + assigned_migration_rx_i) / Rnic
)
```

只要 chunk scheduler 分别保证 TX/RX migration 不超过对应方向的 Bhard，就有：

```text
Tnetwork_theory <= Cmax
```

实际 admission 还显式检查方向：source relay 使用
`assigned_migration_tx_i <= min(B_i,Bhard_tx_i)`，destination relay 使用
`assigned_migration_rx_i <= min(B_i,Bhard_rx_i)`。这是每个 server/rail 独立的发送与
接收约束；不能用一个全局 replica 数，也不能只校验 source 或 destination 一端。
每接纳一个 intent 后 Dispatch layout 可能改变，因此必须用新的 layout 重新验证已经
接纳的 TX/RX bytes，不能让后续迁移使早先 chunk 间接越界。

这才是“理论网卡速率乘以上一计算阶段时间”的正确量纲。若从 bytes/tokens 计算时间，
应当是 `bytes / rate`；若从时间计算最大 bytes，才是 `rate * time`。如果以 token 数
表示本窗口 NIC 负载，还必须先乘 Dispatch 的 bytes/token；Combine 使用自己的后续
通信窗口单独核算。

该式保证的是无拥塞、满线速串行模型中的理论上限，不保证 HTSim 的真实 FCT。带宽
打不满不会自动减少已经调度的 bytes，而会使这些 bytes 传得更久；交换机排队、
incast、协议控制和 chunk 离散化也会使实际时间高于理论值。实际偏差由下一节的
实测比例控制器直接修正。

## 9. 基于实测比例的 Probe 字节窗口

每个 Dispatch control observation 记录：

```text
Cmax(t) = Amax(t) 或 Emax(t)
N_i(t) = rank i 的完整 Weight+Dispatch TX/RX stage elapsed time
Nmax(t) = max_i(N_i(t))
U_i(t) = rank i 的 Weight RDMA+Dispatch fabric active time（辅助指标）
D_i(t) = NIC i 的 Dispatch baseline bytes
Mtx_i(t) = NIC i 的 Expert Weight RDMA TX bytes
Mrx_i(t) = NIC i 的 Expert Weight RDMA RX bytes
M_i(t) = max(Mtx_i(t), Mrx_i(t))
```

`M_i` 是单个 full-duplex endpoint 的控制 footprint，不是线上总字节。使用
`max(TX,RX)` 是因为同一 NIC 的两个方向可以并行；跨 rank 对 `M_i` 求和会同时计入
source TX 和 destination RX，通常把一次 RDMA 传输计算两遍。实际跨机 expert-weight
字节固定使用 `sum_i Mtx_i = sum_i Mrx_i = remote_weight_rdma_bytes`。日志必须同时
保留 TX、RX 和 endpoint 三个数组，并做字节守恒校验。

工程目标不是令通信时间严格等于计算时间，而是保留 10% 余量：

```text
target_overlap_ratio = beta = 0.90
Ttarget = beta * Cmax
```

全局 barrier 的控制比例为：

```text
scale(t) = beta * Cmax(t) / Nmax(t)
```

同一 Attention 或 MoE 状态链的所有 NIC 使用同一个 `scale`。若
`Nmax=0.5*Cmax`，则 `scale=1.8`；若 `Nmax=2*Cmax`，则
`scale=0.45`。这保证增窗/减窗方向与全局 barrier 一致，不会因为某个
较快 rank 的局部 `N_i` 就给它单独增窗。

全局比例决定下一次允许的总通信字节，但不能缩放“配置预算”。预算只是 expert
migration admission 上限，可能没有被实际用满；token Dispatch 则无论预算大小都必须
传输。先找到产生 `Nmax` 的瓶颈 rank，并读取它实际完成的 Token+Weight endpoint bytes：

```text
P_i(t) = max(
  Dtx_i(t) + Mtx_i(t),
  Drx_i(t) + Mrx_i(t)
)

Bsample(t) = max(P_i(t) for i where N_i(t) == Nmax(t))

Bprobe_total(t+1) = floor(scale(t) * Bsample(t))
Btheory_total(t+1) = floor(Rnic * Cmax(t))
Btotal_max(t+1) = min(Bprobe_total(t+1), Btheory_total(t+1))

Bdesired_i(t+1)
  = max(0, Btotal_max(t+1) - max(Dtx_i(t+1), Drx_i(t+1)))

Bnext_i(t+1)
  = min(Bdesired_i(t+1), Bhard_i(t+1))
```

`Btotal_max` 是所有 NIC 共用的总通信字节上限；各 NIC 的 token baseline 不同，因此
扣除 token 后得到的 expert migration budget 可以略有不同。`B_i(t)` 只允许压制或
放宽 Expert Weight，不得裁剪、延后到下一 invocation 或改变 token Dispatch bytes。
若 token baseline 本身已经超过 `Btotal_max`，该 NIC 的 migration budget 为 0，token
仍完整传输，实际 `Nmax` 超过目标的事实留给下一次 observation 继续反馈。

`N_i`、方向 bytes 和 endpoint footprint 全部保留，用于定位瓶颈、计算理论上限和
校验字节守恒；所有 rank 使用同一个全局 `scale` 和 `Btotal_max`，不根据局部 `N_i`
独立决定增减方向。只要存在非零 Token+Weight observation，每次都更新窗口；是否有
deferred intent 不影响反馈，因为下一层 Gate 可能重新产生迁移需求。只有没有任何可测
通信字节或时间无效时才保持当前 budget。

不再增加 `0.5-2.0` 人为调节限幅。`beta=0.90` 已为测量误差和排队保留 10%
余量，比例反算结果只经过 `Bhard` 物理硬上限 clamp。这样 `Nmax/Cmax` 偏离较大时，
下一次同类 Dispatch observation 可以一步接近目标，而不是被固定步幅拖慢。若测量
无效或没有任何实际通信字节时保持当前 budget。

### 9.1 H20 两层旧实验核算与修复口径

`run_20260809_210222_024276_probeep_2layer_dynamic_full` 的第一次 Attention-chain
observation 为：

```text
Cmax                 = 14626.104846 us
Nmax                 = 2447.3 us
beta * Cmax          = 13163.494361 us
scale                = 5.378782
旧 budget            = 16 MiB/NIC
NICmax               = 731305242 bytes = 697.43 MiB/NIC
Bhard                = 616.46-619.83 MiB/NIC
```

该历史 run 由修复前代码生成，当时错误执行
`16 MiB * scale = 86.06 MiB/NIC`。它缩放了未必用满的配置预算，并遗漏 token
Dispatch 已经贡献的约 78-81 MiB endpoint bytes。按修复后的全局瓶颈字节公式核算：

```text
Bsample              = 88.32 MiB（Nmax ranks 中较大的 full-duplex endpoint 总字节）
Bprobe_total         = 475.07 MiB/NIC
Btotal_max           = min(475.07, 697.43) = 475.07 MiB/NIC
migration budget     = 394.10-397.48 MiB/NIC（扣除各 NIC token baseline）
```

本次两层实验中 `86.06 MiB/NIC` 仍足以接纳最后一层的 4/4 planned replicas，因而
最后一层 server real routes 为 `262174/262213/262276/261913`，不影响该次 placement
已接近均值的事实；但 `86.06 MiB/NIC` 只能作为旧错误实现的观测值，不能写成设计
结论。修复后必须重新运行同一 H20 full case，以新 observation 中的
`bottleneck_observed_total_bytes/probed_total_nic_max_bytes/effective_total_nic_max_bytes`
作为验收证据。

修复后的完整实验为
`run_20260809_225300_474846_probeep_2layer_dynamic_full`。四次 observation 为：

| ID | 窗口 | `Cmax` us | `Nmax` us | `alpha*k` | `Bsample` MiB | `Bprobe_total` MiB | `Btheory_total` MiB | 下一 migration budget 均值 MiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | L0/MB0 Attention | 14626.10 | 2447.30 | 5.379 | 88.32 | 475.07 | 697.43 | 395.92 |
| 1 | L0/MB1 MoE | 34729.12 | 2251.90 | 13.880 | 87.82 | 1218.99 | 1656.01 | 1139.77 |
| 2 | L1/MB0 Attention | 14626.10 | 13659.00 | 0.964 | 90.23 | 86.96 | 697.43 | 8.28 |
| 3 | L1/MB1 MoE | 26635.90 | 13682.00 | 1.752 | 90.05 | 157.77 | 1270.10 | 79.16 |

ID 2 说明通信超过 `0.9*Cmax` 时会压缩 Expert Weight 空间；ID 3 即使当前没有
deferred intent 也继续按实测字节更新，避免陈旧的大 budget 污染下一层。四次
`dispatch_baseline_bytes_by_rank` 在更新前后完全不变，controller 只改变 Expert Weight
admission 上限。

### 9.2 Attention/MoE 独立状态

Attention 和 MoE 的 `Cmax` 差异很大，不能共享一套 budget。控制器固定维护：

```text
B_attention[i]
B_moe[i]
```

一个 Attention-chain Dispatch observation 只更新 `B_attention`，一个 MoE-chain
Dispatch observation 只更新 `B_moe`。Combine observation 不更新任何 budget。
否则较短 Attention 窗口会把共享 budget 压低，较长 MoE 窗口只能从过小值重新增长，
失去比例控制的一步逼近能力。

### 9.3 收敛边界

在 workload 稳定、`Nmax` 随全局窗口单调且局部近似线性时，上述公式理想上一步到达
`beta*Cmax`。实际系统只要求工程上接近，不声称对任意动态 trace 严格收敛：

- remote expert 以完整 `invocation.expert_weight_bytes` 接纳，决策不是连续变量；
- chunk 为 4 MiB，budget/流量存在离散台阶；
- 网络拥塞和 Gate load 会随 invocation 改变；
- 双 microbatch 存在 delayed feedback；
- 完整 expert/chunk 使有效发送量可能在两个离散档位之间切换；
- 若不可删除的 Dispatch baseline 已经满足 `Nmax > beta*Cmax`，将 migration budget
  降为 0 也不能继续缩短网络时间；
- 若所有能继续降低 `(global server padded max, global max-rank compute)` 的
  compute planner intents 已接纳，继续增大 NIC budget 也不应凭空产生无效
  expert migration，网络时间可能停在 `beta*Cmax` 以下。

当前设计可以严格保证的是：

1. `B_i <= Bhard_i`，有效迁移预算不会无限增长；
2. 理论满线速总通信时间不超过 `Cmax`；
3. 每次有效测量都按全局 `beta*Cmax/Nmax` 直接修正，不依赖固定 5% 步长；
4. Attention/MoE 状态互不污染。

因此两层、双 microbatch 的 4-Dispatch-observation 专项实验用于验证“一次同类
feedback 后立即修正”和工程边界，不把第 4 个 Dispatch observation 必须精确达到
`0.90` 作为正确性条件。

正式实验把连续若干同类 Dispatch observations 的 `Nmax/Cmax` 和有效 budget 保持在
小范围内定义为
“运行时稳定”，不把它表述成对任意动态 trace 的数学必然收敛。

## 10. Chunk-to-NIC Water-Filling

一个 weight chunk 选择 rail `i` 时，同时消耗：

```text
source server rank[i] 的 TX budget
destination server rank[i] 的 RX budget
```

chunk 均衡状态按有向 server pair 独立维护：

```text
pair_load[source_server, destination_server, rail]
```

同一个 source server 发往不同 destination servers 时，不能共享一个 water-filling
游标。例如 `server0->server1` 的历史 chunk 分布不能改变 `server0->server2` 从哪条
rail 开始均衡。same-rail RDMA 使 pair 上 rail `i` 的 TX bytes 与 RX bytes 天然相等：

```text
pair_tx[s,d,i] = pair_rx[s,d,i] = pair_load[s,d,i]
```

只要每个 pair 内满足：

```text
pair_load[s,d,0] ~= ... ~= pair_load[s,d,G-1]
```

则聚合后的发送和接收也基本均衡：

```text
total_tx[s,i] = sum_d pair_load[s,d,i]
total_rx[d,i] = sum_s pair_load[s,d,i]
```

离散误差主要来自最后一个不足 chunk 和 `weight_chunk_bytes` 粒度。多个 pair 的余数可能
选择同一 rail，因此相同 pair load 时仍用物理 endpoint total 做 tie-break。该均匀性是
在对应 rail 均满足双端 budget 时的尽力目标；若某些 rail 不可行，admission 正确性优先，
manifest 必须记录实际 `max_min_spread_bytes`。

物理容量状态继续按 endpoint 总量维护：

```text
assigned_tx[source_server, rail] = sum_d pair_load[source_server,d,rail]
assigned_rx[destination_server, rail] = sum_s pair_load[s,destination_server,rail]

server_send_remaining[s] = sum_rail(B_tx[s,rail] - assigned_tx[s,rail])
server_recv_remaining[d] = sum_rail(B_rx[d,rail] - assigned_rx[d,rail])
```

`pair_load` 决定均匀 stripe；`assigned_tx/assigned_rx` 负责真实 NIC budget admission，
两者不能互相替代。服务器级 sum 只用于展示“该服务器还能发送/接收多少权重”，不能
替代逐 rail 检查。
不从剩余 bytes 派生固定的“还能迁移几个 replicas”：模型变化会改变 expert weight，
未来异构 experts 甚至没有统一除数。Dashboard 只报告剩余 TX/RX bytes 和实际
sent/received replica 数。

由于跨机权重 chunk 使用 same-rail RDMA，只有 source TX 和 destination RX 的同一个
rail 都能容纳完整 chunk 时，该 rail 才是合法候选。NIC 是 full-duplex，TX/RX 预算
分别核算，不用一个全局 replica 数把所有服务器串在一起。

`B_i` 已经从理论 wire capacity 中减过 Dispatch baseline，因此它只表示 migration bytes；
chunk scheduler 不能再从 `B_i` 重复扣一次 Dispatch/Combine bytes。但选择轻载 rail
时，仍需要把 layout 已知的去冗余 token fabric bytes 放进 score：

```text
token_tx_i / token_rx_i = 从当前 route-to-server layout 推导
migration_remaining_tx_i = B_i - assigned_migration_tx_i
migration_remaining_rx_i = B_i - assigned_migration_rx_i
```

对 `(source_server=s,destination_server=d)` 的一个大小为 `x` 的 chunk，先排除任一物理
endpoint budget 无法容纳该 chunk 的 rail，再使用确定性字典序 score：

```text
score(i) = (
  pair_load[s,d,i] + x,
  max(
    token_tx[source_i] + assigned_tx[source_i] + x,
    token_rx[destination_i] + assigned_rx[destination_i] + x
  ),
  token_tx[source_i] + assigned_tx[source_i]
    + token_rx[destination_i] + assigned_rx[destination_i]
    + 2*x,
  i
)
```

第一项使每个 server pair 独立均匀；第二、三项在 pair 负载相同时避开总体更重的 source
TX/destination RX endpoint；最后按 rail id 稳定打破平局。每放一个 chunk 必须同时更新
`pair_load`、`assigned_tx` 和 `assigned_rx`。token bytes 不影响 expert 迁移 intent 的
生成和优先级，只参与 endpoint tie-break 和 budget hard cap。

路径显式 lowering 为：

```text
expert home/seed rank
  -> source relay[i]                 NVLink scatter
  -> destination relay[i]            same-rail RDMA
  -> remote seed rank                NVLink gather
```

NIC 选择不回头改变 expert source server、destination server、迁移优先级或服务器内
execution placement。

## 11. 收敛与展示

58 层、双 microbatch 的完整实验展示分为两个区域。

### 11.1 前 N 次 Dispatch 观测

逐 Dispatch observation 展示控制过程：

```text
dispatch_observation_id / invocation_index / layer / microbatch
communication_kind=dispatch
dispatch_overlap_compute_kind / Amax / Emax / Cmax
Nmax / Nmax-Cmax / global scale
每 rank A_i / E_i / N_i / dispatch_baseline_i / Bhard_i / B_i
planned/admitted/deferred intents
每 NIC migration bytes
remote replica 数和 moved routes
```

观察 budget 的 decrease/increase、bottleneck NIC 转移和 compute imbalance 改善。

### 11.2 收敛后

从满足以下条件的连续同类 Dispatch observations 中选择稳定区间：

```text
abs(Nmax - Cmax) / max(Cmax, epsilon) <= convergence_tolerance
budget change <= budget_tolerance
连续 convergence_observations 次成立
```

展示稳定区间的均值/P95：

```text
Amax, Emax, Cmax, Nmax, makespan
compute imbalance
per-NIC Bhard/budget/utilization
admitted migration bytes
remote replica 数
```

当前 2 层测试只有 4 个 Dispatch control observations，不做“收敛前/后”性能结论，
只检查 observation 编号、A/M 状态链、Combine 排除和 DAG 依赖完整。

## 12. 单 HTSim 动态 DAG 闭环

ProbeEP 不再使用 external baseline/replay。Python controller 和 HTSim 通过
`-dag_control` 的增量协议同步，整个实验只有一个 HTSim PID、一条
EventList 时间线和一份最终 task map。

2-microbatch 的跨层 wavefront 按以下单元追加：

```text
append Layer L body，暂留 Layer L / MB1 Final
  -> observe MB0 Weight+Dispatch terminals
  -> update Attention controller state, CONTINUE
  -> observe MB1 Weight+Dispatch terminals
  -> update MoE controller state
  -> atomically append:
       Layer L / MB1 Final
       + Layer L+1 body using both latest states
```

中间层暂留 MB1 Final，是因为它在 compute stream 上必须排在下一层
MB0 Attention/Router 之后。如果提前提交，要么后续非法回写它的前驱，
要么丢失跨层 overlap。暂留任务尚未交给 HTSim，因此与下一层同批
原子提交不会修改旧 DAG。

最后一层不再需要跨层插入，因此其 MB1 Final 在该层 body 中直接
提交。最后一层的 MB1 observation 后发送 `DAG_CLOSE`，然后让已提交的
Expert/Combine/Reduce 自然 drain。下一层在上一层 MB1 Dispatch 完成后已经
提交，因此 Layer L+1/MB0 Attention 仍可与 Layer L/MB1 Combine overlap；
不需要等整层 drain。

observation 本身只是一组 barrier terminal 的 join，不是伪 compute task。
HTSim 在该模拟时刻同步等待 controller，Python planner 的 wall-clock
时间不进入 makespan。controller 从同一进程已输出的 task start/done
事件计算 per-rank `Weight+Dispatch` elapsed，然后更新对应的
Attention/MoE 状态链。Combine 仍只作 telemetry。

动态 builder 只能给新任务加前驱，不得修改已 append 任务。它持久保存：

- 每 rank compute stream tail；
- 每 rank communication stream tail；
- 上一层两个 microbatch 的 final terminal；
- Attention/MoE 两类 NIC budget state；
- 全局 task/barrier/observation 单调 ID 分配器。

Python emitter 和 HTSim manager 都按 batch 原子提交。batch/task/barrier/observation ID
在一次进程内均不可重复；任何 observation terminal、依赖或 ID 校验失败时，不能消耗
新 ID、修改累计 `workload.dag` 或留下半批 task。

静态 `.dag` 仍用于 NCCL/DeepEP/EPLB/MoonEP 和 ProbeEP 非闭环功能测试；
只有需要消费实测 FCT 的 ProbeEP 实验必须走动态协议。

## 13. 复杂度

Gate 若直接提供 `(server, expert)` route histogram，在线控制路径为：

```text
server raw quota:        O(E log E + M)
padding refinement:      O(I * S^2 * E)
server-local packing:    O(E log E + B * G)
controller update:       O(P)
chunk scheduling:        O(weight_chunk_count * G)
```

`S` 是 server 数，`M` 是 quota 产生的 server-expert moves，`I` 是实际接受的
padding refinement 步数，`B` 是 server-local expert padding blocks。实现为 `I`
设置 `S*E` 硬上限，并且每步必须严格降低 `(global padded max, spread)`，因此不会
循环震荡。EP32/DSV3 当前 BF16 profile 中 `E=256`、`P=32`、`G=8`，每 replica 示例为 21
chunks；通用复杂度使用 `ceil(expert_weight_bytes/weight_chunk_bytes)`。在线 planner 不扫描、
保存或排序全部百万 routes。若输入只有 expanded assignments，则先聚合 histogram
需要一次 `O(R)`；仿真生成器把最终 moved counts/execution quotas 展开回具体 route
keys 还需要 `O(R)`。这两个线性 pass 是输入处理/仿真 lowering，不是 greedy 搜索
复杂度，也不作为 DAG task。

## 14. 配置草案

```yaml
algorithm: probeep
planning_mode: oracle_current_routes
planner_runtime_model: not_in_dag
route_lowering_runtime_model: offline_not_in_dag

compute_planner:
  inter_server: global_quota_then_padding_refinement
  intra_server: padding_block_capacity_packing
  route_chunk_tokens: 4096
  token_padding: 128

nic_controller:
  feedback_source: in_process_dynamic_dispatch_observation
  compute_windows: [attention, moe]
  nic_line_rate_gbps: 400
  initial_budget_bytes: 16777216
  target_overlap_ratio: 0.90
  adjustment: global_bottleneck_actual_bytes_times_measured_ratio
  hard_limit: 400Gbps_times_compute_minus_dispatch
  state_partition: per_compute_window

weight_transport:
  chunk_bytes: 4194304
  nic_assignment: dispatch_baseline_plus_migration_waterfill
  require_complete_replica: true
  scale_out_transport: same_rail_rdma
  scale_up_transport: server_local_fullmesh

token_transport: deepep_hierarchical
```

当前不建模 HBM 容量、expert 生命周期或 replica-count 上限，因此代码和配置中不保留
虚假的 slot/capacity 字段。每个 remote replica 只按实际 expert weight bytes 消耗逐 NIC
迁移预算；若以后研究显存限制，需要另行建立按 layer、dtype、optimizer state 和 replica
lifetime 计算的容量模型。

`nic_line_rate_gbps=400` 是单张 rank NIC 的线速，不是服务器 8 NIC 的聚合速率。
`NICmax=Rnic*Cmax` 只约束满线速理论时间；实际网络使用
`target_overlap_ratio*Cmax/Nmax` 的全局 barrier 实测比例缩放瓶颈 NIC 当前实际
Token+Weight 总字节。当前实现不使用固定 `max_budget_bytes`；probe total 先由
`NICmax` 封顶，再扣除各 NIC token baseline 得到 Expert Weight budget。

## 15. Manifest

每个 MoE invocation 至少记录：

```text
invocation_index, layer, microbatch
planner_runtime_model, route_lowering_runtime_model
controlled_communication_phase=dispatch
combine_controller_role=telemetry_only
dispatch_observation_id
dispatch_overlap_compute_kind
attention_compute_us_by_rank, moe_compute_us_by_rank
attention_max_us, moe_max_us, selected_compute_ref_us
server_target_routes
server_load_before/planned/admitted, donor_surplus, receiver_deficit
server_padded_load_before/planned/admitted
rank_load_before/after
global max/min rank compute before/after
planned/admitted/deferred migration intents
remote replica seed/local replicas
remote_replicas_sent/received_by_server
remote_weight_bytes_sent/received_by_server
weight chunks and rail assignments
assigned migration bytes by directed server pair and rail
predicted Dispatch TX/RX bytes by rank
dispatch_tx_bytes_by_rank, dispatch_rx_bytes_by_rank
nic_line_rate_gbps, nic_theoretical_max_bytes
observed_total_bytes_by_rank
bottleneck_observed_total_bytes
probed_total_nic_max_bytes, effective_total_nic_max_bytes
hard_migration_cap_by_rank
budget_before/unclamped_target_budget_by_rank/budget_after
global_network_to_compute_ratio, global_adjustment_factor, target_overlap_ratio
per-rank diagnostic effective_rate_bytes_per_us, adjustment_factor
Dispatch feedback A_i/E_i/K/N_i/Nmax/source
migration_tx_bytes_by_rank, migration_rx_bytes_by_rank
migration_endpoint_bytes_by_rank=max(TX,RX)
remote_weight_rdma_bytes, local_weight_bytes
controller action and bottleneck ranks
dispatch/combine hierarchy bytes
```

固定 Gate assignment、固定 controller state 和固定配置必须得到确定性的 migration plan、
admission 和 DAG。Python 生成器 wall-clock 不进入 manifest 性能口径，也不参与
确定性 plan tie-break。

## 16. 当前不做

- backward replica gradient reduction 和 optimizer state；
- 未完整权重上的部分 Expert FFN；
- 跨 invocation 传一半权重并缓存到下一轮；
- 动态 SM/HBM/NVLink 竞争；
- HBM 容量、跨 invocation replica lifetime 和权重缓存回收；
- 多 plane 联合窗口控制；
- 在静态 DAG 内伪造 HTSim 运行时反馈。

这些边界保留了核心问题：用低复杂度 histogram greedy 先规划计算均衡，再用类似 cwnd 的
per-rank 字节窗口逐 Dispatch observation 探测可被计算掩盖的 RDMA 迁移量，最后由轻载 NIC 承担
更多完整 expert weight chunks。

## 17. 功能开发表

状态列是唯一功能核查口径；只有实现、文档和对应自动化检查一致时才标记 `√`。空白项表示
尚未完成或本轮尚未核查，不能解释为已支持。

| ID | 功能 | 代码入口或边界 | 状态 | 核查说明 |
|---|---|---|:---:|---|
| F01 | DeepEP token 去冗余与分层 Dispatch/Combine | `build_hierarchical_dispatch/combine` | √ | `probeep_cross_server` 与 HTSim ProbeEP case 通过 |
| F02 | 全部服务器参与的 raw quota，再按 padded compute 做跨机 refinement | `_plan_inter_server/_refine_inter_server_padding` | √ | global quota 与 padding 反例都通过 |
| F03 | admitted server mapping 上的 padding-aware 服务器内 capacity packing | `ProbeEPBuilder._plan_intra_server` | √ | 每 server GPU padded spread <= 1 block，同 expert 尽量连续装箱 |
| F04 | 计算规划与网络 admission 分离 | `ProbeEPBuilder.plan/_admit_intents` | √ | planner 输入与 admission 断言通过 |
| F05 | 单 NIC 400 Gbps 理论硬上限与 Dispatch baseline 扣减 | `nic_theoretical_max_bytes`、`hard_migration_cap_by_rank` | √ | `probeep_controller` 与双端 budget 断言通过 |
| F06 | Attention/MoE 两套独立 per-rank 预算状态 | `ProbeNICController._budgets_by_kind` | √ | `probeep_controller` 通过 |
| F07 | 基于全局 `Cmax/Nmax`、瓶颈 NIC 实际 Token+Weight bytes 和 0.90 目标的统一窗口更新 | `ProbeNICController.update` | √ | `Bprobe=alpha*Cmax/Nmax*Bsample`，再由 400 Gbps 封顶并扣 Token baseline；30/30 与 H20 full 通过 |
| F08 | 完整 expert weight 分 chunk，完整到达后才执行 Expert FFN | `_schedule_weight_chunks` 与 weight task 依赖 | √ | 权重字节守恒、FFN 前驱和 HTSim 完成检查通过 |
| F09 | 按有向 `(src_server,dst_server)` 独立维护 per-rail pair load | `_schedule_weight_chunks` | √ | `probeep_pair_aware_chunks` 通过 |
| F10 | 每个 chunk 同时检查 source TX 和 destination RX 预算 | `_schedule_weight_chunks` | √ | endpoint totals 与逐 rank budget 断言通过 |
| F11 | 同一 source rail 的 remote Weight TX 完成后立即释放 Dispatch TX | ProbeEP task edge + 合并后的 `weight_dispatch` communication stage | √ | `probeep_cross_server` 检查依赖边；H20 动态 full 完成 240 条 Weight->Dispatch 顺序检查 |
| F12 | Combine 只做网络仿真和 telemetry | Transformer ProbeEP observation wiring | √ | `probeep_dispatch_observations` 通过 |
| F13 | 增量 builder 将当前 Dispatch budget 注入新层 | `probeep_nic_budget_by_dispatch` | √ | 零预算注入的 admission 断言通过 |
| F14 | 同一次 HTSim 内读取 FCT 并动态追加后续 invocation | `-dag_control` + incremental driver | √ | 2-layer full 为单 PID、2 append、4 observations、1 summary；64 个跨层 Attention release gap 均为 0 |
| F15 | HBM 容量、replica 生命周期与回收 | 当前不做 |  | 源码不保留虚假 slot/capacity 参数 |
| F16 | 多 plane 联合窗口控制 | 当前不做 |  | 当前 ProbeEP 控制口径为 plane=1 |

## 18. 测试表

只有本轮实际执行并通过的测试才标记 `√`；历史实验不能替代当前代码的回归结果。

| ID | 测试 | 覆盖范围 | 状态 | 最近核查证据 |
|---|---|---|:---:|---|
| T01 | `probeep_cross_server` | 跨机规划、权重 chunks、DAG 依赖、字节守恒 | √ | `run_20260809_205120_765390_workload_generator` |
| T02 | `probeep_pair_aware_chunks` | server-pair 状态隔离、rail 均衡、双端字节守恒 | √ | `run_20260809_205120_765390_workload_generator` |
| T03 | `probeep_global_quota` | 全部 donor/receiver、热点优先、raw quota | √ | `run_20260809_205120_765390_workload_generator` |
| T04 | `probeep_padding_aware_global_balance` | raw load 已相等但 padded server load 不等的 refinement 反例 | √ | `run_20260809_205120_765390_workload_generator`：padded max 16 -> 12，local spread <= 1 block |
| T05 | `probeep_controller` | A/M 独立状态、full-duplex 瓶颈总字节比例更新、Token baseline 扣减、理论硬上限、零字节 hold | √ | `run_20260809_225244_921443_workload_generator`：30/30 passed；含 TX/RX 方向反例 |
| T06 | `probeep_dispatch_observations` | 2 layer x 2 microbatch 的 4 次 Dispatch 观测、合并 communication stage 与 budget 注入 | √ | `run_20260809_205120_765390_workload_generator` |
| T07 | workload generator 全量功能回归 | ProbeEP、动态 emitter/wavefront 与共享 DAG/算法基础功能 | √ | `run_20260809_205120_765390_workload_generator`，30/30 passed |
| T08 | timeline/dashboard 功能回归 | 32 GPU lanes、动态 task map、Gate/link/ZIP 产物 | √ | `run_20260809_225300_474846_probeep_2layer_dynamic_full`，Dashboard 分列 sample/probe/theory total 与 migration budget |
| T09 | 2-layer、4096 tokens/rank、EP32 ProbeEP H20 单进程完整实验 | 动态反馈、跨机 expert migration、source-rail TX 顺序、32 GPU lanes | √ | `run_20260809_225300_474846_probeep_2layer_dynamic_full`：12 replicas、240 条顺序检查、64 条零等待检查 |
| T10 | 同一次 HTSim 内动态反馈闭环 | observation FCT 驱动后续 invocation | √ | `run_20260809_225300_474846_probeep_2layer_dynamic_full`：4 observations、2 append、1 summary |
| T11 | HTSim 动态 DAG 协议最小测试 | 5 us observation 后同时刻 append 7 us task，单 PID/单 summary | √ | `run_20260809_202850_354975_dynamic_dag_functional`，makespan=12 us |
| T12 | 迁移方向字节与 endpoint 口径 | `max(TokenTX+WeightTX,TokenRX+WeightRX)`、TX/RX/RDMA 守恒 | √ | `run_20260809_225300_474846_probeep_2layer_dynamic_full`：4/4 observations 守恒 |
| T13 | 当前 telemetry schema 下的非零跨机迁移 full 验证 | 动态 FCT、sample/probe/theory total、非零 TX/RX/RDMA、Dashboard/ZIP | √ | `run_20260809_225300_474846_probeep_2layer_dynamic_full`，H20 theoretical profile |
| T14 | Directed load 柱状图 | 首层 raw/末层 admitted、末层 server-pair/per-NIC Dispatch、Combine、Expert Weight | √ | `run_20260809_224833_908121_gate_workload_visualization`，4/4 passed；同口径进入 full ZIP |
