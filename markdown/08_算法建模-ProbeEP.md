# 算法建模：ProbeEP 两阶段专家迁移与反馈式 NIC 窗口

## 1. 定义

ProbeEP 是一个面向训练/prefill forward 的跨服务器临时 expert replica 算法。
它由三个职责严格分离的模块组成：

```text
Compute Migration Planner
  只根据 Expert FFN 计算负载规划迁移

NIC Budget Controller
  根据上一采样的 compute/NIC 时间维护每张 NIC 的迁移字节窗口

Weight Chunk Scheduler
  在窗口内把完整 expert 的权重 chunks 分给轻载 NIC
```

核心优化目标只有计算均衡：

```text
minimize max_rank(expert_compute_us)
```

网络不是 expert source/destination 规划的目标或约束。网络反馈只控制“本轮允许接纳
多少迁移”，NIC 负载只影响已接纳 expert 的 weight chunks 使用哪些 rail。

ProbeEP 是独立算法。`ProbeEPBuilder` 不导入、继承或调用 `MoonEPBuilder`；它在自己
的代码中实现服务器间和服务器内两阶段 planner，二者可以独立演进。Token
dispatch/combine 继续使用共享的 DeepEP hierarchy 数据面。

## 2. 一次采样

本文把一次完整 MoE invocation 定义为一个 sample：

```text
Dispatch -> Expert FFN -> Combine
```

Router/Gate 产生本次 sample 的 expert route histogram，迁移决策直接读取 histogram。
目标闭环中，每次决策使用“此时已经完成的最新 sample”更新 NIC 窗口，和
congestion window 的反馈时序一致。当前实现只把决策结果降低为静态 DAG，不把
planner 的 Python/CUDA/C++ 实现开销放进运行时 DAG。

对于两个 microbatch 和 `L` 个 MoE layer：

```text
sample_count = 2 * L
```

完整 58 层实验有：

```text
2 microbatch * 58 layers = 116 samples
```

这 116 次是 116 个观测/更新机会。双 microbatch 会有两个 invocation 在 pipeline 中，
后一个 planner 启动时，前一个 Dispatch/Expert/Combine 可能尚未全部结束。因此闭环
driver 必须建模 delayed feedback：使用启动时最新的已完成反馈，不能强行假设编号
`t` 必然直接更新编号 `t+1`。

这足以观察窗口从初值逐步靠近计算掩盖边界。当前两层测试只有 4 个 samples，只验证
状态推进和 DAG 正确性，不宣称控制器已经收敛。

不同 layer 的 logical experts 是不同参数，但它们共享同一组 GPU/NIC。expert
migration plan 每个 sample 重新计算；每 rank NIC budget 可以跨 layer/sample 保留，
因为它学习的是共享 endpoint 的可用迁移窗口。

## 3. Planner 与 DAG 边界

当前 sample 的因果关系固定为：

```text
all Router/Gate tasks
        |                   |
        v                   v
Weight Migration       Token Dispatch
        |                   |
        +---------+---------+
                  v
              Expert FFN
                  |
                  v
                Combine
```

当前契约：

- planner 只产生 migration intents、execution placement 和 NIC chunk assignment；
- `.dag` 不生成 CPU planner task，不增加 `compute_us`、barrier 或 timeline lane；
- Router 完成后直接释放已经确定的 Weight Migration 与 Token Dispatch；
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
```

当前 DSV3 expert 同构时，route 数与 FFN 时间成正比；最终判断仍使用 padded
`compute_us`，为未来异构 expert 保留正确语义。

### 4.2 Greedy/LPT 过程

服务器间 planner 完全不读取 NIC load、NIC budget、RDMA bandwidth 或 weight bytes：

1. 找当前预测计算负载最高的 source server。
2. 找当前预测计算负载最低的 destination server。
3. 在 source server 中选择计算负载最大的可移动 expert routes。
4. 移动不超过 source surplus、destination deficit 和 `route_chunk_tokens` 的 routes。
5. 若 destination server 尚无该 expert，生成一个 remote replica intent。
6. 更新两个 server 的预测计算负载并重复。

理论目标：

```text
server_target_compute = total_expert_compute / server_count
```

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

第一阶段输出按计算收益排序的迁移 intents。此时不考虑网络。

NIC controller 随后按顺序尝试接纳 intents：

- 若 remote replica 已存在/已在本轮接纳，新增 routes 不产生额外权重通信；
- 若需要新 replica，只有全部 weight chunks 能放入本轮 endpoint budgets 才接纳；
- 无法容纳完整权重时，整个 intent 延后，routes 本轮仍留在原 server；
- 不允许只传一部分权重就执行 remote Expert FFN。

因此网络不会改变 intent 的计算优先级，只决定本轮能执行 intent 序列的哪个可行
子集。

## 5. 第二阶段：服务器内计算均衡

NIC admission 确定最终 route-to-server mapping 后，每个 server 独立运行 MoonEP-style
greedy：

```text
server 内的 routes
  -> 按 expert load 从大到小
  -> 优先放到当前 compute load 最低的 rank
  -> 必要时建立 server-local replica
```

目标是：

```text
minimize max_rank_compute_us within each server
```

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

当前 DSV3 BF16 expert 权重为：

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
weight_chunks_per_expert = 84 / 4 = 21
```

Weight chunk 只表示网络分片，不表示 expert tensor parallel。remote seed rank 必须
等待 21 个 chunks 全部到齐，才能执行该 replica 的任何 routes。

## 7. 每 Rank NIC 状态

每个 rank 管理一张 GPU 和对应的 rail NIC。控制器为 rank `i` 维护：

```text
C_i(t) = sample t 的 overlap compute window
N_i(t) = max(sample t 的 NIC TX span, NIC RX span)
B_i(t) = sample t 可用于 expert migration 的字节窗口
```

sample 完成后 AllGather：

```text
(C_i, N_i, migration_bytes_i) for every rank
```

EP32 只有几十个标量，payload 开销可忽略；同步 latency 不是严格为零，正式实验需要
作为 control-plane overhead 在论文实现中单独实测，不进入当前 DAG。

比较的 compute 和 NIC time 必须来自同一个 overlap release/deadline 窗口，不能拿
整轮任意两个不对齐的区间直接比较。

## 8. Probe/AIMD 字节窗口

全局观察量：

```text
Cmax(t) = max_i C_i(t)
Nmax(t) = max_i N_i(t)
```

每个 sample 后更新下一轮 budget。默认参数：

```text
beta = 0.9
deadband = 0.05
additive_step_bytes = 1 MiB
```

规则：

```text
if Nmax > (1 + deadband) * Cmax:
    对所有接近 Nmax 的 bottleneck NIC:
        B_i(t+1) = max(min_budget, beta * B_i(t))

elif Nmax < (1 - deadband) * Cmax and pending_migration_exists:
    对非瓶颈 NIC:
        B_i(t+1) = min(max_budget, B_i(t) + additive_step_bytes)

else:
    B_i(t+1) = B_i(t)
```

乘当前 `budget`，不直接乘实际发送字节。若某一 sample 没有可迁移 expert，实际发送
可能为 0，但这不能证明 NIC capacity 为 0。

只有最差 NIC 决定 `Nmax` 时才缩小其窗口；其他轻载 NIC 可以保持或缓慢增加。随着
samples 推进，weight chunks 会从拥塞 NIC 转向轻载 NIC，`Nmax` 逐步接近 `Cmax`。

为避免采样噪声导致振荡，首版支持 deadband；EWMA、连续多次超阈值和更复杂控制律
留给 58 层收敛实验后再决定。

## 9. Chunk-to-NIC Water-Filling

一个 weight chunk 选择 rail `i` 时，同时消耗：

```text
source server rank[i] 的 TX budget
destination server rank[i] 的 RX budget
```

budget 只约束 migration bytes，不能把 Dispatch/Combine 的 token bytes 再从该窗口扣
一次。但选择轻载 rail 时，需要把 layout 已知的去冗余 token fabric bytes 作为基线：

```text
token_tx_i / token_rx_i = 从当前 route-to-server layout 推导
migration_remaining_tx_i = B_i - assigned_migration_tx_i
migration_remaining_rx_i = B_i - assigned_migration_rx_i
```

对一个大小为 `x` 的 chunk：

```text
score(i) = max(
  token_tx[source_i] + assigned_migration_tx[source_i] + x,
  token_rx[destination_i] + assigned_migration_rx[destination_i] + x
)
```

当前各 rail 均为 400 Gbps，因此比较 bytes 等价于比较预测传输时间。在
source/destination migration budget 都能容纳完整 chunk 的 rails 中选择 score 最小
者；每放一个 chunk 就更新 migration assigned bytes，相当于确定性的 greedy
water-filling。相同 score 按双端总字节数、再按 rail id 选择。这样 token 热点 NIC
会自然少接收 weight chunks，但它的 token bytes 不影响 expert 迁移 intent 的生成和
优先级。

路径显式 lowering 为：

```text
expert home/seed rank
  -> source relay[i]                 NVLink scatter
  -> destination relay[i]            same-rail RDMA
  -> remote seed rank                NVLink gather
```

NIC 选择不回头改变 expert source server、destination server、迁移优先级或服务器内
execution placement。

## 10. 收敛与展示

58 层、双 microbatch 的完整实验展示分为两个区域。

### 10.1 前 N 次采样

逐 sample 展示控制过程：

```text
sample_id / layer / microbatch
Cmax / Nmax / Nmax-Cmax
每 rank C_i / N_i / B_i
planned/admitted/deferred intents
每 NIC migration bytes
remote replica 数和 moved routes
```

观察 budget 的 decrease/increase、bottleneck NIC 转移和 compute imbalance 改善。

### 10.2 收敛后

从满足以下条件的连续 samples 中选择稳定区间：

```text
abs(Nmax - Cmax) / max(Cmax, epsilon) <= convergence_tolerance
budget change <= budget_tolerance
连续 convergence_samples 次成立
```

展示稳定区间的均值/P95：

```text
Cmax, Nmax, makespan
compute imbalance
per-NIC budget/utilization
admitted migration bytes
remote replica 数
```

当前 2 层/4-sample 测试不做“收敛前/后”性能结论，只检查 sample 编号、状态字段和
依赖完整。

## 11. 静态 DAG 与真实反馈边界

当前 Transformer workload 是一次性生成的静态 DAG。HTSim 开始执行后，Python
builder 不能读取 sample `t` 的真实 FCT 再改写同一 DAG 中 sample `t+1` 的 tasks。

因此分两步实现：

```text
首版：
  独立 stateful NIC controller
  + 可注入 SampleFeedback API
  + 纯 Python 合成反馈收敛测试
  + 2-layer 静态 DAG 验证 sample/task/依赖

58-layer 闭环实验：
  维护双 microbatch in-flight 队列
  -> sample 完成时解析真实 C_i/N_i 并 controller.update()
  -> 新迁移决策读取最新已完成 feedback
  -> 生成/执行下一个可释放 sample
```

在闭环 driver 完成前，静态 2-layer DAG 不应把 analytical proxy 声称为真实网络
probe 结果。这个限制不影响当前 DAG 对已接纳 weight migration、token communication
和 Expert FFN overlap 的仿真；它只表示后一个 sample 尚不能读取本次 HTSim 运行中
前一个 sample 的实测 FCT 并在线改写自身决策。

## 12. 复杂度

Gate 若直接提供 `(server, expert)` route histogram，在线控制路径为：

```text
服务器间 greedy: O(E log E + M log S)
服务器内 greedy: O(E log E + M log G)
controller update: O(P)
chunk scheduling:  O(weight_chunk_count * G)
```

EP32/DSV3 中 `E=256`、`P=32`、`G=8`、每 replica 21 chunks。在线 planner 不扫描、
保存或排序全部百万 routes。若输入只有 expanded assignments，则先聚合 histogram
需要一次 `O(R)`；仿真生成器把最终 moved counts/execution quotas 展开回具体 route
keys 还需要 `O(R)`。这两个线性 pass 是输入处理/仿真 lowering，不是 greedy 搜索
复杂度，也不作为 DAG task。

## 13. 配置草案

```yaml
algorithm: probeep
planning_mode: oracle_current_routes
planner_runtime_model: not_in_dag
route_lowering_runtime_model: offline_not_in_dag

compute_planner:
  server_first: true
  local_second: true
  route_chunk_tokens: 4096
  token_padding: 128
  expert_slots_per_rank: 40

nic_controller:
  feedback_source: external_sample_feedback
  initial_budget_bytes: 16777216
  min_budget_bytes: 0
  max_budget_bytes: 134217728
  multiplicative_decrease: 0.9
  additive_increase_bytes: 1048576
  deadband_ratio: 0.05

weight_transport:
  chunk_bytes: 4194304
  nic_assignment: token_baseline_plus_migration_waterfill
  require_complete_replica: true
  scale_out_transport: same_rail_rdma
  scale_up_transport: server_local_fullmesh

token_transport: deepep_hierarchical
```

`expert_slots_per_rank` 是每个 rank 的总 expert 权重槽位，包含静态 home experts 和
临时 local/remote replicas，不是“最多迁移多少个 expert”。DSV3 EP32 每 rank 固有
8 个 home experts；测试配置 40 个总槽位，相当于最多再容纳 32 个临时 experts。
当前研究假设显存足够，因此测试直接给足容量，不把 slot cap 当作算法瓶颈。

## 14. Manifest

每个 sample 至少记录：

```text
sample_id, layer, microbatch
planner_runtime_model, route_lowering_runtime_model
server_load_before/planned/admitted
rank_load_before/after
planned/admitted/deferred migration intents
remote replica seed/local replicas
expert_slots_per_rank, home_expert_slots_by_rank
weight chunks and rail assignments
predicted token TX/RX bytes by rank
budget_before/budget_after
feedback C_i/N_i/source
controller action and bottleneck ranks
dispatch/combine hierarchy bytes
```

固定 Gate sample、固定 controller state 和固定配置必须得到确定性的 migration plan、
admission 和 DAG。Python 生成器 wall-clock 不进入 manifest 性能口径，也不参与
确定性 plan tie-break。

## 15. 验收标准

1. 服务器间 planner 不读取 NIC load/budget/bandwidth/weight bytes。
2. 第一阶段先降低 server compute imbalance，第二阶段再降低 server 内 rank imbalance。
3. 网络只能接纳/延后 intents，不能改变 compute planner 的优先级。
4. 一个 remote replica 必须完整传输 84 MiB/21 chunks 后才能执行。
5. chunk scheduler 同时检查 source TX 和 destination RX budget。
6. 轻载 NIC 获得更多 chunks；相同输入和 state 的分配确定性。
7. remote seed 之外的同服务器 replicas 只通过 NVLink 获取权重。
8. ProbeEP DAG 不包含 CPU planner task，manifest 必须报告 `cpu_task_count=0` 和
   `cpu_streams_global=0`，timeline 不出现 CPU planner lane。
9. expert slot 容量按 home + temporary 总数校验；显存充足测试不得因人为小容量失败。
10. `Nmax` 超过 deadband 时 bottleneck budget 乘 0.9；计算仍为瓶颈时窗口缓慢恢复。
11. 无迁移任务的 sample 不把 budget 错误降到 0。
12. 2-layer 测试验证 4 个 sample 状态，但不声称收敛。
13. 58-layer 闭环实验必须分别展示前 N 次和收敛后区间。

## 16. 当前不做

- backward replica gradient reduction 和 optimizer state；
- 未完整权重上的部分 Expert FFN；
- 跨 sample 传一半权重并缓存到下一轮；
- 动态 SM/HBM/NVLink 竞争；
- 多 plane 联合窗口控制；
- 在静态 DAG 内伪造 HTSim 运行时反馈。

这些边界保留了核心问题：用低复杂度 histogram greedy 先规划计算均衡，再用类似 cwnd 的
per-rank 字节窗口逐 sample 探测可被计算掩盖的 RDMA 迁移量，最后由轻载 NIC 承担
更多完整 expert weight chunks。
