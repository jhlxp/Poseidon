# 算法建模：EPLB 理念与边界

## 1. 本项目只关心什么

本文基于本地官方仓库 `/home/xuheng/EPLB` commit `d52c72d`，抽取 DeepSeek
EPLB 在训练和 prefill 场景中的核心理念。

EPLB 是 expert replication/placement planner，不是 token 通信库：

```text
估计的每 expert load
  -> 复制哪些热点 logical experts
  -> physical expert replicas 放在哪些 node/GPU
  -> 输出 logical/physical expert 映射
```

它不负责：

- 收集或预测 expert load；
- 决定多久 rebalance 一次；
- 在新旧 placement 之间搬运 expert weights；
- 把当前 batch 的每条 token route 分配给哪个 replica；
- 执行 dispatch/combine、RDMA 或 NVLink 通信。

本项目已把 EPLB 放在 DeepEP workload 之前，作为低频 placement 层，而不是复制
MoonEP 的在线 planner。实现位于 `pysrc/moe_dag/algorithms/eplb.py`，CLI 算法名为
`eplb`。

## 2. 输入 load 是什么

官方入口：

```python
rebalance_experts(
    weight,
    num_replicas,
    num_groups,
    num_nodes,
    num_gpus,
)
```

其中：

```text
weight.shape = [num_moe_layers, num_logical_experts]
weight[layer, expert] = 该 logical expert 的估计 load
```

`weight` 不是模型 parameter weight，而是负载权重。官方仓库不规定其预测方法，
README 只给出一个常见方案：对历史 expert load statistics 做 moving average。

所以“EPLB 使用历史”更准确地说是：

```text
EPLB 只接收 estimated loads；
历史移动平均是调用方常用的数据来源，但不是 eplb.py 内置逻辑。
```

训练/prefill 仿真若使用历史方法，至少需要在 manifest 记录：

```text
load_source = historical_moving_average
statistics_window
rebalance_interval
load_snapshot_id
```

这些字段用于说明 placement 的来源，不应被解释为新的 DAG 字段。

## 3. `num_replicas` 的准确含义

官方参数 `num_replicas` 实际表示 replication 后的 physical expert 总数：

```text
num_physical_experts = num_replicas
num_redundant_experts = num_physical_experts - num_logical_experts
```

例如 12 个 logical experts、`num_replicas=16` 表示：

```text
12 个基础 physical instances
+ 4 个 redundant instances
= 16 个 physical expert slots
```

它不是“每个 logical expert 有 16 份”。文档和未来配置必须避免把该字段误写成
redundant count。

## 4. 三个核心函数

### 4.1 `balanced_packing`

输入 `n` 个带 weight 的对象和 `m` 个 packs，要求：

```text
n % m == 0
每个 pack 恰好放 n/m 个对象
```

算法先按 weight 从大到小遍历对象，每次选择：

```text
尚未达到对象数量上限、当前累计 weight 最小的 pack
```

输出：

```text
pack_index[item]   = item 被放入哪个 pack
rank_in_pack[item] = item 在该 pack 内的槽位
```

它是确定性的 greedy packing，不求解全局最优装箱问题。它同时约束“每个 pack 的
对象数量相等”和“累计估计 load 尽量平衡”。

### 4.2 `replicate_experts`

输入 logical expert loads 和 physical expert 总数。初始每个 logical expert 有一个
physical instance；每增加一个 redundant slot，选择：

```text
argmax_expert(weight[expert] / current_replica_count[expert])
```

然后将该 expert 的 replica count 加一。这个 greedy 规则近似最小化最大
per-replica load，并隐含假设 runtime 可以把 logical expert 的 token load近似均分
到它的所有 physical replicas。

输出：

```text
phy2log[physical_slot] = logical expert
phyrank[physical_slot] = 该 physical instance 是 logical expert 的第几份
logcnt[logical_expert] = physical replica count
```

### 4.3 `rebalance_experts_hierarchical`

训练/prefill 关注的 hierarchical policy 分三步。

第一步，expert group 放到 node：

```text
tokens_per_group = sum(weight of experts in group)
balanced_packing(groups, num_nodes)
```

每个 node 获得相同数量的 expert groups，同时尽量平衡 group load。group-limited
routing 下，把同 group experts 保留在同一 node 可以减少跨节点目的域数量。

第二步，在每个 node 内复制热点 experts：

```text
replicate_experts(
  logical experts assigned to this node,
  physical slots assigned to this node,
)
```

replica 不跨越第一步确定的 node 边界。

第三步，把 physical experts 放到 node 内 GPU：

```text
estimated replica load = logical expert load / replica count
balanced_packing(physical experts, GPUs in node)
```

每张 GPU 获得完全相同数量的 physical expert slots，同时累计估计 load 尽量平衡。

## 5. Hierarchical Policy 的约束

官方代码要求：

```text
num_logical_experts % num_groups == 0
num_groups % num_nodes == 0
num_gpus % num_nodes == 0
num_physical_experts % num_gpus == 0
```

相应含义：

- 每个 expert group 的 logical expert 数相等；
- 每个 node 的 expert group 数相等；
- 每个 node 的 GPU 数相等；
- 每张 GPU 的 physical expert slot 数相等。

官方入口在 `num_groups % num_nodes != 0` 时退化为 global policy，本质上使用：

```text
num_groups = 1
num_nodes = 1
```

然后跨全部 GPU 全局复制和 packing。该路径主要面向 decode；本项目只研究训练和
prefill，因此首版 EPLB 只需要 hierarchical policy，不开发 global/decode 分支。

## 6. 输出映射

`rebalance_experts` 返回：

```text
phy2log[layer, physical_slot]
log2phy[layer, logical_expert, replica_index]
logcnt[layer, logical_expert]
```

其中：

- `phy2log` 回答每个 physical slot 放的是哪个 logical expert；
- `log2phy` 回答一个 logical expert 有哪些 physical slots，无效尾部填 `-1`；
- `logcnt` 记录每个 logical expert 的实际 replica 数。

这些输出只定义 placement。给定一条：

```text
(src_rank, token_id, topk_slot, logical_expert)
```

官方 EPLB 代码没有决定应选择 `log2phy` 中哪一个 physical replica。当前 workload
生成器明确增加了独立的 `deterministic_round_robin_v1` selector；它是本项目把
placement 转换为具体 `.dag` 的策略，不是官方 EPLB 源码已有功能。

## 7. 在本项目中的位置

训练/prefill 的目标组合为：

```text
历史/预测 load
  -> EPLB hierarchical placement
  -> runtime replica selection
  -> DeepEP dispatch
  -> Expert compute
  -> DeepEP combine
```

EPLB 生成的是后续若干 iteration 共用的 placement。正常稳态 block DAG 不应在
每层、每 microbatch 重复生成 expert migration flow。

只有明确研究 rebalance event 时，才在两个 placement epoch 之间加入一次性：

```text
old physical expert rank
  -> new physical expert rank
  -> placement activation barrier
```

这类 weight migration 应跨多个 iteration 摊销。没有 `rebalance_interval` 时，不能
默认把它塞进每次 MoE invocation。

## 8. 与 DeepEP、MoonEP 的区别

| 维度 | EPLB | MoonEP |
|---|---|---|
| 规划输入 | estimated expert loads | 当前 invocation 的真实 router routes |
| 常见时间尺度 | 历史窗口后周期性 rebalance | 每次 invocation 在线 planning |
| 核心输出 | persistent physical expert placement | 当前 routes 的 execution rank |
| replica 范围 | hierarchical 模式先 node、再 node 内 GPU | 本项目固定 expert home server 内 |
| 当前 batch 保证 | 只平衡估计 load | per-server real routes floor/ceil 平衡 |
| weight movement | placement epoch 切换时一次性处理，官方代码未实现 | 当前 invocation 的 prefetch task |
| token 通信 | 交给 DeepEP 等通信库 | 本项目复用 DeepEP scale-out transport |

二者都使用 redundant experts，所以单个静态快照可能很像。根本差异是：

```text
EPLB 改变一段时间内使用的 physical expert placement；
MoonEP 改变当前 invocation 每条 token route 的 execution placement。
```

## 9. 对 DAG 的影响

稳态 EPLB workload 的 DAG 与 DeepEP 基本相同，但输入 placement 已变化：

- logical expert 可能有多个 physical ranks；
- replica selector 改变 dispatch 的最终 destination rank；
- 每 rank 的 route count 和 Expert FFN `compute_us` 随之变化；
- DeepEP 仍负责去冗余和 destination-side forwarding；
- combine 从实际 execution rank 返回 origin rank。

EPLB planner 本身不需要伪造成每个 block 的 compute task。只有研究 planner 开销
或 placement epoch 切换时，才单独建模 planning/migration。

## 10. 当前配置与实现边界

CLI 用法：

```bash
python3 pysrc/generate_moe_dag.py \
  --output generated_workloads/eplb_demo \
  --algorithm eplb \
  --num-ranks 32 --gpus-per-server 8 \
  --num-experts 8 --topk 2 \
  --eplb-num-physical-experts 32 \
  --eplb-num-groups 4 \
  --eplb-loads 16,1,8,1,4,1,2,1
```

当前约定：

- `--eplb-num-physical-experts` 是全局 physical expert 总数，必须能被 rank 数整除；
- 该参数为 `0` 时，在按 rank 对齐的基础槽位上，为每个 rank 再增加一个冗余槽；
- `--eplb-num-groups` 为 `0` 时使用 server 数；
- `--eplb-loads` 是逗号分隔的 logical expert estimated loads；
- 未传 loads 时，第一次 invocation 的 route count 作为静态代理，后续 microbatch
  复用同一 placement；
- runtime selector 固定为确定性 round-robin；
- token transport 固定复用 DeepEP destination-rank 去重和目标端转发；
- 稳态 workload 不生成 planning、weight migration 或 weight prefetch task。

`rebalance_interval`、历史 moving-average 计算器、placement epoch 切换和迁移流尚未
实现。调用方可以先离线计算历史 load snapshot，再通过 `--eplb-loads` 输入。

## 11. 性能假设

当前研究直觉为：

```text
NCCL < DeepEP < EPLB < MoonEP
```

它是待实验验证的工作假设，不是实现保证。该顺序预期在 token load 明显倾斜、
EPLB 快照仍有效、MoonEP 的 invocation-level prefetch 成本可摊销时成立。均匀负载、
过期 EPLB 快照、小消息或较高权重预取成本都可能改变排序。测试只验证算法语义与
DAG 完成，不把 makespan 排序写成 correctness assertion。

## 12. 当前验收边界

- 使用官方示例 load 时，`phy2log/log2phy/logcnt` 与官方实现一致。
- 每张 GPU 的 physical expert 数完全相同。
- hierarchical 模式下每 node 的 group 数相同，replica 不离开 group 所在 node。
- 每个 logical expert 至少有一个 physical instance。
- replica estimated load 使用 `weight/logcnt`。
- 固定 load snapshot 时 placement 逐字节稳定。
- 稳态 DAG 不重复计算 weight migration bytes。
- 文档和 manifest 明确区分 estimated placement balance 与当前 batch exact balance。

当前 `tests/run_workload_generator.py` 还验证热点快照下最大 rank route count 从
DeepEP baseline 的 16 降为 EPLB 的 3，并由 HTSim 完整执行 EPLB 稳态 DAG。

## 13. 代码依据

- `/home/xuheng/EPLB/README.md`：load 来源、hierarchical/global policy 和官方示例。
- `/home/xuheng/EPLB/eplb.py`：`balanced_packing`、`replicate_experts`、
  `rebalance_experts_hierarchical` 和输出映射。
