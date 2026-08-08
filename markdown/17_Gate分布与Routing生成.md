# Gate 分布与 Routing 生成

## 1. 目标与范围

Gate（也称 MoE Router）决定每个 source token 选择哪些逻辑专家。
本模块负责把可配置的概率分布或接收侧统计转成当前 MoE IR 需要的
逐 token routing assignment：

```text
(src_rank, token_id, topk_slot, logical_expert_id, route_weight)
```

输出必须能被 `MoEInvocation.assignments` 直接消费，并在 NCCL、DeepEP、
EPLB 和 MoonEP 之间共享。算法可以改变 token 最终在哪个物理副本上
执行，但不得改变 Gate 已经选中的逻辑专家。

本文档同时描述当前已经实现的数据语义、生成模式、可视化和验证契约。

## 2. 不要混淆的三个层次

```text
Gate 分布
  -> 逐 token 选择 logical experts
  -> 算法根据 placement/replica 选择 execution ranks
  -> NCCL/DeepEP 把 assignments 聚合为通信 flow
```

| 层次 | 描述 | 本模块是否决定 |
|---|---|---|
| Gate/Router | token 选中哪些 logical expert | 是 |
| Expert placement | logical expert 的 master/replica 在哪些 rank | 否 |
| Network routing | flow 在 MpRail 中选 ECMP/spray/source route | 否 |

Gate 采样概率和 `route_weight` 也不是一个概念：

- `selection_probability[e]` 用于决定 expert `e` 被 Top-K 选中的频率；
- `route_weight` 是 combine 时使用的 Gate 权重；
- 只有 expert 接收计数时无法恢复真实 `route_weight`，默认写为 `1 / topk`。

## 3. Source 发送量是否固定

对 source rank `r`，设当前 microbatch 有 `S_r` 个 token，Top-K 为 `K`。
在不 drop token 的模式下：

```text
logical_route_count[r] = S_r * K
global_logical_routes  = sum_r(S_r) * K
```

因此每个 source rank 的逻辑 route 发送预算是固定的。Gate 分布只改变
这些 route 发往哪些 expert/rank/server，不改变 token 数和 Top-K 数。

但底层网络字节不一定固定：

- NCCL 按每条 logical route 传 payload，核算量接近 `S_r * K`；
- DeepEP 先按同 token/同 destination rank 去重，再按同 destination server
  合并 scale-out payload；
- 同一 token 的 K 个 expert 落到多少个 rank/server 由 Gate 分布决定，
  所以 DeepEP 的实际 TX bytes 会随分布变化；
- source 本地 expert 的 route 不会使用 scale-out RDMA。

报告必须同时记录 logical route 数、unique destination rank/server 数和最终
网络字节，不能只用“每 rank 发送量固定”概括。

## 4. Gate provider 契约

统一接口概念上为：

```python
sample(
    layer_id,
    microbatch_id,
    tokens_per_source_rank,
    num_logical_experts,
    topk,
    seed,
) -> GateSample
```

`GateSample` 至少包含：

```text
assignments              # tuple[RoutingAssignment, ...]
provider_name
provider_parameters
base_seed
routing_fidelity
target_expert_weights    # 可选，未量化或归一化概率
realized_expert_loads
realized_rank_loads      # 按 baseline placement 聚合
realized_server_loads
```

所有 provider 共享以下强约束：

1. 每个 `(src_rank, token_id)` 恰好有 `K` 个 assignment。
2. 同一 token 的 K 个 logical expert 不重复。
3. `expert_id` 在 `[0, E)` 内，不出现 physical replica slot ID。
4. 每个 source rank 的 token 数与模型输入完全一致。
5. 同一 seed 和配置必须产生完全一致的 assignments。
6. 四个 MoE 算法在同一实验中必须复用同一份 GateSample，不得
   分别采样。

## 5. Provider 模式

### 5.1 `balanced_permuted`：精确均衡基线

对 expert ID 使用固定 seed 打乱，再按全局 token 序号循环分配 Top-K。
它对应当前 `make_uniform_assignments()` 的语义：

- 路由数在 logical experts 间严格均衡；
- 输出确定，适合 smoke 和回归测试；
- 它不是独立同分布的 uniform random sample。

### 5.2 `uniform_random`：独立均匀采样

对每个 token 从 E 个 logical experts 中等概率、无放回地选 K 个。
它仅保证期望均衡，小 token 数时会出现采样波动。

### 5.3 `ultra_rank_zipf`：UltraEP 风格专家热度

UltraEP 测试不只对全部 expert ID 使用一个 Zipf 序列，而是将连续
placement 下的 expert 权重分解为：

```text
weight(rank, local_expert)
  = 1 / (rank_index + 1)^rank_alpha
    * 1 / (local_index + 1)^local_alpha
```

然后根据目标 rank 不均衡度进行二分搜索：

```text
target_rank_imbalance = expected_max_rank_load / expected_mean_rank_load
```

支持两种参数方式，二选一：

```text
rank_alpha + local_alpha
target_rank_imbalance
```

默认使用 Top-K 无放回加权采样。为了避免每层永远是同一批
expert/rank 变热，先用 `(base_seed, layer_id)` 对 expert ID 做稳定 permutation，
再应用权重。

参考实现是 [UltraEP tests/utils.py](../../UltraEP/tests/utils.py)；本项目只复用
分布建模思路，不引入 UltraEP 的 replica planner。

### 5.4 `fast_matrix_zipf`：FAST 风格的 source-specific skew

FAST 原始代码直接生成 `rank x rank` All-to-Allv traffic matrix：

- `fixed_distribution`：每个 cell 相同；
- `uniform_distribution`：每个 cell 独立 uniform random；
- `zipf_distribution`：每个 cell 通过 inverse Zipf CDF 生成；
- `zipf_distribution2`：Zipf 采样后再缩放到指定平均 per-GPU size。

这些数据是流量矩阵，不是 Gate trace，不能把 FAST 的可变 row sum 直接
当作每个 source rank 的 token 数。在 Gate provider 中做如下适配：

```text
W[src_rank, logical_expert] <- FAST-style uniform/Zipf sample
P[src_rank, :]              <- normalize(W[src_rank, :])
assignments[src_rank]       <- weighted Top-K sampling without replacement
```

这样保留 source-specific 不均匀性，同时强制每个 source rank 的总 route 数
仍为 `S_r * K`。本模式是对 FAST 分布形状的 Gate 适配，不声称逐 cell
复现 FAST 论文 workload。

参考代码为
[FAST nvidia/alltoall_nvshmem.cpp](../../FAST/nvidia/alltoall_nvshmem.cpp) 和
[FAST simulation/test.cpp](../../FAST/simulation/test.cpp)。

### 5.5 `raw_receive_cdf`：经验接收分布的精确 quota

读取 `workload/raw_data` 的 physical expert 接收计数，借助同层 placement
JSON 折叠为 logical expert 直方图，再按当前仿真总 route 数换算成整数 quota。
公开 provider 名保留 `raw_receive_cdf` 以兼容已有 CLI，但当前默认实现不是独立
CDF 抽样，而是全局直方图 quota 精确匹配。

该模式的 fidelity 必须标记为：

```text
routing_fidelity = quota_matched_global_receive_histogram
```

它保留每层逻辑专家的热度形状，但不是原始逐 token routing trace。

## 6. `raw_data` 实际格式

当前数据由一个 placement JSON 和 32 个 CSV 组成：

```text
workload/raw_data/
  ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json
  decode_0.csv
  ...
  decode_31.csv
```

已核对的 shape：

```text
layers                         = 58
devices/ranks                  = 32
physical expert slots/rank     = 9
physical slots/layer           = 32 * 9 = 288
unique logical experts/layer   = 256
redundant slots/layer          = 288 - 256 = 32
CSV shape/file                 = [58 layers, 9 physical slots]
sum of receives/layer          = 2,868,128  # 当前数据每层相同
```

`decode_r.csv` 的第 `l` 行给出 rank `r` 在 layer `l` 的 9 个 physical
slot 接收量。JSON 中的：

```text
layer_list[l].device_list[r].device_expert[s]
```

给出 physical slot `(l, r, s)` 对应的 logical expert ID。相对每 rank 8 个
基线 expert 的布局，每 rank 多出 1 个 physical capacity，全层共有 32 个
额外实例。但 CSV 的第 9 列不能固定解释为 replica：JSON 中 slot
已经重排，也没有 master/replica 标记。只能根据同一 logical expert ID
在全层 288 个 slot 中的重复出现识别额外实例；一个热专家可能有多个副本。

### 6.1 先从 physical 计数折叠回 logical expert

设 CSV 接收计数为 `C[l,r,s]`，JSON 映射为 `M[l,r,s]`，则：

```text
H[l,e] = sum over (r,s) where M[l,r,s] == e of C[l,r,s]
P[l,e] = H[l,e] / sum_e H[l,e]
```

如果 expert `e` 在 master 和 replica slot 中同时出现，必须先把这些 physical
slot 的计数相加。Gate 选择的是 logical expert，不能把 288 个 physical
slot 当作 288 个不同专家，否则会：

- 改变原模型 `E=256` 的数学语义；
- 重复计算副本 expert 的 Gate 概率；
- 允许同一 token 在 Top-K 中两次选中同一 logical expert。

### 6.2 精确 quota 分配

设本次 invocation 总 token 数为 `T`、Top-K 为 `K`，先计算：

```text
R = T * K
expected_quota[e] = P[layer,e] * R
integer_quota     = floor + largest-remainder 补齐到 R
```

然后使用 seeded bipartite greedy 将全部 expert quota 分配到 token：

1. 每个 token 恰好选择 K 个 logical experts。
2. 同一个 token 不允许重复选择同一个 logical expert。
3. 每个 source rank 仍恰好发送 `S_r*K` 条 routes。
4. 每个 expert 的 realized count 与实数目标 quota 的取整误差不超过 1 route。
5. seed 只改变 route 落到哪些 source/token 和 Top-K slot，不改变全局整数 quota。
6. 默认 `route_weight=1/K`。

因此它保证当前仿真的全局 logical-expert 接收直方图匹配 raw 热点分布；有限 route
数仍有不可避免的整数取整误差。它不保证每个 source rank 各自复现原 trace，因为
raw receive 数据没有 source-token 关联。

### 6.3 缩放到当前仿真 token 数

raw CSV 的 `2,868,128` 是原始采样窗口的 route 计数，不直接作为
仿真的 token 数。当前 invocation 仍使用模型配置：

```text
full:  S_r = 4096 tokens/rank/microbatch
smoke: S_r = 2 tokens/rank/microbatch
K    = model.topk
```

raw data 只提供归一化后的 expert 热度 `P[l,e]`。因此无论原始统计
窗口有多大，当前 EP32 full workload 的逻辑 route 总数仍为：

```text
32 * 4096 * K
```

### 6.4 Layer 和 microbatch 映射

默认 layer policy 为 direct mapping：仿真 layer 0/1 使用 raw layer 0/1。
也可通过显式 list 选择代表性 raw layers，例如 `[3, 41]`。不做静默 modulo
或自动重复，超出 58 层时应报错或要求显式 policy。

同一 raw layer 可以为多个 microbatch 提供同一份 quota 分布。每个 microbatch
使用不同稳定子 seed 改变 source/token assignment，但全局 expert quota 相同；
这不能声称恢复了原 trace 的时间非平稳性。

## 7. 精度分类

| `routing_fidelity` | 输入 | 能表达 | 不能表达 |
|---|---|---|---|
| `exact_token_trace` | 逐 token Top-K trace | 真实 source/expert 关联 | 未记录的 Gate logits |
| `synthetic_assignments` | balanced/uniform/Zipf 生成器 | 可控分布和可复现性 | 真实模型相关性 |
| `sampled_from_source_expert_matrix` | FAST-style `src x expert` 权重 | source-specific skew | 真实 token 语义 |
| `quota_matched_global_receive_histogram` | raw receive counts | 每层全局 expert 热度，整数误差不超过 1 route | 原始 source rank 和 token 关联 |

`raw_receive_cdf` 不知道每个 expert 的 token 来自哪个 source rank。首版对所有
source tokens 只执行全局 quota 分配，不额外伪造 source-expert 相关性；各 source
rank 的局部直方图可以有确定性的分配波动。

## 8. 当前目录与代码边界

相关 Python 代码放在项目根目录的 `workload/gate/`：

```text
workload/
  raw_data/              # 只读原始数据，不原地改写
  gate/
    __init__.py
    base.py              # 加权采样、精确 quota 和稳定子 seed
    synthetic.py         # uniform、UltraEP-style 和 FAST-style 分布
    raw_receive.py       # CSV + placement JSON 解析和精确 quota

pysrc/moe_dag/
  gate.py                # GateProvider/GateSample、balanced 和统一统计
  load_profile.py        # 算法执行前后的 expert-instance/rank 负载
```

Gate 模块只生成 `RoutingAssignment`，不依赖 HTSim，不创建 network task，
不在内部实现 NCCL/DeepEP/EPLB/MoonEP/ProbeEP 逻辑。Transformer builder 通过
provider 参数注入 assignments，不再由 Transformer builder 写死 assignment
生成逻辑。五种算法都会把统一的 `expert_load_profile_v1` 写入 manifest。

## 9. 当前配置

```yaml
gate:
  provider: raw_receive_cdf
  seed: 0
  route_weight_policy: equal
  microbatch_sampling: independent
  raw_receive:
    directory: workload/raw_data
    placement_json: ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json
    csv_pattern: decode_{rank}.csv
    layer_map: [0, 1]
```

CLI 与其对应：

```text
--gate-provider balanced_permuted|uniform_random|ultra_rank_zipf|fast_matrix_zipf|raw_receive_cdf
--gate-seed 0
--gate-raw-placement-json <path>
--gate-raw-csv-pattern 'decode_{rank}.csv'
--gate-layer-map 0,1
--gate-rank-alpha <float>
--gate-local-alpha <float>
--gate-target-rank-imbalance <float>
--gate-fast-skew <float>
```

未选 provider 时保持当前 `balanced_permuted` 行为，避免现有 smoke/full
测试在不显式改参数时改变 workload。

## 10. 输出报告与 before/after

每个 layer/microbatch 至少输出：

```text
provider + parameters + seed
raw layer ID / model layer ID
tokens per source rank
logical route count
logical expert histogram
baseline destination-rank histogram
baseline destination-server histogram
source-rank x destination-rank route matrix
source-server x destination-server route matrix
rank max/mean imbalance
server max/mean imbalance
target-vs-realized histogram distance
routing_fidelity
```

上述 Gate 统计写在每条 `micro_batch_algorithms[*].gate` 中。算法执行后的
physical instance 负载另写在同一条记录的 `expert_load_profile`：

```text
before = 原始 logical expert 按 baseline placement 执行
after  = 当前算法选中的 physical expert instance / execution rank
```

`before` 和 `after` 都记录 instance、logical expert、rank、server 的负载及
`max/mean`。两者必须保持相同的 `total_routes` 和 `logical_expert_loads`；算法
只允许改变执行位置和实例分流，不允许偷偷改变 Gate 结果。NCCL/DeepEP 没有
专家副本重排，所以 before 和 after 相同；EPLB/MoonEP 才可能不同。

### 10.1 UltraEP profiler 与本项目的对应关系

UltraEP 的 profiler 在每个 EP group、每个 microbatch、每层采集 token 重路由
前后的专家负载，并在 HTML 中提供总体分布和单 microbatch 的 rank/expert
明细。它的重点是在线实测运行时状态。

本项目采用相同的层次化观察口径，但数据来源不同：

| 口径 | UltraEP | 本项目 |
|---|---|---|
| before | Gate 后、重路由前的实际负载 | `MoEInvocation.assignments` 按 baseline placement 聚合 |
| after | quota/replica reroute 后的实际负载 | 算法 builder 的 execution instance/rank 聚合 |
| 时间点 | 运行时 profiler 异步采集 | DAG 生成时确定性写入 manifest |
| 粒度 | EP group/layer/microbatch/rank/expert | algorithm/layer/microbatch/rank/expert instance |

因此页面是 workload/算法计划的可视化，不冒充 GPU 运行时 profiler，也不从
HTSim 的 flow 日志反推专家负载。

### 10.2 HTML 输出

```bash
python3 visualization/gate_load_profile.py \
  --workload-dir <generator-output-dir> \
  --output-dir <case-dir>/gate_load
```

输出：

```text
gate_load/
├── gate_load_profile.html
├── gate_load_profile.csv
└── gate_load_profile_summary.json
```

HTML 可选择 layer/microbatch，使用相同的 load 横轴并排显示 before/after 的
32 个 rank；每个 rank 内按 logical expert 分段。下方另画 Gate logical-expert
分布，并提供按 state/rank 筛选的 expert-instance 明细。该模块也会进入五算法
可视化 ZIP，服务器 run 目录在 ZIP 校验成功后不保留散装 HTML。

## 11. 验证和功能测试

独立功能测试入口为：

```bash
python3 tests/run_gate_workload_visualization.py
```

测试代码放在 `tests/`，全部产物放在一次独立的 `test_logs/run_*` 目录，覆盖：

1. raw parser 验证 `32 x 58 x 9` 和 JSON 中的 `58 x 32 x 9`。
2. 每层 288 physical slots 折叠后恰好得到 256 logical experts。
3. 折叠前后的 receive count 总和完全一致。
4. 每个 token 恰好有 K 个不重复 logical expert。
5. 每 source rank 的 route count 恰好为 `S_r * K`。
6. 同 seed 输出逐 assignment 完全相同，不同 seed 的直方图仍贴近目标。
7. full 的 target/realized histogram 仅允许 largest-remainder 取整误差。
8. 五算法同一 case 的 Gate assignment digest 完全相同。
9. NCCL logical route 总数不变，DeepEP unique rank/server payload 随分布正确变化。
10. 错误 layer 数、CSV 列数、负数 count、越界 expert ID 和少于 K 个正权重
    expert 必须显式失败。

## 12. 明确不建模的内容

首版不过度开发，不包括：

- 从模型 Gate 权重和 hidden state 实际计算 logits；
- 从接收直方图唯一恢复原始 source-token-expert trace；
- 伪造 raw data 中不存在的 route weights 或 source 相关性；
- 在 Gate provider 内执行 expert replication 或网络路由；
- 将 `decode_*.csv` 直接宣称为训练/prefill 真实分布。当前数据
  可用作 empirical-skew workload，但实验报告必须保留其 decode 来源标签。

## 13. 已完成的数据路径

```text
GateProvider -> GateSample
  -> TransformerWorkloadConfig.gate_provider
  -> MoEInvocation.assignments
  -> NCCL / DeepEP / EPLB / MoonEP / ProbeEP builder
  -> gate + expert_load_profile 写入 manifest
  -> gate_load_profile.html
  -> 五算法 dsv3_visualization_bundle.zip
```

五算法 runner 会检查每个算法的 assignment digest 序列完全相同，然后才允许
生成合并 ZIP。Gate 分布错误时，应先检查 pure-Python 的采样与守恒报告，不应用
网络结果反向猜测问题。偏斜 Gate 也可能超过 MoonEP 配置的临时副本容量；runner
不会静默放宽算法约束，可通过 `--moonep-replicas-per-rank` 显式给出容量。当前
raw layer 0/1 的 2-token smoke、seed 17 下，MoonEP 需要显式给足临时 replica 容量；
相同 layer/seed 的 4096tpr MoonEP quota 曾使用 14 temporary replicas/rank。ProbeEP
采用不同口径：`expert_slots_per_rank=40` 是 8 home + 最多 32 temporary 的总容量。
当前 ProbeEP 实验假设显存充足，不用人为 slot cap 限制迁移。
