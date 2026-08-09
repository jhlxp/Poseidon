# 第二层：模型到 DAG 生成

## 1. 第二层的职责

第二层把真实模型结构、并行策略和执行场景转换为完整 workload，并在遇到 MoE sublayer 时调用第一层算法插件。

它需要回答：

```text
一个 block、多个 block 或一次 iteration 中，
每张 GPU 何时执行哪个 kernel、产生多少 token、调用哪次 EP 通信，
以及这些操作之间如何依赖或重叠？
```

当前实现位于 `pysrc/moe_dag/models/transformer.py`。它按 `num_layers` 生成一个或
多个 forward MoE block 的 Attention、Router 和算法专属 Dispatch/Expert/Combine，
并支持多个 microbatch 和跨层依赖。当前不是下面完整算子清单的逐 kernel 实现：
norm、RoPE、residual、shared expert、TP/PP 和 backward 仍是后续范围。

当前内置合成 routing 是 `balanced_permuted_deterministic`：固定 seed 0 打散
expert index，再按 TopK 分组循环。它保证 expert route 总量严格均衡，同时允许同一
token 的多个 expert 落到同 rank 或同 server，从而能够验证 DeepEP 两级去冗余。
真实 trace、Zipf/hot-expert provider 尚未接入 CLI；第一层 API 可以直接传入完整
`RoutingAssignment`。

## 2. 模型规格

初始配置建议使用 YAML，核心字段如下：

```yaml
model:
  name: example_moe
  num_layers: 1
  hidden: 7168
  ffn_hidden: 2048
  num_attention_heads: 56
  num_kv_heads: 8
  head_dim: 128
  num_experts: 256
  topk: 8
  shared_experts: 1
  activation: silu
  norm: rmsnorm

execution:
  mode: prefill       # train | prefill | decode
  micro_batch: 1
  sequence_length: 4096
  dtype: bf16
  dispatch_dtype: fp8
  combine_dtype: bf16

hardware:
  gpu: h100_sxm
  compute_placeholder: bf16_dense_peak
  peak_tflops_per_gpu: 989
  total_sms_per_gpu: 132
  communication_sms_per_rank: 20

parallel:
  dp: 1
  tp: 1
  pp: 1
  ep: 16
  gpus_per_server: 8

moe_algorithm:
  name: deepep
  forwarding: destination
```

目标算法名包括：

```text
nccl           # 无去冗余、无 relay forwarding 的 rank-direct All-to-All
deepep         # destination-rank 去重 + 同 index relay/目标端 forwarding
eplb           # 持久 hierarchical placement + DeepEP 稳态传输
moonep         # 动态 expert replica workload
probeep        # 探测式跨服务器 replica + 多 NIC 权重迁移
```

`eplb` 先为一段 placement epoch 生成 physical expert mapping，再使用确定性
round-robin replica selector 和 DeepEP 构造每个 block 的通信计算 DAG。CLI 使用
`--eplb-num-physical-experts`、`--eplb-num-groups` 和 `--eplb-loads` 配置它；没有
显式 load snapshot 时，以第一次 invocation 的 route count 作为静态代理。

首版限制为 `TP=1`、`PP=1`。先把 EP、router 和完整 block 依赖做对，再扩展 TP collectives 和 PP microbatch pipeline。

## 3. 一个 block 的逻辑结构

推荐的 forward 粒度：

```text
input
  -> RMSNorm
  -> QKV projection
  -> RoPE
  -> attention core
  -> output projection
  -> residual add
  -> RMSNorm
  -> router projection
  -> top-k / routing
  -> MoE algorithm: planning + dispatch
  -> routed expert gate/up GEMM
  -> activation
  -> routed expert down GEMM
  -> MoE algorithm: combine
  -> shared expert branch merge
  -> residual add
  -> output
```

不是每一行都必须成为独立 compute task。拆分条件是至少满足一个：

- 中间存在通信或依赖边界；
- 可与另一分支重叠；
- 需要单独核算理论 FLOP 数；
- shape 或资源类型显著不同；
- 需要观察该阶段的完成时间。

例如 QKV projection 可作为一个 fused task；expert gate/up 可以按实际 kernel 是否 fused 决定一个或两个 task。

## 4. Token 数

对无 sequence parallel 的简单场景：

```text
tokens_per_rank = micro_batch * sequence_length / data_parallel_shard_factor
```

实际实现必须显式处理：

- padding 和变长 sequence；
- prefill 与 decode 的 token 数差异；
- sequence/context parallel 的 token shard；
- pipeline microbatch；
- dropped token 或 capacity factor；
- shared expert 是否处理全部 token。

模型层只决定 source token tensor；top-k assignment 交给 routing provider。

### 4.1 DeepEP 训练/prefill token 口径

DeepEP normal-kernel 测试的 `--num-tokens` 默认值为 4096，每个分布式进程在自己的
rank 上创建 `[4096, hidden]` 输入。因此本项目的标准 DSV3 训练/prefill workload
固定解释为：

```text
tokens_per_rank_per_microbatch = 4096
```

这个值不是所有 rank 的全局 token 总数。EP32 下每个 microbatch 的全局 token 数为
`4096 * 32 = 131072`。功能 smoke test 可以使用更小值加快逐包仿真，但 manifest、
测试报告和文档必须标记为 smoke，不能据此得出性能结论。

## 5. Router 输入来源

模型层把 routing provider 分成以下三类；当前 CLI 只内置第 5.2 节中的
balanced-permuted 功能基线：

### 5.1 Trace

输入真实模型采集的：

```text
(layer, step, src_rank, token_id) -> [(expert_id, weight), ...]
```

这是准确性最高的方式。trace metadata 必须记录模型版本、batch、sequence、并行布局和采集 commit。

### 5.2 合成分布

规划用于功能测试和参数扫描的合成分布包括：

- balanced-permuted deterministic（当前已实现，seed 0）；
- perfectly balanced；
- uniform random；
- Zipf/skew；
- hot expert；
- group-limited top-k；
- layer-correlated 或 step-correlated routing。

合成器必须使用显式 seed，并输出每 expert/rank/server 的 token histogram。

### 5.3 仅计数输入

只有 `tokens_per_expert` 不足以恢复精确通信矩阵，因为不知道 token 来自哪个 source rank，也不知道同一个 token 的多个 expert 是否位于相同 rank/node。

因此仅计数输入只能用于 approximate 模式，manifest 必须标记：

```text
routing_fidelity = histogram_only
```

## 6. Compute 固定时长占位

### 6.1 默认理论硬件口径

没有指定计算 JSON 时，生成器采用 H100 SXM 的 dense BF16 Tensor Core 理论峰值，
并为重叠通信固定预留 20 SM：

```text
per_gpu_peak = 989 TFLOP/s = 989e12 FLOP/s
eight_gpu_peak = 7.912 PFLOP/s
total_sms_per_gpu = 132
communication_sms_per_rank = 20
overlap_compute_sms = 112
overlap_compute_peak = 989 * 112 / 132 ~= 839.15 TFLOP/s
```

NVIDIA 当前 H100 产品规格给出的 BF16/FP16 `1,979 TFLOP/s` 带结构化稀疏；首版不假设模型满足 2:4 sparsity，因此使用约一半的 dense 值 `989 TFLOP/s`。每个 compute task 绑定一张 GPU/rank，所以只使用单卡值。8 卡总值仅写入报告。

`communication_sms_per_rank=20` 是本项目实验配置，不是 DeepEP 的固定事实；这里
用一个固定参数近似通信 kernel 和计算 kernel 的主要资源竞争。

### 6.2 固定时间来源

默认模式离线计算理论 FLOP 数，然后根据 task 是否位于通信 overlap 窗口写入固定
`compute_us`：

```text
compute_us_normal  = operation_flops / 989e12 * 1e6
compute_us_overlap = operation_flops / 839.15e12 * 1e6
```

FMA 按 2 FLOPs 计算。写入 `.dag` 后 HTSim 只等待这个固定时间，不根据并发 task 或 GPU 状态重新计算。首版约定 overlap compute 在整个 task 生命周期中都使用 112 SM；通信提前结束时不恢复额外 SM。

GEMM 的基础 FLOP：

```text
linear(M, K, N) = 2 * M * K * N
```

典型 SwiGLU expert 对每个实际计算 token 的主 GEMM FLOP 近似：

```text
gate: 2 * H * H_ff
up:   2 * H * H_ff
down: 2 * H_ff * H
total ~= 6 * H * H_ff
```

首版只核算能够明确写出公式的主要计算。activation、bias、kernel launch、padding、内存访问和实际 Tensor Core 利用率暂不修正，所以该时间是乐观理论下界，而不是性能预测。

attention 仍需按 prefill/decode 和实际 shape 计算 FLOP 数，但同样统一除以 `989e12`，不单独建模 KV cache 或内存瓶颈。

指定 `--compute-config` 后，生成器改为读取模块级 JSON。每个模块包含
`theoretical_us_per_token` 和 `profiled_us_per_token`，`selected_source` 决定
单 token 单价，再与当前 task 的 `token_count` 相乘得到 `compute_us`。JSON 模式
仍记录理论 `operation_flops`，但不再用它换算时间，也不按 overlap SM 比例二次
缩放。不同 rank 的 expert route 数不同，FFN 时间也必须不同。完整格式、模块名和失败规则见
[16_计算时间JSON配置.md](16_计算时间JSON配置.md)。

### 6.3 与网络的边界

只有模型计算 FLOP 进入 `compute_us`。以下时间不加入 compute task：

- HTSim 已经仿真的 network task FCT；
- DeepEP hierarchy legs 或通用 `server_forward` 的网络/本地 flow 时间；
- 等待 predecessor barrier 的时间。

profiling 时间不能包含 RDMA/NVLink 等待，否则会与 HTSim network FCT 重复计算。

一个逻辑通信 phase 可以展开成多条 destination flow 或 chunk flow。这些 flow 在同一 rank 上共享一次 20 SM 预留，而不是每条 flow 各占 20 SM。phase 归属记录在 `task_map.json`/manifest 中，不修改 `.dag` 行格式。

## 7. 并行分支

### 7.1 Shared expert

shared expert 通常处理全部本地 token，可以与 routed token dispatch 或 routed expert compute 形成并行分支：

```text
router_done
  +-> routed dispatch -> routed experts -> combine -+
  +-> shared expert compute -----------------------+-> merge
```

是否重叠由 framework 和 stream 语义决定，不能默认。配置应显式选择：

```text
shared_expert_schedule = serial | overlap_dispatch | overlap_moe
```

### 7.2 Microbatch overlap

训练/prefill 固定使用每 GPU 一条 compute stream 和一条 communication stream。
两条 stream 是所有模型级 workload 共同使用的执行契约，不是可配置的 stream 池。
相邻两个 microbatch 的固定顺序为：

```text
Compute: Attention0 -> Router0 -> Attention1 -> Router1
                                      -> Expert0 -> Expert1 -> Reduce0 -> Reduce1

Comm:               Dispatch0 -> Dispatch1 -> Combine0 -> Combine1

Overlap:             D0 || A1
                                  D1 || E0
                                                C0 || E1
                                                         C1 || R0
```

`streams.py` 在完整 task graph 上增加顺序边。每个 rank 的 compute task 严格串行；
每个 rank 的 communication phase 严格串行；一个 phase 内拆出的多条 flow/chunk
仍并行。通信 phase 的完成 event 连接到真正消费数据的 Expert/Reduce，独立 compute
不等待通信。

当 `num_layers > 1` 时，compute stream 在层边界使用 wavefront：

```text
L0 Final0 -> L1 Attention0/Router0 -> L0 Final1 -> L1 Attention1/Router1
```

`L1 Attention0` 只保留 `L0 Final0` 的真实同 microbatch 数据依赖，不再等待
`L0 Final1`。这使 `L1 MB0 Attention || L0 MB1 Combine` 成为合法跨层
overlap。单 compute stream 和单 communication stream 的各自串行约束不变。

`micro_batches=N` 不表示同时展开 N 路流水线，而是按相邻两个进行 double-buffer
分组：

```text
group 0: MB0 + MB1
group 1: MB2 + MB3
group 2: MB4 + MB5
...
```

前一组必须 drain，即该组覆盖的完整 workload 全部完成，后一组才能启动；组与组
之间没有计算或通信 overlap。`N` 为奇数时，最后一个 microbatch 单独组成尾组。

“完整 workload”不能一律写成“完整模型”，必须以 builder 的建模范围为准：

- 当前 `build_transformer_workload()` 生成 `num_layers` 个同构 forward MoE block，
  因此组间边界是两个 microbatch 都完成这些 block；
- 将来完整 DSV3 builder 生成全部 Transformer 层时，组间边界就是两个 microbatch
  都完成完整 DSV3 prefill，包括全部层及最终任务；
- 不允许在前一组仍位于中间层时提前启动下一组，从而把当前双缓冲语义误解为
  任意数量 microbatch 的 steady-state pipeline。

这些 stream 语义完全通过 predecessor/barrier 表达，不增加 `.dag` 字段，也不要求
HTSim 实现 CUDA runtime。HTSim 仍使用 network FCT 和固定 compute event 执行时间线。

当前两 microbatch builder 将第一个 microbatch 的 Attention/Router 按 132 SM
计算；后续 microbatch 的 Attention/Router 因可与前一 microbatch 通信重叠，按
112 SM 计算。Expert task 同样按 112 SM 计算。该选择是生成时的静态 schedule
假设，不会根据实际 FCT 动态切换。

## 8. Training 扩展

训练不能只把 forward 反向播放。至少需要：

- combine backward，本质为使用 forward plan 的 dispatch；
- expert MLP backward；
- dispatch backward，本质为 combine；
- router/gate backward；
- MoonEP duplicated expert grad reduce；
- TP/DP 参数梯度 collective；
- activation checkpoint/recompute；
- optimizer step，若目标是完整 iteration。

因此首版先做 forward。训练作为独立 milestone，并要求算法文档给出 forward/backward 对称与非对称部分。

## 9. 多层与迭代

构造多个 block 时不应简单复制 task ID。模型生成器需要：

```text
global_task_key = (iteration, microbatch, pipeline_stage, layer, op, rank, shard)
```

然后稳定映射为整数 task ID。

对于同构 layer，可选择：

- 完整展开，得到真实跨层并发；
- 只生成一个代表 block；
- 按 layer group 采样并在报告中外推。

不能只把单 block makespan 乘 layer 数来替代存在 pipeline、cache 或跨层 routing 相关性的 workload。

## 10. 输出校验

模型级生成完成后至少检查：

- 每层每 rank 的输入/输出 token 数守恒；
- 每 token 恰有 K 个逻辑 routed expert，除非配置允许 drop；
- expert ID 和 rank placement 合法；
- dispatch 与 combine 使用同一个 algorithm plan；
- shared/routed 分支在 merge 前全部完成；
- 每个 compute task 都记录 `operation_flops`、单卡峰值和理论 `compute_us`；
- overlap compute task 使用 112/132 SM，普通 compute task 使用 132/132 SM；
- 同一 rank、同一逻辑通信 phase 的多条 flow 只计一次 20 SM；
- 同一 rank 的 compute stream 不存在 task overlap；
- 同一 rank 的 communication phase 不存在 phase overlap；
- 同一 phase 内独立 flow 保持并行；DeepEP local leg 只等待对应 fabric leg；
- double-buffer schedule 保留四个预期 compute/communication overlap 窗口；
- 多层 schedule 保留 `Layer N+1 / MB0 Attention || Layer N / MB1 Combine`
  跨层 overlap；
- 下一层 MB0 Attention 不得依赖上一层 MB1 Final；
- task graph 无环；
- HTSim barrier 映射没有未声明的交叉等待或保守串行化；
- 最终 `.dag` 能通过 parser dry-run。

## 11. 首版示例目标

统一端到端对比配置固定为：

```text
2 representative DSV3 MoE layers
prefill forward
TP=1, PP=1, EP=32
4 servers x 8 GPUs
2 microbatches
每 rank 每 microbatch 4096 tokens（smoke 为 2）
H=7168, E=256, top-k=8
连续 expert placement
seed 0 的 balanced-permuted router assignment
NCCL、DeepEP、EPLB、MoonEP 与 ProbeEP 五种输出
```

该配置使用 plane=1、8 Leaf、4 Spine、400 Gbps RDMA 的专用测试拓扑。NCCL 验证
rank-direct All-to-All 和 Spine 流量；DeepEP 验证 rank/server 两级去冗余以及显式
RDMA/NVLink hierarchy legs；EPLB 验证持久 hierarchical placement 和稳态 DeepEP
transport；MoonEP 在每个 expert home server 内独立规划 replica，并复用 DeepEP
跨服务器 transport。该 MoonEP 组合是本项目的核心算法抽象，不声称官方实现提供
多节点 RDMA。ProbeEP 在 MoonEP/DeepEP 语义上增加跨服务器临时专家迁移和
单进程 HTSim 内的 Dispatch FCT 闭环反馈。

## 12. 硬件规格来源

- [NVIDIA H100 产品规格](https://www.nvidia.com/en-us/data-center/h100/)：H100 SXM BF16/FP16 Tensor Core 稀疏峰值为 `1,979 TFLOP/s`，规格页脚注标明使用 sparsity。
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)：H100 SXM 使用 132 SM，BF16 的 dense/sparse 理论口径约为 `1,000/2,000 TFLOP/s`；本文使用量产规格对应的 dense 值 `989 TFLOP/s`。
