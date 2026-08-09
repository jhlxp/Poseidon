# DAG 工作负载生成总体设计

## 1. 结论

MoE workload 生成器与 HTSim/MpRail 仿真器应当解耦。

生成器负责：

- 描述模型、并行布局、expert placement 和 router 输出；
- 把 NCCL、DeepEP、EPLB、MoonEP、ProbeEP 等算法展开成计算与通信任务图；
- 用内置 H100 理论模型，或显式选择的 H100/H20 schema-v2 JSON，把算子降低成固定 `compute_us`；
- 根据 tensor 形状、数据类型、去重、padding 和转发方案计算 `transfer_bytes`；
- 最终输出 HTSim 可读取的 `.dag` 和空 `.cm`。

HTSim 负责：

- 根据 `.dag` 的依赖启动任务；
- 对 network task 执行真实 UEC/MpRail 网络仿真；
- 对 compute task 执行固定时长事件；
- 输出 FCT、链路负载和 DAG makespan。

两者不是零耦合，而是只通过一个稳定的文本契约耦合：rank 编号、`.dag`
字段、动态 `APPEND/OBSERVE/CLOSE` 行协议、route 语义和时间/字节单位。
生成器不能依赖 HTSim 内部的 C++ 类；静态和动态模式都只传输文本 task。

## 2. 两层架构

```text
第二层：模型到 workload
  模型结构 + 并行配置 + batch/sequence + router trace + 硬件 cost
                            |
                            v
  Transformer block / iteration 的逻辑任务图
                            |
                            v
第一层：MoE 算法建模
  MoE invocation + expert placement + token routes + algorithm config
                            |
                            v
  NCCL / DeepEP / EPLB / MoonEP / ProbeEP 的 TaskGraph fragment
                            |
                            v
共享 lowering
  task-level IR -> barrier DAG + empty CM + manifest
                            |
                            v
HTSim / MpRail
```

第一层可以独立使用：给定一次 MoE invocation，直接生成仅包含 router、dispatch、expert、combine 的小型 workload。

第二层复用第一层：当前构造 Attention、Router、MoE 和 Reduce 组成的
代表性 forward block，再将多个 layer 和 microbatch 拼接起来。Norm、Residual、
完整 DSV3 kernel 和 backward 仍属于未实现边界。

## 3. 当前代码目录

Python 代码位于仓库根目录的 `pysrc/`：

```text
pysrc/
├── generate_moe_dag.py
├── moe_dag/
│   ├── __init__.py
│   ├── schema.py
│   ├── graph.py
│   ├── cost.py
│   ├── gate.py
│   ├── load_profile.py
│   ├── emitter.py
│   ├── dynamic.py
│   ├── algorithms/
│   │   ├── common.py
│   │   ├── nccl.py
│   │   ├── deepep.py
│   │   ├── eplb.py
│   │   ├── moonep.py
│   │   └── probeep.py
│   └── models/
│       ├── transformer.py
│       ├── incremental.py
│       └── streams.py
workload/
└── gate/
tests/
├── run_workload_generator.py
├── run_dynamic_dag_functional.py
└── run_probeep_2layer_ratio_full.py
```

`schema.py` 保存 placement、routing assignment 和 invocation；`graph.py` 是
task-level IR；`emitter.py` 负责 barrier lowering 和五类输出；算法与模型层不依赖
HTSim C++ 类。

当前 CLI 示例：

```bash
python3 pysrc/generate_moe_dag.py \
  --output generated_workloads/deepep_demo \
  --algorithm deepep \
  --num-ranks 16 --gpus-per-server 8 \
  --num-experts 16 --topk 8 \
  --tokens-per-rank 128 --micro-batches 2 \
  --chunk-tokens 32
```

## 4. 输入与输出

### 4.1 输入

生成一次 workload 至少需要四类输入：

| 输入 | 主要内容 |
|---|---|
| 模型规格 | hidden、FFN hidden、head、layer、expert、top-k、激活和 dtype |
| 并行与 placement | DP/TP/PP/EP、rank 到 server 的映射、expert 到 rank 的映射 |
| token routing | 每个 source rank 的 token 选择了哪些 expert；来自 trace 或合成分布 |
| 硬件 cost | 内置 H100 smoke 理论模型，或显式 H100/H20 schema-v2 per-token JSON |

算法配置是第一层的额外输入。本阶段代码接受：

- `nccl` rank-direct/no-dedup forward；
- `deepep` 训练/prefill forward 核心路径；
- `eplb` hierarchical placement、确定性 replica selection 和 DeepEP 稳态传输；
- dispatch、combine、weight dtype 与 `chunk_tokens`；
- `moonep` 的 per-server replica planning、`replicas_per_rank` 和 `token_padding`。
- `probeep` 的独立 per-server baseline、跨服务器临时 replica、route/weight chunk、
  多 NIC 权重迁移和通信掩盖预算。

EPLB 已作为 `--algorithm eplb` 接入。它根据 estimated expert loads 生成一段
placement epoch 共用的 physical expert mapping，再为当前 token routes 确定 execution
rank，并复用 DeepEP 的两级去冗余和分层 transport。稳态 DAG 不包含 planner 或 weight
migration task；当前 CLI 可直接传负载快照，未传时使用第一次 invocation 的 route
count 作为静态代理。

DeepEP/MoonEP/ProbeEP backward、zero-copy 和真实 kernel profiling 尚未
实现，CLI 不提供对应开关。DeepEP low-latency/decode 不属于当前项目范围。

### 4.2 输出

每次生成应输出一个完整、可复现的目录：

```text
generated_workloads/<name>/
├── workload.dag
├── nodes.cm
├── manifest.json
├── task_map.json
└── 生成报告.md
```

| 文件 | 用途 |
|---|---|
| `workload.dag` | 静态 HTSim 的计算通信任务输入；动态模式保存所有已 append task 的累计审计快照，最后一次 append 后即完整，不要求等待仿真结束 |
| `nodes.cm` | `Nodes N`、`Connections 0`，满足当前 DAG CLI 契约 |
| `manifest.json` | 完整输入、随机种子、版本、算法配置和 cost 来源 |
| `task_map.json` | task 与 barrier 到模型 layer、kernel、通信 phase 的映射 |
| `生成报告.md` | 中文汇总 token、字节、计算时间、关键假设和降级项 |

`.dag` 本身不携带足够的算法语义，因此 `manifest.json` 和 `task_map.json` 不是可选调试文件，而是可审计性的必要组成。

## 5. 核心设计选择

### 5.1 共享语义，不共享固定流程

不能让每个算法从头实现 token、expert、rank 和 byte 计算，否则同一模型在不同算法间不可比较。

也不能定义一个固定的 `dispatch -> expert -> combine` 模板后只替换路由。NCCL
保留全部 top-k payload 并直达 expert rank；MoonEP 包含 per-server planning proxy、
动态冗余 expert 和权重预取，并复用 DeepEP scale-out transport；DeepEP 按
destination rank 和 destination server 两级去重，通过同 index relay 显式完成
RDMA、local fanout 和 local reduce。
这些不是普通 all-to-all 的参数变化。

因此采用：

```text
共享：MoE 数学语义、placement、routing trace、IR、cost 接口、去冗余策略接口、DAG emitter
独立：每个算法的 planner、TaskGraph fragment、去冗余开关/scope、复制规则、通信聚合规则
```

### 5.2 Task graph 直接映射到 barrier DAG

生成器内部使用 task 级前驱，输出时再分配 `barrier_id`。原因是算法中存在：

- attention 与另一个 microbatch 的 dispatch 重叠；
- MoonEP planning、prefetch 与 dispatch 的组合；
- 分块转发和多条并行通信；
- shared expert 与 routed expert 分支后汇合。

HTSim 的 `barrier_id` 不是时间阶段。默认让每个 task 使用独立 barrier 后，任意 task edge 都可以无损转换为 predecessor barrier edge：

```text
task A -> task B
barrier(A) -> barrier(B)
```

多个 task 只有在启动前驱完全相同，并且所有消费者都应等待它们全部完成时，才能显式合并到同一个 barrier。emitter 不按可视化时间层做 levelize，也不能为了减少 barrier 数而引入交叉等待。

### 5.3 每 GPU 两条逻辑 stream

模型层为每张 GPU **固定**一条 compute stream 和一条 communication stream。这是
当前项目的 workload 执行契约，不是算法参数，也不提供改变 stream 数量的配置。
NCCL、DeepEP、EPLB、MoonEP 和 ProbeEP 都复用该模型层契约；算法插件只负责向
`TaskGraph` 追加各自的 MoE task fragment。

stream 不是新的 `.dag` 字段，而是在生成阶段降低为普通 predecessor edges：

```text
compute stream: Attention/Router -> Attention/Router -> Expert -> Expert -> Reduce -> Reduce
comm stream:    Dispatch 0 -> Dispatch 1 -> Combine 0 -> Combine 1
```

相邻两个 microbatch 组成一个 double-buffer group。`micro_batches=N` 的固定分组为：

```text
N=1: (MB0)
N=2: (MB0, MB1)
N=3: (MB0, MB1) -> (MB2)
N=4: (MB0, MB1) -> (MB2, MB3)
```

每组使用两条 stream 做组内 overlap；前一组在所有 rank 上的完整 workload 全部完成
后，下一组才启动，组与组之间不 overlap。这里的“完整 workload”由 `num_layers`
决定：`num_layers=1` 表示两个 microbatch 都走完一个 block；`num_layers=2` 表示都
走完两个 block。未来完整 DSV3 builder 覆盖全部真实层和最终任务时，同一语义才
表示两个 microbatch 走完整 DSV3 prefill。奇数个 microbatch 的最后一个单独成组。

同一个 Dispatch/Combine phase 内的 destination flow 和 chunk 保持并行；不同通信
phase 只在实际参与的 rank 上按 comm stream tail 串行。compute producer、
communication phase 和 compute consumer 之间使用普通 DAG edge 表达 CUDA event
语义。

HTSim 不感知或调度 CUDA stream；它只执行 lowering 后的 barrier DAG。这样既禁止
同一 GPU 上两个 compute task 错误重叠，也保留 `D0||Attention1`、`D1||Expert0`、
`C0||Expert1` 和 `C1||Reduce0`。

多层 workload 在 double-buffer group 内使用跨层 wavefront，不按 layer 整体 drain：

```text
Layer N / MB0 Final
    -> Layer N+1 / MB0 Attention+Router
    -> Layer N / MB1 Final
    -> Layer N+1 / MB1 Attention+Router
```

同一 microbatch 的下一层 Attention 仍必须等待本 microbatch 上一层的
Combine/Final；MB0 下一层 Attention 不等待 MB1 上一层尾部。因此可形成：

```text
compute stream: Layer N+1 / MB0 Attention
comm stream:    Layer N   / MB1 Combine
overlap:        cross-layer Attention || Combine
```

通信 phase 仍在单条 communication stream 上串行；跨层 wavefront 只新增合法的
compute/communication overlap，不允许 compute/compute 或 communication/communication
重叠。

### 5.4 Compute 在生成时降低为固定时长

HTSim 中的 compute task 始终是固定 `compute_us`。生成器支持两种写入来源：未传
JSON 时使用内置 H100 理论 FLOP 公式，或者显式读取模块级 H100/H20 JSON per-token 时间。JSON 同时保存
`theoretical_us_per_token` 和 `profiled_us_per_token`，通过 `selected_source`
二选一，再乘当前 task 的实际 token 数，格式见
[16_计算时间JSON配置.md](16_计算时间JSON配置.md)。

未传 JSON 时，生成器按单张 H100 SXM 的 dense BF16 Tensor Core 峰值换算。普通
计算使用完整峰值；明确与通信 kernel 重叠的计算使用固定 SM 预留后的峰值：

```text
peak_flops_per_gpu = 989e12 FLOP/s
h100_sms = 132
communication_sms = 20
overlap_compute_sms = 112
overlap_peak_flops = 989e12 * 112 / 132 ~= 839.15e12 FLOP/s

compute_us_normal  = operation_flops / 989e12 * 1e6
compute_us_overlap = operation_flops / 839.15e12 * 1e6
```

`communication_sms=20` 是本项目的首版仿真假设，不是 DeepEP 对所有模式和拓扑的固定值。它按“每个 rank 上的逻辑通信 kernel/phase”预留一次；同一个 dispatch/combine phase 拆出的多个 flow 或 chunk 共享这 20 SM，不能按 flow 数重复扣减。

传入 JSON 后使用 `compute_us = token_count * selected_us_per_token`，不再按 FLOP
或 overlap SM 比例缩放；`total_sms`/`communication_sms` 只记录资源假设。
`expert_ffn` 的 token_count 是当前 execution rank 的 routed expert tokens，MoonEP
使用 padding 后 tokens，因此热点和均衡算法会真实改变 FFN task 时长。

换算或读取结果写入 `.dag` 后就是固定时长事件。HTSim 不做运行时 SM 调度，也不
动态恢复 SM。network task 的 FCT 仍完全由 UEC/MpRail 决定。

该占位仍不考虑 Tensor Core 实际利用率、HBM、kernel launch、融合、padding、warp、抢占或 cache，因此只是简化理论下界。profiling 和完整 GPU 资源模型留到后续开发。

## 6. 当前 HTSim 能力边界

| 需求 | 当前状态 | 首版处理 |
|---|---|---|
| rank 间网络传输 | 已支持 | network task |
| 同服务器 rank 间传输 | 已支持高速 FullMesh | 普通 local flow 或显式 local route |
| 固定完整路径 | 已支持 | `explicit` route |
| 严格消息级服务器转发 | 已支持 | `server_forward` route |
| 任意 task 级依赖 | 已支持 | 默认每个 task 分配独立 barrier |
| 每 GPU compute/comm stream 顺序 | 已支持静态 lowering | 模型生成器增加 predecessor edges；HTSim 不做运行时 stream 调度 |
| 同进程动态 DAG 追加 | ProbeEP 闭环使用 | `-dag_control`；HTSim 端整批校验后原子 append，observation 点不推进模拟时间 |
| chunk 级流水转发 | 已支持表达 | 显式拆成多个 network task，每个 flow 完成后释放自己的 barrier |
| 单 flow 内部流式事件 | 不支持且首版不需要 | 不监听第 N 个 packet；需要时拆 chunk flow |
| GPU compute | 固定时长事件 | 内置 H100 smoke 公式，或显式 H100/H20 JSON theoretical/profiled 二选一 |
| 通信/计算 SM 竞争 | 静态近似 | 每 rank 的逻辑通信 phase 固定预留 20/132 SM |
| HBM/NVLink memory contention | 未支持 | 不进入首版 compute 占位，报告中标注 |
| multicast/one-to-many 原语 | 无专用原语 | DeepEP 用一个 fabric task 加多个 local fanout task 表达 |

`server_forward` 仍支持通用三阶段 store-and-forward，但 DeepEP/EPLB/MoonEP
不再用它封装算法传输。算法 builder 显式输出 fabric、local fanout/local reduce
task，使一份服务器级 RDMA payload 可以被多个 rank 共享。`chunk_tokens` 通过多个
完整 flow task 表达 chunk overlap，单条 flow 内 packet-progress 不建模。

MpRail 的 server-local FullMesh 也只是对 NVLink rank-to-rank 数据移动的网络级近似。UEC flow 的可靠传输、ACK 和拥塞控制不等价于 NVLink load/store、TMA 或 symmetric-memory 语义。首版可以研究字节、依赖和本地带宽竞争，但不能把理论 compute 占位或节点内 FCT 宣称为真实 DeepEP/MoonEP kernel latency。

## 7. 开发顺序

1. 实现 schema、placement、routing assignment、task-level graph 和确定性 ID。
2. 实现 HTSim emitter、manifest 和静态校验。
3. 实现 NCCL direct/no-dedup 与 DeepEP hierarchical scaleout/scaleup 模型。
4. 实现单个 MoE sublayer：router、dispatch、expert、combine。
5. 实现完整 transformer block forward：attention、residual、norm、MoE。
6. 实现 H100 dense BF16 峰值和通信预留 20 SM 的固定时长占位。
7. 实现 per-server MoonEP planning、weight prefetch 和 DeepEP scale-out 组合。
8. 实现 EPLB hierarchical placement、稳态 replica selection 和 DeepEP transport。
9. 再增加训练 backward 和 gradient reduce。

## 8. 首版非目标

- 不解释或执行真实 PyTorch/CUDA kernel。
- 不从模型权重推断 router 决策；router trace 必须输入或合成。
- 不在生成器中复制 HTSim 的链路/FCT 计算。
- 不宣称理论 FLOP 模型等价于真实 kernel profiling。
- 不在首版实现 profiling、roofline、HBM 或动态 SM 资源调度；SM 只使用固定 20/132 预留。
- 不支持单条 flow 内部按 packet 到达释放依赖；chunk pipeline 使用显式 flow task。
- 不把所有 CUDA microkernel 都拆成 DAG task；拆分粒度以依赖、资源类型和可测 cost 是否不同为准。
- 不在第一版同时支持任意 DP/TP/PP/EP 组合。

统一算法验收使用 EP32、4 servers x 8 GPUs、plane=1、400 Gbps：NCCL 直达真实
rank，DeepEP 使用 hierarchical scaleout/scaleup transport，EPLB 使用持久 hierarchical
placement，MoonEP 在每个 expert home server 内独立创建 replica 并复用 DeepEP
scale-out transport，ProbeEP 使用单 HTSim 动态闭环。五者共享相同 routing
assignment digest、expert placement 和 dtype；有效性能对比不得用静态 ProbeEP 代替动态反馈结果。
