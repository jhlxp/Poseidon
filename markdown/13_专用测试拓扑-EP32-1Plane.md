# 专用测试拓扑：EP32 / 1 Plane

## 1. 用途

本拓扑是 MpRail 后续功能测试的统一物理基准。测试脚本不得自行改用其他
rank、plane、Leaf、Spine 或链路速率组合；算法级纯 Python 单元测试不受此限制。

这里固定 `plane=1` 只是为了控制功能测试规模，不是 MpRail 的能力上限。
仿真器仍支持 `plane=1..8`；每增加一个 plane，就增加一套完全独立的
8-Leaf/8-Spine scale-out fabric。

## 2. 固定配置

| 参数 | 固定值 |
|---|---:|
| GPU/rank | 32 |
| server | 4 |
| GPU/server | 8 |
| plane | 1 |
| rail | 8 |
| L0 Leaf | 8 |
| L1 Spine | 8 |
| GPU RDMA 链路 | 400 Gbps |
| L0-L1 链路 | 400 Gbps |
| 每个 L0-L1 pair 的 bundle | 1 |
| server-local FullMesh | 3200 Gbps |

## 3. Rail 与 Leaf 映射

rail 由 GPU 在服务器内的 local index 决定：

```text
server_id = rank // 8
rail_id = rank % 8
L0_id = rail_id
```

每个 Leaf 恰好连接四张 GPU，每台服务器贡献一张：

| Leaf/rail | server 0 | server 1 | server 2 | server 3 |
|---:|---:|---:|---:|---:|
| 0 | rank 0 | rank 8 | rank 16 | rank 24 |
| 1 | rank 1 | rank 9 | rank 17 | rank 25 |
| 2 | rank 2 | rank 10 | rank 18 | rank 26 |
| 3 | rank 3 | rank 11 | rank 19 | rank 27 |
| 4 | rank 4 | rank 12 | rank 20 | rank 28 |
| 5 | rank 5 | rank 13 | rank 21 | rank 29 |
| 6 | rank 6 | rank 14 | rank 22 | rank 30 |
| 7 | rank 7 | rank 15 | rank 23 | rank 31 |

同服务器 GPU 之间不进入 Leaf-Spine，直接使用服务器内部 FullMesh。

## 4. Leaf-Spine 连接

单 plane 内，8 个 L0 Leaf 与 8 个 L1 Spine 完全连接：

```text
for leaf in 0..7:
    for spine in 0..7:
        L0[leaf] <-> L1[spine] at 400 Gbps
```

每台 L0 有 4 个 400 Gbps GPU-facing 端口和 8 个 400 Gbps spine-facing
端口。每张 GPU 只有一个 400 Gbps fabric 端口，因此单 rank 外部带宽为
400 Gbps，不再按多个 plane 聚合。

这两句只描述本测试。若配置 `plane=8`，每张 GPU 在每个 plane 各有一个
400 Gbps 端口，聚合 scale-out 带宽为 `8 * 400 = 3200 Gbps`。

## 5. 三类测试路径

固定使用以下端点验证路径分支：

```text
same server: rank 0 -> rank 1
same rail:   rank 0 -> rank 8
cross rail:  rank 0 -> rank 9
```

- `0 -> 1` 只走 server-local FullMesh。
- `0 -> 8` 走 `L0[rail 0]`，不经过 L1。
- `0 -> 9` 走 `L0[rail 0] -> L1 -> L0[rail 1]`。

## 6. HTSim 参数

```bash
-topology mprail \
-mprail_planes 1 \
-mprail_gpus_per_server 8 \
-mprail_l1_eps_per_plane 8 \
-mprail_l0_l1_links_per_spine 1 \
-linkspeed 400000 \
-local_linkspeed 3200000
```

`.cm` 必须声明 `Nodes 32`。DAG 模式继续使用 `Connections 0` 的空 `.cm`
提供 rank 总数。

## 7. 验收条件

- 启动日志必须报告 `servers=4`、`rails=8`、`planes=1`、`l0=8`、`l1=8`。
- Host-L0 与 L0-L1 物理链路速率必须为 400 Gbps。
- 每个 Leaf 的四个 rank 必须来自四台不同服务器。
- same-server、same-rail、cross-rail 三类路径均按第 5 节执行。
- cross-rail 只能经过一个 L1，不能出现 L0-L0 直连。
- 所有 L0-L1 链路都留在 plane 0。
- flow ECMP、oblivious spray 和 ecmp_rr 在单 plane 内对 8 个 Spine 做负载均衡。

## 8. 标准 DSV3 两层 workload

本拓扑后续观察 MoE workload、网络路径和计算通信 overlap 时，固定使用以下代表性
案例。它把 workload 和物理拓扑放在同一份测试契约中，不再维护另一份集成案例
文档。

官方 DeepSeek-V3 有 61 个 Transformer layer，前三层使用 dense FFN。为了观察 EP
通信，本例抽取两个代表性的 DSV3 MoE layer，并在测试 DAG 中编号为 layer 0 和
layer 1。这里是测试内逻辑编号，不是官方模型的物理层号。

| 参数 | 值 |
|---|---:|
| 模型口径 | DeepSeek-V3 representative MoE layers |
| layer | 2 |
| microbatch | 2 |
| hidden | 7168 |
| MoE expert hidden | 2048 |
| routed expert | 256 |
| expert/rank | 8，round-robin placement |
| top-k | 8 |
| attention head | 128 |
| sequence length | 4096 |
| dispatch dtype | FP8 |
| combine dtype | BF16 |
| 算法 | NCCL、DeepEP、EPLB、MoonEP |
| 功能回归 | 2 tokens/rank/microbatch，chunk 32 |
| 完整测试 | 4096 tokens/rank/microbatch，chunk 4096 |
| 完整测试 compute config | `pysrc/compute_profiles/H100_DSV3_EP32_compute_4096tpr.json` |

这里的 `4096` 是**每个 rank、每个 microbatch 的本地 token 数**，不是 32 rank
合计值。32 rank 时，每个 microbatch 的全局 token 数为：

```text
4096 tokens/rank * 32 ranks = 131072 tokens
```

每个 token 选择 top-8 expert，因此进入 routing 的 expert assignment 总数为：

```text
131072 tokens * 8 = 1048576 routes / microbatch
```

两个 microbatch 依次通过两个 MoE layer，因此完整 DAG 包含四次 MoE invocation：

```text
MoE invocations = 2 microbatches * 2 layers = 4
unique input tokens = 4096 * 32 * 2 microbatches = 262144
layer-token executions = 4096 * 32 * 2 microbatches * 2 layers = 524288
expert route executions = 524288 * topk 8 = 4194304
```

这里 `unique input tokens` 不因经过两层而重复计数；`layer-token executions` 和
`expert route executions` 会按层计数，因为每层都重新执行 Router、Dispatch、Expert
和 Combine。

DeepEP normal-kernel 的 internode/intranode 测试把 `--num-tokens` 默认设为 4096，
并由每个分布式 rank 分别创建 `[num_tokens, hidden]` 本地输入，所以本项目将正常
训练/prefill 口径明确解释为 `4096 tokens/rank/microbatch`。层数可以为了观察缩成
2 层，但完整 workload 不应再缩小 token 数、通信字节或 expert route 数；
2-token 只是单独的 smoke 功能回归口径。

完整测试使用 `chunk_tokens=4096`，把同一 `(src_rank, dst_rank)` 的 payload 聚合成大
flow，以控制 DAG task 数。聚合不会减少 token、route 或 `transfer_bytes`；HTSim
仍对大 flow 内部的 packet 执行逐包网络仿真。

uniform routing 下，每个 source/destination rank pair 每次 invocation 聚合 1024 个
token payload。FP8 Dispatch 和 BF16 Combine 的单 flow 字节分别为：

```text
dispatch flow = 1024 * hidden 7168 * 1 byte = 7340032 bytes
combine flow  = 1024 * hidden 7168 * 2 bytes = 14680064 bytes
```

下面的 task/字节数是 **DeepEP uniform full case** 的固定参考值。不同算法的
payload 去重、placement、prefetch 和 planning 语义不同，不要强制它们与 DeepEP
具有相同 task 数或字节数。DeepEP 每个 phase 排除 `src_rank == dst_rank` 的本地
payload，因此网络 task 数为：

```text
rank pairs per phase = 32 * 31 = 992
network phases = 2 directions * 2 microbatches * 2 layers = 8
transfer tasks = 992 * 8 = 7936
compute tasks = 32 ranks * 4 compute modules * 2 microbatches * 2 layers = 512
total tasks = 7936 + 512 = 8448
```

固定 uniform routing 下，四次 MoE invocation（2 layers x 2 microbatches）应生成：

```text
transfer tasks = 7936
logical transfer bytes = 87375740928
```

完整测试中 DeepEP 对这两个数做验收；其他算法使用各自的 manifest、
route 和总字节验收规则。

模型参数参考官方
[DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)。
DeepEP 训练/prefill 口径参考本机 `DeepEP/docs/legacy.md` 中的 hidden 7168、top-8、
FP8 dispatch、BF16 combine 和 400 Gbps RDMA 配置。

## 9. 两层 DAG 和双 stream

每个 GPU 固定一条 compute stream 和一条 communication stream。两个 microbatch
构成唯一的 double-buffer group，每层顺序为：

```text
Compute: A0 -> R0 -> A1 -> R1 -> E0 -> E1 -> Reduce0 -> Reduce1
Comm:              D0 -> D1 -> C0 -> C1

Overlap: D0||A1, D1||E0, C0||E1, C1||Reduce0
```

layer 1 的 Attention 必须等待同 microbatch 的 layer 0 Combine/Reduce。当前每张 GPU
的 compute stream 使用 layer-major 顺序，即该 GPU 上两个 microbatch 的 layer 0
任务排在 layer 1 之前；层间不额外增加全局 barrier，DeepEP communication 仍等待
该 invocation 的真实 Router/stream 前驱。两个 microbatch 完成 layer 1 的全部 rank
任务后，整个 workload 才 drain。

若以后把 `micro_batches` 增加到 4，则执行：

```text
(MB0, MB1) 完成两个 layer并全局 drain
    -> (MB2, MB3) 才启动
```

## 10. 建模边界

当前两层 builder 已经建模：

- 两层之间的真实 DAG 依赖；
- Attention、Router、Dispatch、Expert、Combine/Reduce；
- DeepEP destination-rank 去冗余和 destination-side `server_forward`；
- 每层的双 microbatch 计算通信 overlap；
- HTSim 的 flow FCT、MpRail 拥塞和 400 Gbps 链路；
- JSON 中选择的模块级固定计算时间。

当前 builder 暂不建模：

- DSV3 前三个 dense layer 和完整 61 层；
- MLA、RMSNorm、residual、shared expert 的逐 kernel 实现；
- layer-specific 真实 router trace；当前两层复用 deterministic uniform routing；
- backward、梯度通信、CUDA runtime、动态 SM 和 HBM 竞争。

因此“DSV3”表示采用 DSV3 的核心 MoE shape 和通信口径，不表示当前 DAG 已覆盖
官方完整模型的所有 kernel。计算 JSON 的格式和 profiling 切换方法见
[15_计算时间JSON配置.md](15_计算时间JSON配置.md)。

## 11. 运行四算法测试与产物

```bash
cd /home/xuheng/EP_ExpertTrans
python3 tests/run_dsv3_2layer_algorithms.py

# 仅在明确要求“完整测试”时使用
python3 tests/run_dsv3_2layer_algorithms.py --full --workers 4
```

默认入口是 `2 tokens/rank/microbatch` smoke，用于日常功能回归。`--full`
才使用 4096tpr JSON 和完整通信字节。四个算法默认由 4 个独立进程
并行仿真；`--workers` 可限制并发数。

每次运行生成独立目录：

```text
test_logs/run_<timestamp>_dsv3_2layer_4algo_<smoke|full>/
├── 配置.json
├── 构建.log
├── 测试报告.md
├── summary.json
├── dsv3_algorithm_comparison.html
└── algorithms/
    └── <nccl|deepep|eplb|moonep>/
        ├── workload/
        │   ├── workload.dag
        │   ├── nodes.cm
        │   ├── manifest.json
        │   └── task_map.json
        ├── simulation/
        │   ├── htsim.log
        │   ├── htsim.dat
        │   └── output_metrics/
        ├── timeline/
        │   ├── dag_gpu_timeline.html
        │   ├── dag_task_timeline.csv
        │   ├── dag_rank_overlap_summary.csv
        │   └── dag_timeline_summary.json
        └── link_load/
            ├── mprail_link_load_by_layer.png
            ├── mprail_link_load_summary.csv
            └── mprail_endpoint_load_summary.csv
```

HTML 默认只展示 server 0 的 rank 0-7，仿真仍执行全部 32 rank。减少
展示 rank 只是为了观察，不改变 workload 或网络竞争。四算法和每个
`Task details` 均可折叠；每条 timeline 默认 Fit 全局，可水平缩放并实时显示
`100 px` 对应的微秒数。

## 12. DSV3 两层验收条件

1. 每个算法 manifest 必须报告 `num_ranks=32`、`num_layers=2`、
   `micro_batches=2`。
2. smoke 必须报告 `tokens_per_rank=2`、`chunk_tokens=32`；full 必须报告
   `tokens_per_rank=4096`、`chunk_tokens=4096`。
3. stream manifest 必须报告每 rank 一条 compute、一条 communication stream。
4. `mb0.layer1.attention.rank0` 必须等待 layer 0 的同 rank Combine/Reduce。
5. workload 必须包含跨服务器和跨 rail transfer；NCCL 不得使用
   `server_forward`，DeepEP/EPLB/MoonEP 必须使用。
6. full compute manifest 必须记录 4096tpr JSON 路径和 `selected_source`。
7. 四个 HTSim 进程必须完成各自 manifest 中全部 task/barrier。
8. 两个 layer 上的参与 GPU 集合中都必须实际出现 `D0||A1`、`D1||E0`、
   `C0||E1`、`C1||Reduce0`；不要求八个窗口固定落在同一张 GPU。
9. 每个算法的 HTML、task CSV、rank overlap CSV、timeline summary、链路负载 PNG
   和链路/endpoint summary 必须全部生成。
10. 总览 HTML 必须包含四个可折叠算法区域，并能进入每个算法的可缩放
    timeline 和 MpRail 链路负载图。
