# 专用测试拓扑：EP32 / 1 Plane

## 1. 用途

本拓扑是 MpRail 后续功能测试的统一物理基准。测试脚本不得自行改用其他
rank、plane、Leaf、Spine 或链路速率组合；算法级纯 Python 单元测试不受此限制。

这里固定 `plane=1` 只是为了控制功能测试规模，不是 MpRail 的能力上限。
仿真器仍支持 `plane=1..8`；每增加一个 plane，就增加一套完全独立的
8-Leaf/4-Spine scale-out fabric。

## 2. 固定配置

| 参数 | 固定值 |
|---|---:|
| GPU/rank | 32 |
| server | 4 |
| GPU/server | 8 |
| plane | 1 |
| rail | 8 |
| L0 Leaf | 8 |
| L1 Spine | 4 |
| GPU RDMA 链路 | 400 Gbps |
| L0-L1 链路 | 400 Gbps |
| 每个 L0-L1 pair 的 bundle | 1 |
| server-local FullMesh / rank local injection | 7200 Gbps |
| MTU | 4150 bytes |
| 每个有向链路 queue | 128 packets = 531200 bytes，约 518.75 KiB |
| ECN low（实际 marking threshold） | 4 packets = 16600 bytes = queue 的 3.125% |
| ECN high（仅解析/校验） | 13 packets = 53950 bytes = queue 的 10.15625% |

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
这里采用 H100 SXM 每 GPU `900 GB/s = 7200 Gbps` 双向聚合 NVLink 带宽。
它是每 GPU 聚合上限，不是 7 个本地 peer 各自都能同时获得 7200 Gbps；仿真器用
独立的 7200 Gbps local injection 资源约束一个 rank 的全部本地流量。

## 4. Leaf-Spine 连接

单 plane 内，8 个 L0 Leaf 与 4 个 L1 Spine 完全连接：

```text
for leaf in 0..7:
    for spine in 0..3:
        L0[leaf] <-> L1[spine] at 400 Gbps
```

每台 L0 有 4 个 400 Gbps GPU-facing 端口和 4 个 400 Gbps spine-facing
端口。下行 GPU 聚合带宽与上行 Spine 聚合带宽均为 1.6 Tbps，因此本测试拓扑
不超卖。每张 GPU 只有一个 400 Gbps fabric 端口，因此单 rank 外部带宽为
400 Gbps。

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
-mprail_l1_eps_per_plane 4 \
-mprail_l0_l1_links_per_spine 1 \
-linkspeed 400000 \
-local_linkspeed 7200000 \
-mtu 4150 \
-q 128
```

`.cm` 必须声明 `Nodes 32`。DAG 模式继续使用 `Connections 0` 的空 `.cm`
提供 rank 总数。

## 7. 验收条件

- 启动日志必须报告 `servers=4`、`rails=8`、`planes=1`、`l0=8`、`l1=4`。
- Host-L0 与 L0-L1 物理链路速率必须为 400 Gbps。
- server-local pair link 与每 rank local injection 必须为 7200 Gbps，且与
  400 Gbps RDMA NIC 独立。
- 每个 Leaf 的四个 rank 必须来自四台不同服务器。
- same-server、same-rail、cross-rail 三类路径均按第 5 节执行。
- cross-rail 只能经过一个 L1，不能出现 L0-L0 直连。
- 所有 L0-L1 链路都留在 plane 0。
- flow ECMP、oblivious spray 和 ecmp_rr 在单 plane 内对 4 个 Spine 做负载均衡。

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
| expert/rank | 8，contiguous placement |
| top-k | 8 |
| attention head | 128 |
| sequence length | 4096 |
| dispatch dtype | FP8 |
| combine dtype | BF16 |
| 算法 | NCCL、DeepEP、EPLB、MoonEP、ProbeEP |
| 功能回归 | 2 tokens/rank/microbatch，chunk 32 |
| 完整测试 | 4096 tokens/rank/microbatch，chunk 4096 |
| 完整测试 compute config | H100/H20 schema-v2 profile 显式二选一，默认 H20 |
| ProbeEP migration planning | 离线生成决策，不进入 DAG；论文单独分析复杂度/实现开销 |
| ProbeEP CPU runtime task | 0；`cpu_task_count=0`、`cpu_streams_global=0` |
| ProbeEP memory/slot admission | 关闭；ProbeEP 配置和 manifest 不包含 slot/replica-capacity 字段 |

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

完整测试使用 `chunk_tokens=4096`，把同一 hierarchy leg/rank pair 的 payload 聚合成
大 flow，以控制 DAG task 数。聚合不会减少 token、expert route 或应传 payload；
HTSim 仍对大 flow 内部的 packet 执行逐包网络仿真。

256 个专家按 index 连续放置，每个 rank 持有连续 8 个专家。合成 router 使用固定
种子 0 的 expert permutation，并按 TopK 分组循环：总 expert route 严格均衡，同时
允许同 token 的多个专家落在同 rank 或同 server，用于真实验证 DeepEP 去冗余。

不同算法的 task 和逻辑总字节不应强制相同：NCCL 保留 route multiplicity；DeepEP
显式计算 rank payload、server/RDMA payload 和目标服务器 local fanout/reduce；EPLB
与 MoonEP 还会改变 execution placement；ProbeEP 进一步加入跨服务器 replica 和
分块权重迁移。完整测试通过每次 invocation 的以下关系
验收去冗余，而不使用旧 uniform/round-robin 的固定 task 数：

```text
route_count > unique_token_payload_count > unique_server_payload_count
```

Manifest 的 `hierarchical_transfer` 必须分别给出四类 leg 的 task、payload 和字节，
链路负载图再验证 fabric RDMA 与 server-local FullMesh 的实际吞吐。

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

layer 1 的 Attention 必须等待同 microbatch 的 layer 0 Combine/Reduce，但 MB0
的 layer 1 Attention 不等待 MB1 的 layer 0 Combine/Reduce。每张 GPU 在层边界
使用 wavefront：

```text
L0 Reduce0 -> L1 Attention0/Router0 -> L0 Reduce1 -> L1 Attention1/Router1

Cross-layer overlap:
L1 MB0 Attention || L0 MB1 Combine
```

通信 phase 仍在单 communication stream 上串行，因此不会引入跨层通信 phase
互相 overlap。两个 microbatch 完成 layer 1 的全部 rank 任务后，整个
double-buffer group 才 drain。

若以后把 `micro_batches` 增加到 4，则执行：

```text
(MB0, MB1) 完成两个 layer并全局 drain
    -> (MB2, MB3) 才启动
```

## 10. 建模边界

当前两层 builder 已经建模：

- 两层之间的真实 DAG 依赖；
- Attention、Router、Dispatch、Expert、Combine/Reduce；
- DeepEP destination-rank/server 两级去冗余和显式 RDMA/NVLink hierarchy legs；
- 每层的双 microbatch 计算通信 overlap；
- HTSim 的 flow FCT、MpRail 拥塞和 400 Gbps 链路；
- JSON 中选择的模块级固定计算时间。

当前 builder 暂不建模：

- DSV3 前三个 dense layer 和完整 61 层；
- MLA、RMSNorm、residual、shared expert 的逐 kernel 实现；
- 原始逐 token router trace；默认使用 deterministic `balanced_permuted`，也可显式
  选择 uniform、UltraEP-style、FAST-style 或 raw receive CDF provider；
- backward、梯度通信、CUDA runtime、动态 SM 和 HBM 竞争。

因此“DSV3”表示采用 DSV3 的核心 MoE shape 和通信口径，不表示当前 DAG 已覆盖
官方完整模型的所有 kernel。计算 JSON 的格式和 profiling 切换方法见
[16_计算时间JSON配置.md](16_计算时间JSON配置.md)。

## 11. 运行五算法测试与产物

有效性能对比由四个静态算法 case 和一个动态 ProbeEP case 组成。两条
入口必须使用相同的硬件 profile、Gate provider、seed 和 layer map：

```bash
cd /home/xuheng/EP_ExpertTrans
# 日常 2-token 静态功能回归
python3 tests/run_dsv3_2layer_algorithms.py \
  --algorithms nccl,deepep,eplb,moonep

# 日常 2-token ProbeEP 单进程动态回归
python3 tests/run_probeep_2layer_ratio_full.py --gate-layer-map 0,1

# 仅在明确要求“完整测试”时运行：四个静态 case
python3 tests/run_dsv3_2layer_algorithms.py \
  --full --workers 4 --algorithms nccl,deepep,eplb,moonep \
  --gate-provider raw_receive_cdf --gate-layer-map 0,1 --gate-seed 17 \
  --moonep-replicas-per-rank 256 \
  --compute-config pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json

# 同一完整对比的动态 ProbeEP case
python3 tests/run_probeep_2layer_ratio_full.py \
  --full --gate-layer-map 0,1 --gate-seed 17 \
  --compute-config pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json
```

H100 对照将两条 full 命令的 compute config 同时替换为
`H100_DSV3_EP32_compute_4096tpr.json`，其余参数保持不变。

默认都是 `2 tokens/rank/microbatch` smoke；`--full` 才使用 4096tpr JSON
和完整通信字节。静态 runner 的默认算法列表仍包含静态 ProbeEP，但它只用于
task 结构、字节和可视化回归，不能用于 ProbeEP 闭环性能结论。
`run_probeep_2layer_ratio_full.py` 不接受 baseline/replay；compute config 直接决定
同一 HTSim PID 动态时间线中的所有 compute task。

当前没有单独的、已纳入仓库的 H20/H100 十 case 统一 runner。五算法总览必须在
上述五个有效 case 完成后聚合，不能将通用 runner 的静态 ProbeEP 页面替换
动态结果。当前保留的 H20/H100 完整例证是
`test_logs/run_20260809_234100_h20_h100_2layer_5algo_full/`，其 ProbeEP 每个硬件
case 都只启动一个 HTSim。

每次运行生成独立目录：

```text
test_logs/run_<timestamp>_dsv3_2layer_<N>algo_<smoke|full>/
├── 配置.json
├── 构建.log
├── 测试报告.md
├── summary.json
├── dsv3_visualization_bundle.zip
└── algorithms/
    └── <nccl|deepep|eplb|moonep|probeep>/
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
        │   ├── dag_task_timeline.csv
        │   ├── dag_rank_overlap_summary.csv
        │   └── dag_timeline_summary.json
        ├── gate_load/
        │   ├── gate_load_profile.csv
        │   └── gate_load_profile_summary.json
        └── link_load/
            ├── mprail_link_load_by_layer.png
            ├── mprail_link_load_summary.csv
            └── mprail_endpoint_load_summary.csv
```

EP32 smoke/full 的 timeline 必须展示 rank 0-31，也就是四台服务器的全部 32 张
GPU，不能只截取 server 0。有效聚合 ZIP 的根页只列出五个算法导航入口；
真正 timeline 中的 `Task details` 仍可折叠；
每条 timeline 默认
Fit 全局，可水平缩放并实时显示 `100 px` 对应的微秒数。
`dsv3_visualization_bundle.zip` 只打包总览 HTML、五算法 Gate/expert before-after、
timeline 和链路负载产物，不包含 workload 和 simulation 大文件。服务器 run 目录
不保留散装 HTML。可用 `--gate-provider` 选择分布；同一次五算法实验必须使用完全
相同的 Gate assignment digest 序列。MoonEP 的 `replicas_per_rank` 是真实容量约束；
偏斜分布在默认值 2 下无法满足 planner 的严格 rank 均衡时会明确失败，不会静默
改变 Gate。通过 `--moonep-replicas-per-rank` 显式配置实验容量，并在 `配置.json`
中记录。
ProbeEP 不复用 MoonEP 的临时 replica 参数。当前测试假设显存足够，不建模 HBM
capacity 或全局/per-server replica 数量上限，源码和配置中也不保留虚假 slot 参数。
remote replica admission 只由每个 source NIC TX 和 destination NIC RX 的动态字节窗口
约束。

比例控制专项入口固定使用 raw receive layers 0-1 和两个 microbatch。
默认是 2-token smoke；显式 `--full` 才使用 4096tpr JSON。最终累计
`workload.dag` 快照包含
`2 layers x 2 microbatches = 4` 个 MoE invocations、4 个 `Weight+Dispatch` control
observations 和 4 个 Combine telemetry observations，整个 workload 只启动一次
HTSim。所有 observations 共享同一条连续时间线和网络状态；GPU timeline 固定
展示 rank 0-31 全部 32 条 lane。可视化写入
`probeep_2layer_dynamic_visualization.zip`，服务器目录不保留散装 HTML。

每层按四个主要 compute stage 解读：`Attention0`、
`Attention1||Weight+Dispatch0`、`Expert0||Weight+Dispatch1`、
`Expert1||Combine0`，尾部 `Combine1` 可与下一层 `Attention0` overlap。两个
Weight+Dispatch 分别进入 Attention/MoE 独立反馈链。ProbeEP 不设置
`prefetch -> dispatch` phase barrier；只保证同 source rail 的 remote Weight RDMA TX
早于该 rail 的 Dispatch fabric TX。专项报告分别给出 remote RDMA 权重和 local NVLink
权重，不能只看统一的 Expert Weight 颜色判断跨机迁移量。

专项实验使用 `-dag_control` 在同一 HTSim PID 中逐层 append DAG fragment。
MB0/MB1 `Weight+Dispatch` observation 完成时，controller 直接消费该进程的
实测 task start/done 事件；MB1 observation 后生成下一层。不使用
baseline run、external replay 或第二条仿真时间线。Combine 只作 telemetry，不进入
controller。
当前 smoke 默认 `end=5000 us`，用于容纳 ProbeEP/MoonEP 的本地权重预取；
`--simulation-end-us` 只延长仿真截止时间，不改变 token 数或 workload。

ZIP 内采用三级按需导航：

```text
dsv3_algorithm_comparison.html
  -> algorithms/<algorithm>/algorithm_dashboard.html
       -> gate_load/gate_load_profile.html
       -> timeline/dag_gpu_timeline.html
       -> link_load/mprail_link_load_by_layer.png
```

单算法 dashboard 已包含该算法的 Gate、timeline、link-load 和 CSV 入口；根页面
只展示五个算法摘要和 dashboard 链接。根页和单算法页都不包含 iframe，
也不预加载 Gate、timeline 或 PNG；只有点击对应入口后才加载一个重量模块。
所有页面依赖 ZIP 中的相对路径资源，因此必须先完整解压 ZIP。

## 12. DSV3 两层验收条件

1. 每个算法 manifest 必须报告 `num_ranks=32`、`num_layers=2`、
   `micro_batches=2`。
2. smoke 必须报告 `tokens_per_rank=2`、`chunk_tokens=32`；full 必须报告
   `tokens_per_rank=4096`、`chunk_tokens=4096`。
3. stream manifest 必须报告每 rank 一条 compute、一条 communication stream。
   ProbeEP 必须报告 `planner_runtime_model=not_in_dag`、
   `route_lowering_runtime_model=offline_not_in_dag`、`cpu_task_count=0` 和
   `cpu_streams_global=0`。
4. `mb0.layer1.attention.rank0` 必须等待 `mb0.layer0` 的同 rank
   Combine/Reduce，且不得等待 `mb1.layer0` 的 Combine/Reduce。
5. workload 必须包含跨服务器和跨 rail transfer；NCCL 不得出现 hierarchy leg，
   DeepEP/EPLB/MoonEP/ProbeEP 必须包含 dispatch/combine 的 fabric/local 四类 leg，
   且四者都不得再把算法通信封装为 `server_forward`。
6. full compute manifest 必须记录 4096tpr JSON 路径和 `selected_source`。
7. NCCL/DeepEP/EPLB/MoonEP 各使用一个静态 HTSim 进程；ProbeEP 使用一个
   持续的动态 HTSim 进程。五个 case 都必须完成各自全部 task/barrier。
8. 两个 layer 上的参与 GPU 集合中都必须实际出现 `D0||A1`、`D1||E0`、
   `C0||E1`、`C1||Reduce0`；不要求八个窗口固定落在同一张 GPU。另外
   必须实际出现 `L1 MB0 Attention || L0 MB1 Combine` 跨层窗口。
9. 每个算法的 task CSV、rank overlap CSV、timeline summary、Gate load CSV/JSON、
   链路负载 PNG 和链路/endpoint summary 必须在服务器 run 目录中保留。
10. ZIP 必须包含五个单算法 dashboard 和一个总 dashboard；总览包含五个
    算法导航入口，单算法页面能再进入覆盖 rank 0-31 的 Gate before-after、
    可缩放 timeline 和 MpRail 链路负载图；五份
    timeline summary 的 `selected_ranks` 必须均为完整 `0..31`；总览和单算法
    dashboard 不得包含 iframe 或预加载图片。
11. 可视化 ZIP 必须可完整解压，包含总览页和所有被引用资源，且不得
    包含 workload 或 simulation 目录。
12. ZIP 校验通过后，服务器 run 目录中不得留下任何 `*.html`。
13. 五算法每个 layer/microbatch 的 Gate assignment digest 必须逐项相等；每份
    before/after profile 的 total routes 和 logical expert histogram 必须守恒。
