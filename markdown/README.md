# EP workload 与 MpRail 仿真文档

本目录描述两部分：Python workload 生成器的设计，以及当前 MpRail/HTSim 的拓扑、DAG 执行与测试契约。旧的光交换和历史实验叙事不再作为当前实现基线。

## 文档

| 文档 | 内容 |
|---|---|
| [01_DAG工作负载生成总体设计.md](01_DAG工作负载生成总体设计.md) | Python 生成器的两层架构、目录、输入输出和能力边界 |
| [02_第一层-MoE算法工作负载建模.md](02_第一层-MoE算法工作负载建模.md) | 共享 MoE IR、算法插件、payload 去冗余策略、字节核算和 barrier 映射 |
| [03_第二层-模型到DAG生成.md](03_第二层-模型到DAG生成.md) | Transformer block、router、多层 workload，以及理论公式/JSON 固定计算时间 |
| [04_算法建模-NCCL.md](04_算法建模-NCCL.md) | NCCL MoE All-to-Allv、无 payload 去冗余、真实目标 rank 直达和验收边界 |
| [05_算法建模-DeepEP.md](05_算法建模-DeepEP.md) | 训练/prefill 的 rank/server 两级去冗余及显式 RDMA/NVLink hierarchy legs |
| [06_算法建模-EPLB.md](06_算法建模-EPLB.md) | estimated load、hierarchical expert replication/placement 和 placement epoch 边界 |
| [07_算法建模-MoonEP.md](07_算法建模-MoonEP.md) | per-server 动态 expert replica、权重预取和 DeepEP scale-out 组合 |
| [08_算法建模-ProbeEP.md](08_算法建模-ProbeEP.md) | 两阶段计算均衡、反馈式 NIC 窗口、跨服务器 replica 与多 NIC 分块 RDMA |
| [09_MpRail拓扑开发文档.md](09_MpRail拓扑开发文档.md) | 拓扑定义、端口模型、路由、配置和开发边界 |
| [10_DAG任务与执行模型.md](10_DAG任务与执行模型.md) | `.cm` 纯 flow 与 `.dag` 一体化 workload、barrier 依赖和完成语义 |
| [11_MpRail源路由与服务器转发.md](11_MpRail源路由与服务器转发.md) | CM/DAG 显式路径、服务器内部 relay 转发、校验与完成语义 |
| [12_测试与日志规范.md](12_测试与日志规范.md) | Python 功能测试分类及 `test_logs/` 产物结构 |
| [13_MpRail链路负载可视化.md](13_MpRail链路负载可视化.md) | link-load 采样、七面板吞吐图、坐标解析和统计口径 |
| [14_专用测试拓扑-EP32-1Plane.md](14_专用测试拓扑-EP32-1Plane.md) | 32 GPU、8 Leaf/4 Spine 单 plane 拓扑及 DSV3 两层五算法 smoke/full 测试 |
| [15_DAG执行时间线可视化.md](15_DAG执行时间线可视化.md) | GPU 0-31 可缩放 timeline、Gate/expert before-after、折叠明细和五算法合并页 |
| [16_计算时间JSON配置.md](16_计算时间JSON配置.md) | 模块级 theoretical/profiled 固定时间、二选一规则、文件格式和适用边界 |
| [17_Gate分布与Routing生成.md](17_Gate分布与Routing生成.md) | Gate provider、UltraEP/FAST 合成分布、raw receive 精确 quota 和 routing fidelity |

## 仿真器的两种输入模式

MpRail 当前支持两种互斥的仿真模式：

| 模式 | 输入 | 表达内容 | 启动方式 |
|---|---|---|---|
| 静态 flow 仿真 | `.cm` | 纯网络 flow：源、目的、开始时间和字节数 | 每条 flow 按自己的 `start` 启动 |
| 一体化 workload 仿真 | `.dag` | network task、compute task、barrier 并行和前驱依赖 | task 随所属 barrier 启动 |

`.dag` 是完整的计算通信任务序列，但其中每个 network task 在底层仍会动态创建一条 UEC flow。两种模式不能混用：启用 `-dag` 时，`.cm` 的 `Connections` 必须为 0。目前仍需要这个空 `.cm` 提供 `Nodes` 总数。

## 当前仿真目标

MpRail 是完全独立的多平面两层 Leaf-Spine 网络：

```text
GPU -> server-local FullMesh -> plane-specific L0-EPS
    -> same-plane L1-EPS -> destination L0-EPS -> GPU
```

- 统一测试目标为 1 个 plane、8 条 rail、8 台 L0 Leaf 和 4 台 L1 Spine。
- rail 由 GPU 在服务器内的 local index 定义；4 台 8-GPU server 时每条 rail 挂 4 张 GPU。
- 每张 GPU 只有一条 400 Gbps RDMA fabric 链路。
- `plane=1` 仅是测试基准；仿真器支持最多 8 个完全独立的并行 scale-out plane。
- 每个 plane 内，所有 rail 的 L0-EPS 与该 plane 的全部 L1-EPS 全连接。
- 不存在跨 plane 交换机链路，不存在光交换语义。
- 同服务器 GPU 之间使用独立的高速 FullMesh 本地路径。

固定 rank/Leaf 映射和测试参数见
[14_专用测试拓扑-EP32-1Plane.md](14_专用测试拓扑-EP32-1Plane.md)。

实现顺序固定为：文档契约、拓扑与路由、DAG 接入、Python 功能测试。

## 生成一个 workload

```bash
python3 pysrc/generate_moe_dag.py \
  --output generated_workloads/deepep_demo \
  --algorithm deepep \
  --num-ranks 16 --gpus-per-server 8 \
  --num-experts 16 --tokens-per-rank 128 \
  --topk 8 --micro-batches 2 --chunk-tokens 32
```

输出包含 `workload.dag`、空 `nodes.cm`、`manifest.json`、`task_map.json` 和
`生成报告.md`。模型层固定把相邻两个 microbatch 按每 GPU 一条 compute stream、
一条 communication stream 做 double-buffer lowering；两条 stream 是固定契约，
不能配置为其他数量。`micro_batches=N` 按 `(MB0, MB1)`、`(MB2, MB3)` 依次分组，
前一组完成生成范围内的完整 workload 后下一组才启动，组间不 overlap，奇数尾项
单独成组。完整 DSV3 workload 中，这意味着每组两个 microbatch 走完全部 DSV3 层
后才进入下一组。组内多层使用 wavefront，允许 `Layer N+1 / MB0 Attention`
与 `Layer N / MB1 Combine` 重叠；同 MB 数据依赖和两条 stream 各自串行不变。
stream 顺序最终表现为普通 predecessor barriers。生成器功能与
HTSim 集成测试运行：

EPLB 稳态 workload 示例：

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

```bash
python3 tests/run_workload_generator.py
```

## 绘制 MpRail 链路负载

```bash
python3 visualization/mprail_link_load.py \
  --metrics-dir <run-dir>/output_metrics \
  --output-dir <run-dir>/visualization
```

采样开关、五类面板和统计语义见
[13_MpRail链路负载可视化.md](13_MpRail链路负载可视化.md)。

## 绘制 DAG 执行时间线

```bash
python3 visualization/dag_timeline.py \
  --workload-dir <generator-output-dir> \
  --htsim-log <case-dir>/htsim.log \
  --output-dir <case-dir>/dag_timeline \
  --gpus-per-server 8
```

输出每 GPU `Compute / Network TX / Network RX` 的 HTML 时间线，以及逐 task
FCT/bytes 和逐 GPU overlap 汇总。ProbeEP 的离线 planner 不进入 timeline；专家权重
NVLink/RDMA task 直接画入对应 GPU 的 `Network TX/RX`，用独立颜色区别于 token
dispatch/combine，不建立额外全局 lane。当前 ProbeEP 的 CPU task/stream 计数均为 0。
完整口径见
[15_DAG执行时间线可视化.md](15_DAG执行时间线可视化.md)。

Gate 分布和算法执行前后专家负载可单独生成：

```bash
python3 visualization/gate_load_profile.py \
  --workload-dir <generator-output-dir> \
  --output-dir <case-dir>/gate_load
```

页面以 microbatch 为选择单位：Before 固定画第一层 raw Gate baseline，After 固定画
最后一层 admitted execution placement；专家迁移数量、server-pair 和 per-NIC 链路负载
也只取最后一层，表示当前实验最接近收敛的状态。分布语义和 provider 配置见
[17_Gate分布与Routing生成.md](17_Gate分布与Routing生成.md)。

## 运行 DSV3 两层五算法案例

```bash
python3 tests/run_dsv3_2layer_algorithms.py
# 只有明确要求完整测试时：
python3 tests/run_dsv3_2layer_algorithms.py --full --workers 4
```

该入口固定使用两个代表性 DSV3 MoE layer、两个 microbatch、
NCCL/DeepEP/EPLB/MoonEP/ProbeEP 和 EP32 单 plane/400 Gbps 拓扑。默认为
`2 tokens/rank/microbatch` smoke；`--full` 才是 `4096 tokens/rank/microbatch`。
每个算法都生成一个完整 `algorithm_dashboard.html`，其中包含 Gate/expert
before-after、GPU 0-31 完整可缩放 timeline 和 MpRail 链路负载图；五个算法页面
再组合到 ZIP 根目录的可折叠 `dsv3_algorithm_comparison.html` 总入口，并生成不含仿真大文件的
`dsv3_visualization_bundle.zip` 供下载。ZIP 成功后服务器 run 目录不保留散装
HTML，其他产物仍保留。参数、边界和验收条件见
[14_专用测试拓扑-EP32-1Plane.md](14_专用测试拓扑-EP32-1Plane.md)，计算时间选择见
[16_计算时间JSON配置.md](16_计算时间JSON配置.md)。

ProbeEP ratio controller 的 2-layer 单进程动态 DAG 实验使用：

```bash
# 日常功能回归：2 tokens/rank
python3 tests/run_probeep_2layer_ratio_full.py --gate-layer-map 0,1

# 明确运行完整实验时：4096 tokens/rank
python3 tests/run_probeep_2layer_ratio_full.py \
  --full --gate-layer-map 0,1 \
  --compute-config pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json
```

该入口固定为 EP32、1 plane、400 Gbps/NIC、raw receive layers 0-1，只运行
ProbeEP。它在同一 HTSim PID 内观测 `Weight+Dispatch` FCT、更新 0.90 目标
预算并追加下一层 DAG，不使用 baseline/replay。预算更新使用全局
`alpha*Cmax/Nmax` 缩放瓶颈 NIC 实际 Token+Weight 总字节，以
`Cmax*400Gbps` 封顶，再扣除不可裁剪的 token Dispatch baseline；controller 只限制
Expert Weight migration。完整实验默认读取
`H20_DSV3_EP32_compute_4096tpr.json`；H100 只在显式传入对应 profile 时使用。
`probeep_2layer_dynamic_visualization.zip` 保留 GPU 0-31 全部 timeline lanes、Gate、
链路负载、4 个 Dispatch control observations 和 4 个 Combine telemetry 汇总；
服务器目录不保留散装 HTML。专项入口会拒绝 A/M state-chain 串线或同 source rail 上
Dispatch TX 早于 remote Weight RDMA TX 的结果，并分别报告 remote RDMA 与 local NVLink
expert weight。
