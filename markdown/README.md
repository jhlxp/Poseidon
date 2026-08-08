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
| [08_MpRail拓扑开发文档.md](08_MpRail拓扑开发文档.md) | 拓扑定义、端口模型、路由、配置和开发边界 |
| [09_DAG任务与执行模型.md](09_DAG任务与执行模型.md) | `.cm` 纯 flow 与 `.dag` 一体化 workload、barrier 依赖和完成语义 |
| [10_MpRail源路由与服务器转发.md](10_MpRail源路由与服务器转发.md) | CM/DAG 显式路径、服务器内部 relay 转发、校验与完成语义 |
| [11_测试与日志规范.md](11_测试与日志规范.md) | Python 功能测试分类及 `test_logs/` 产物结构 |
| [12_MpRail链路负载可视化.md](12_MpRail链路负载可视化.md) | link-load 采样、七面板吞吐图、坐标解析和统计口径 |
| [13_专用测试拓扑-EP32-1Plane.md](13_专用测试拓扑-EP32-1Plane.md) | 32 GPU、8 Leaf/4 Spine 单 plane 拓扑及 DSV3 两层四算法 smoke/full 测试 |
| [14_DAG执行时间线可视化.md](14_DAG执行时间线可视化.md) | 覆盖 GPU 0-31 的可缩放 Compute/TX/RX timeline、折叠明细和四算法合并页 |
| [15_计算时间JSON配置.md](15_计算时间JSON配置.md) | 模块级 theoretical/profiled 固定时间、二选一规则、文件格式和适用边界 |

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
[13_专用测试拓扑-EP32-1Plane.md](13_专用测试拓扑-EP32-1Plane.md)。

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
[12_MpRail链路负载可视化.md](12_MpRail链路负载可视化.md)。

## 绘制 DAG 执行时间线

```bash
python3 visualization/dag_timeline.py \
  --workload-dir <generator-output-dir> \
  --htsim-log <case-dir>/htsim.log \
  --output-dir <case-dir>/dag_timeline \
  --gpus-per-server 8
```

输出每 GPU `Compute / Network TX / Network RX` 的 HTML 时间线，以及逐 task
FCT/bytes 和逐 GPU overlap 汇总。完整口径见
[14_DAG执行时间线可视化.md](14_DAG执行时间线可视化.md)。

## 运行 DSV3 两层四算法案例

```bash
python3 tests/run_dsv3_2layer_algorithms.py
# 只有明确要求完整测试时：
python3 tests/run_dsv3_2layer_algorithms.py --full --workers 4
```

该入口固定使用两个代表性 DSV3 MoE layer、两个 microbatch、
NCCL/DeepEP/EPLB/MoonEP 和 EP32 单 plane/400 Gbps 拓扑。默认为
`2 tokens/rank/microbatch` smoke；`--full` 才是 `4096 tokens/rank/microbatch`。
每个算法都为 GPU 0-31 生成完整可缩放 timeline 和 MpRail 链路负载图，最终组合到
ZIP 内的可折叠 `dsv3_algorithm_comparison.html` 入口，并生成不含仿真大文件的
`dsv3_visualization_bundle.zip` 供下载。ZIP 成功后服务器 run 目录不保留散装
HTML，其他产物仍保留。参数、边界和验收条件见
[13_专用测试拓扑-EP32-1Plane.md](13_专用测试拓扑-EP32-1Plane.md)，计算时间选择见
[15_计算时间JSON配置.md](15_计算时间JSON配置.md)。
