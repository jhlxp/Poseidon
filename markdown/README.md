# EP workload 与 MpRail 仿真文档

本目录描述两部分：Python workload 生成器的设计，以及当前 MpRail/HTSim 的拓扑、DAG 执行与测试契约。旧的光交换和历史实验叙事不再作为当前实现基线。

## 文档

| 文档 | 内容 |
|---|---|
| [01_DAG工作负载生成总体设计.md](01_DAG工作负载生成总体设计.md) | Python 生成器的两层架构、目录、输入输出和能力边界 |
| [02_第一层-MoE算法工作负载建模.md](02_第一层-MoE算法工作负载建模.md) | 共享 MoE IR、算法插件、payload 去冗余策略、字节核算和 barrier 映射 |
| [03_第二层-模型到DAG生成.md](03_第二层-模型到DAG生成.md) | Transformer block、router、H100 理论计算占位、20 SM 通信预留和多层 workload |
| [04_算法建模-NCCL.md](04_算法建模-NCCL.md) | NCCL MoE All-to-Allv、无 payload 去冗余、真实目标 rank 直达和验收边界 |
| [05_算法建模-DeepEP.md](05_算法建模-DeepEP.md) | 训练/prefill 的 destination-rank 去重和目标端两段转发 |
| [06_算法建模-MoonEP.md](06_算法建模-MoonEP.md) | per-server 动态 expert replica、权重预取和 DeepEP scale-out 组合 |
| [07_MpRail拓扑开发文档.md](07_MpRail拓扑开发文档.md) | 拓扑定义、端口模型、路由、配置和开发边界 |
| [08_DAG任务与执行模型.md](08_DAG任务与执行模型.md) | `.cm` 纯 flow 与 `.dag` 一体化 workload、barrier 依赖和完成语义 |
| [09_MpRail源路由与服务器转发.md](09_MpRail源路由与服务器转发.md) | CM/DAG 显式路径、服务器内部 relay 转发、校验与完成语义 |
| [10_测试与日志规范.md](10_测试与日志规范.md) | Python 功能测试分类及 `test_logs/` 产物结构 |
| [11_MpRail链路负载可视化.md](11_MpRail链路负载可视化.md) | link-load 采样、七面板吞吐图、坐标解析和统计口径 |
| [12_专用测试拓扑-EP32-1Plane.md](12_专用测试拓扑-EP32-1Plane.md) | 32 GPU、单 plane、8 Leaf/8 Spine、400 Gbps RDMA 的统一测试基准 |

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

- 统一测试目标为 1 个 plane、8 条 rail、8 台 L0 Leaf 和 8 台 L1 Spine。
- rail 由 GPU 在服务器内的 local index 定义；4 台 8-GPU server 时每条 rail 挂 4 张 GPU。
- 每张 GPU 只有一条 400 Gbps RDMA fabric 链路。
- `plane=1` 仅是测试基准；仿真器支持最多 8 个完全独立的并行 scale-out plane。
- 每个 plane 内，所有 rail 的 L0-EPS 与该 plane 的全部 L1-EPS 全连接。
- 不存在跨 plane 交换机链路，不存在光交换语义。
- 同服务器 GPU 之间使用独立的高速 FullMesh 本地路径。

固定 rank/Leaf 映射和测试参数见
[12_专用测试拓扑-EP32-1Plane.md](12_专用测试拓扑-EP32-1Plane.md)。

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
`生成报告.md`。生成器功能与 HTSim 集成测试运行：

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
[11_MpRail链路负载可视化.md](11_MpRail链路负载可视化.md)。
