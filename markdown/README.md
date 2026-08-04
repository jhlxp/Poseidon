# EP workload 与 MpRail 仿真文档

本目录描述两部分：Python workload 生成器的设计，以及当前 MpRail/HTSim 的拓扑、DAG 执行与测试契约。旧的光交换和历史实验叙事不再作为当前实现基线。

## 文档

| 文档 | 内容 |
|---|---|
| [01_DAG工作负载生成总体设计.md](01_DAG工作负载生成总体设计.md) | Python 生成器的两层架构、目录、输入输出和能力边界 |
| [02_第一层-MoE算法工作负载建模.md](02_第一层-MoE算法工作负载建模.md) | 共享 MoE IR、算法插件、字节核算、冗余通信和 barrier 映射 |
| [03_第二层-模型到DAG生成.md](03_第二层-模型到DAG生成.md) | Transformer block、router、H100 理论计算占位、20 SM 通信预留和多层 workload |
| [04_算法建模-DeepEP.md](04_算法建模-DeepEP.md) | DeepEP V1/V2、hybrid/direct、domain 去重和多级转发 |
| [05_算法建模-MoonEP.md](05_算法建模-MoonEP.md) | 动态冗余 expert、perfect balance、权重预取、combine 和 grad reduce |
| [06_MpRail拓扑开发文档.md](06_MpRail拓扑开发文档.md) | 拓扑定义、端口模型、路由、配置和开发边界 |
| [07_DAG任务与执行模型.md](07_DAG任务与执行模型.md) | `.cm` 纯 flow 与 `.dag` 一体化 workload、barrier 依赖和完成语义 |
| [08_MpRail源路由与服务器转发.md](08_MpRail源路由与服务器转发.md) | CM/DAG 显式路径、服务器内部 relay 转发、校验与完成语义 |
| [09_测试与日志规范.md](09_测试与日志规范.md) | Python 功能测试分类及 `test_logs/` 产物结构 |

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

- 默认目标为 8 个 plane，每个 plane 独立。
- 一个 rail 包含每个 plane 各一台 L0-EPS；8-plane 时一个 rail 有 8 台 L0-EPS。
- 一个 rail 下挂可配置数量的服务器。
- 每个 plane 内，所有 rail 的 L0-EPS 与该 plane 的全部 L1-EPS 全连接。
- 不存在跨 plane 交换机链路，不存在光交换语义。
- 同服务器 GPU 之间使用独立的高速 FullMesh 本地路径。

实现顺序固定为：文档契约、拓扑与路由、DAG 接入、Python 功能测试。

## 生成一个 workload

```bash
python3 pysrc/generate_moe_dag.py \
  --output generated_workloads/deepep_demo \
  --algorithm deepep-hybrid \
  --num-ranks 16 --gpus-per-server 8 \
  --num-experts 16 --tokens-per-rank 128 \
  --topk 8 --micro-batches 2 --chunk-tokens 32
```

输出包含 `workload.dag`、空 `nodes.cm`、`manifest.json`、`task_map.json` 和
`生成报告.md`。生成器功能与 HTSim 集成测试运行：

```bash
python3 tests/run_workload_generator.py
```
