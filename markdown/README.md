# MpRail 仿真器文档

本目录只描述当前 MpRail 拓扑、DAG 执行模型和测试规范。旧的光交换、动态 placement 和历史实验叙事不再作为当前实现基线。

## 文档

| 文档 | 内容 |
|---|---|
| [MpRail拓扑开发文档.md](MpRail拓扑开发文档.md) | 拓扑定义、端口模型、路由、配置和开发边界 |
| [DAG任务与执行模型.md](DAG任务与执行模型.md) | `.cm` 纯 flow 与 `.dag` 一体化 workload、stage 依赖和完成语义 |
| [MpRail源路由与服务器转发.md](MpRail源路由与服务器转发.md) | CM/DAG 显式路径、服务器内部 relay 转发、校验与完成语义 |
| [测试与日志规范.md](测试与日志规范.md) | Python 功能测试分类及 `test_logs/` 产物结构 |

## 两种输入模式

MpRail 当前支持两种互斥的仿真模式：

| 模式 | 输入 | 表达内容 | 启动方式 |
|---|---|---|---|
| 静态 flow 仿真 | `.cm` | 纯网络 flow：源、目的、开始时间和字节数 | 每条 flow 按自己的 `start` 启动 |
| 一体化 workload 仿真 | `.dag` | network task、compute task、stage 并行和前驱依赖 | task 随所属 stage 启动 |

`.dag` 是完整的计算通信任务序列，但其中每个 network task 在底层仍会动态创建一条 UEC flow。两种模式不能混用：启用 `-dag` 时，`.cm` 的 `Connections` 必须为 0。目前仍需要这个空 `.cm` 提供 `Nodes` 总数。

## 当前目标

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
