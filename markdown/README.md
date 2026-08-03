# MpRail 仿真器文档

本目录只描述当前 MpRail 拓扑、DAG 执行模型和测试规范。旧的光交换、动态 placement 和历史实验叙事不再作为当前实现基线。

## 文档

| 文档 | 内容 |
|---|---|
| [MpRail拓扑开发文档.md](MpRail拓扑开发文档.md) | 拓扑定义、端口模型、路由、配置和开发边界 |
| [DAG任务与执行模型.md](DAG任务与执行模型.md) | network/compute task、stage 依赖和完成语义 |
| [测试与日志规范.md](测试与日志规范.md) | Python 功能测试分类及 `logs/` 产物结构 |

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
