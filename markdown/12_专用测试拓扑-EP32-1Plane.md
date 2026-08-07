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
