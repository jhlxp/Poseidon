# CM Flow 与 DAG 一体化任务模型

## 1. 两种模式的边界

### 1.1 `.cm`：静态纯网络 flow

`.cm` 是 connection matrix。它只描述网络 flow，不描述计算任务或 stage 依赖。

```text
Nodes 16
Connections 3
0->1 id 1 start 0 size 16384
0->4 id 2 start 1000000 size 1048576
0->8 id 3 start 2000000 size 1048576
```

字段含义：

| 字段 | 含义 |
|---|---|
| `Nodes` | rank/GPU 端点总数 |
| `Connections` | 后续静态 flow 的数量 |
| `src->dst` | 源 rank 和目的 rank |
| `id` | flow ID；建议显式提供，MpRail 用它选择 plane、spine 和 bundle |
| `start` | flow 启动时刻，当前解析器单位为皮秒 |
| `size` | flow payload 字节数 |

同一 `start` 的多条 flow 可以并发，但它们之间没有 stage barrier。`.cm` 适合验证 FCT、拥塞、链路利用率以及同服务器、同 rail、跨 rail 路径。

运行示例：

```bash
htsim_uec \
  -topology mprail \
  -tm workload.cm \
  -end 1000
```

### 1.2 `.dag`：计算通信一体化 workload

`.dag` 同时描述：

- network task；
- compute task；
- 同一 stage 内的并行任务；
- stage 之间的前驱依赖和 barrier。

它适合表达一次完整训练迭代、kernel/通信交叠或多个通信阶段。DAG 中的 network task 不是另一套网络模型：task 启动时，仿真器仍会动态创建一条 UEC flow，并通过 MpRail 传输。

## 2. 为什么 DAG 还需要空 `.cm`

当前主程序先从 `.cm` 读取 `Nodes`，再加载 `.dag`。因此 DAG 模式仍需提供一个没有静态 flow 的 connection matrix：

```text
Nodes 16
Connections 0
```

运行示例：

```bash
htsim_uec \
  -topology mprail \
  -tm empty.cm \
  -dag workload.dag \
  -end 1000
```

这是当前 CLI 的输入限制，不代表 `.cm` 参与 DAG 调度。启用 `-dag` 时：

- `.cm` 的 `Connections` 必须为 0；
- 不允许静态 `.cm` flow 和 DAG network task 混合；
- `Nodes` 决定 DAG 中 rank 的合法范围。

## 3. DAG 行格式

每个非注释行只表示一个 network task 或一个 compute task。第五个 route 组仅在需要显式路径或服务器内部转发时添加：

```text
task_id stage_id | src_rank dst_rank | transfer_bytes compute_us | predecessor_stages... [| route_spec]
```

| 字段 | 含义 |
|---|---|
| `task_id` | 正整数且全局唯一；network task 同时把它作为 flow ID |
| `stage_id` | task 所属 stage |
| `src_rank` | network task 的源 rank；compute task 的计算 rank |
| `dst_rank` | network task 的目的 rank；compute task 重复填写同一个计算 rank |
| `transfer_bytes` | network task 的传输字节数；compute task 写 `0` |
| `compute_us` | compute task 持续时间，单位微秒；network task 写 `0` |
| `predecessor_stages` | 一组前驱 stage ID，以空格分隔；根 stage 写 `-` |

前三个 `|` 是必需的字段组分隔符，分别把一行划分为：

```text
任务身份 | 通信端点/计算位置 | 传输量/计算时间 | 前驱 stage 集合
```

合法 network task：

```text
10 0 | 0 8 | 1048576 0 | -
```

合法 compute task：

```text
20 0 | 4 4 | 0 25 | -
```

每行必须恰好满足一种任务类型：

```text
network = src_rank != dst_rank and transfer_bytes > 0 and compute_us == 0
compute = src_rank == dst_rank and transfer_bytes == 0 and compute_us > 0
```

不允许一行同时包含网络和计算，也不允许两者同时为空。

network task 可选的第五组支持：

```text
explicit rank:0 l0:r0:p3:b0 l1:p3:s1:b1 l0:r1:p3 rank:8
server_forward src_relay:3 dst_relay:11
```

完整语法、拓扑校验和完成语义见 [MpRail源路由与服务器转发.md](MpRail源路由与服务器转发.md)。compute task 不允许携带 route 组。

## 4. Stage 与 barrier 语义

- 没有前驱的 root stage 在仿真时间 0 启动。
- 同一 stage 的全部 task 在同一模拟时刻启动。
- network task 的完成时间由 UEC 在 MpRail 上产生的真实 FCT 决定。
- compute task 在固定 `compute_us` 后完成。
- stage 在最后一个 task 完成时结束。
- 后继 stage 只在它的全部前驱 stage 完成后启动。
- 同一 stage 的所有 task 必须声明完全相同的前驱 stage 集合。

因此：

```text
T_stage_done = max(T_task_0_done, T_task_1_done, ..., T_task_n_done)
```

## 5. 完整 DAG 示例

```text
# stage 0：两条通信和两个计算同时开始
1 0 | 0 8 | 16384 0 | -
2 0 | 0 9 | 16384 0 | -
3 0 | 0 0 | 0 20 | -
4 0 | 1 1 | 0 30 | -

# stage 1：全部任务等待 stage 0 完成
5 1 | 8 0 | 16384 0 | 0
6 1 | 8 8 | 0 10 | 0
```

在当前功能测试的 MpRail 参数下，三条 network task 都早于同 stage 的最长 compute task 完成，实际观测时间线为：

```text
time 0 us:  stage 0 的 task 1/2/3/4 同时启动
time 30 us: stage 0 最后一个 task 完成，stage 1 启动
time 40 us: stage 1 完成，整个 DAG 完成
```

这组具体时间不是 DAG 格式的固定结果。如果网络 FCT 超过 compute 时间，stage 会继续等待 network task，后继 stage 的启动时刻相应推迟。

一个 source 向多个 destination 通信时，应为每个 destination 写独立 network task。一个 stage 也可以包含多个 compute task；每条 compute task 通过相等的 `src_rank` 和 `dst_rank` 表示计算归属。

`predecessor_stages` 是 stage ID 集合，不是 task ID 集合。例如：

```text
7 2 | 8 8 | 0 15 | 0 1
```

表示 task 7 属于 stage 2，在 rank 8 计算 15 微秒；stage 2 必须同时等待 stage 0 和 stage 1 完成。同一个 stage 内的所有 task 必须填写相同的前驱集合。

## 6. 与 MpRail 的关系

- DAG 本身不直接指定 plane 或 spine。
- network task 提供 `src_rank`、`dst_rank`、`transfer_bytes` 和 task/flow ID。
- `-load_balancing_algo ecmp` 下，MpRail 根据 flow ID 稳定选择 source plane；UEC 固定 `pathid`，因此一个 network task 使用一条 flow-level ECMP 路径。
- `-load_balancing_algo oblivious` 下，UEC 逐包改变 `pathid`，NIC 在可用 plane port 间调度；一个 network task 即可使用全部 8 个 plane。
- L0/L1 在 packet 所属 plane 内，根据 `(flow_id, pathid, switch_salt)` 对 L1-EPS 和 bundle 执行 ECMP。
- 同一 stage 的多个 network task 在两种模式下都可以并发执行并产生网络竞争。
- compute task 不进入网络拓扑。

MpRail 的网络竞争由 Queue 和 UEC 决定。compute task 中相等的 `src_rank`/`dst_rank` 当前只用于校验 rank 并记录计算归属，不影响计算完成时间。compute task 首版是固定时长事件，不模拟 SM occupancy、kernel 抢占、同 GPU kernel 排队或不同 kernel 之间的资源竞争。

## 7. 失败条件

以下输入必须在仿真开始前失败：

- task ID 为 0 或重复；
- network 和 compute 同时存在；
- network 和 compute 同时为空；
- 缺少任意一个 `|` 分隔符；
- 前驱组为空，既没有 `-` 也没有 stage ID；
- `src_rank == dst_rank` 却声明了传输字节数；
- `src_rank != dst_rank` 却声明了计算时间；
- rank 超出 `.cm` 中 `Nodes` 指定的范围；
- 同一 stage 的 task 声明不同前驱；
- 前驱 stage 不存在；
- stage 图存在环。

事件循环结束时 DAG 尚未完成也必须返回失败。
