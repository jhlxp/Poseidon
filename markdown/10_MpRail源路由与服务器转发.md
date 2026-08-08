# MpRail 源路由与服务器转发

## 1. 目标与模式边界

MpRail 的每条 network flow 可以选择以下三种行为之一：

| 模式 | 输入中的 route 子句 | fabric 选路 |
|---|---|---|
| 普通路由 | 不写 | 继承全局 `flow-ecmp`、`oblivious-spray` 或 `ecmp-rr` |
| 完整显式路径 | `route explicit ...` | 严格使用输入指定的 plane、L1 和 bundle |
| 服务器内部转发 | `route server_forward ...` | 服务器内部两段确定性转发；中间 fabric flow 继承全局路由策略 |

`explicit` 与 `server_forward` 互斥。一条 flow 不能同时写两种 route，也不能在 `server_forward` 的 fabric 阶段继续附加显式路径。

## 2. 输入格式

### 2.1 CM

route 子句必须位于一条 connection 的末尾。

普通 flow：

```text
0->9 id 1 start 0 size 16384
```

完整显式路径：

```text
0->9 id 2 start 0 size 16384 route explicit rank:0 l0:r0:p0:b0 l1:p0:s1:b0 l0:r1:p0 rank:9
```

服务器内部转发：

```text
0->9 id 3 start 0 size 16384 route server_forward src_relay:3 dst_relay:10
```

### 2.2 DAG

DAG 在原有四个字段组后增加一个可选 route 组：

```text
task_id barrier_id | src_rank dst_rank | transfer_bytes compute_us | predecessor_barriers | route_spec
```

普通 task 不需要第五组，旧格式保持兼容：

```text
1 0 | 0 9 | 16384 0 | -
```

完整显式路径：

```text
2 0 | 0 9 | 16384 0 | - | explicit rank:0 l0:r0:p0:b0 l1:p0:s1:b0 l0:r1:p0 rank:9
```

服务器内部转发：

```text
3 0 | 0 9 | 16384 0 | - | server_forward src_relay:3 dst_relay:10
```

route 组只能用于 network task。compute task 不允许携带 route 组。

## 3. 完整显式路径

### 3.1 节点坐标

显式路径使用带类型的坐标，不使用容易冲突的裸交换机 ID：

| token | 含义 |
|---|---|
| `rank:N` | rank/GPU 端点 N |
| `l0:rR:pP` | rail R、plane P 的 L0-EPS |
| `l0:rR:pP:bB` | 同一 L0-EPS，并指定它到下一个交换机的出向 bundle B |
| `l1:pP:sS:bB` | plane P、spine S 的 L1-EPS，并指定它到下一个交换机的出向 bundle B |

`bB` 属于“当前交换机到路径中下一个交换机”的有向链路。rank 与直接连接 rank 的末端 L0 不写 bundle。

例如：

```text
rank:0
  -> l0:r0:p0:b0
  -> l1:p0:s1:b0
  -> l0:r1:p0
  -> rank:9
```

表示正向数据使用：

```text
rank 0 -> L0(r0,p0)
L0(r0,p0) -- bundle 0 --> L1(p0,s1)
L1(p0,s1) -- bundle 0 --> L0(r1,p0)
L0(r1,p0) -> rank 9
```

### 3.2 路径形状

根据端点位置，只接受三种完整形状：

```text
同服务器：rank:src rank:dst

不同服务器、同 rail：
rank:src l0:rR:pP rank:dst

跨 rail：
rank:src l0:rSRC:pP:bUP l1:pP:sS:bDOWN l0:rDST:pP rank:dst
```

首版不接受省略节点、额外绕行、重复交换机或跨 plane 跳转。跨 rail 路径必须完整写出源 L0、L1 和目的 L0。

### 3.3 校验规则

仿真开始前必须完成以下校验：

- 首尾 `rank` 必须与 flow 的 `src_rank`、`dst_rank` 完全一致。
- L0 的 rail 必须分别匹配源、目的 rank 所在 rail。
- 同一条路径上的全部 L0/L1 必须属于同一 plane。
- plane、rail、spine 和 bundle 坐标必须在当前 MpRail 配置范围内。
- 只有后继是另一台交换机时才允许、也必须写 bundle。
- 显式路径只能在 `-topology mprail` 下使用。

显式路径覆盖全局负载均衡。该 flow 的数据 packet、重传 packet 和控制反馈都不能因为全局 `spray` 或 `RR` 改到另一条物理路径。

### 3.4 反向路径

用户只写正向路径。ACK、NACK、PULL 等反向控制 packet 的路径由实现自动反推：

- 节点顺序反转；
- 每条交换机链路使用对应的反方向；
- 反向链路沿用其正向物理 pair 的 bundle 编号。

上面的例子自动得到：

```text
rank:9 l0:r1:p0:b0 l1:p0:s1:b0 l0:r0:p0 rank:0
```

## 4. 服务器内部转发

### 4.1 字段

```text
server_forward src_relay:S dst_relay:D
```

逻辑 flow 的源、目的仍由 CM 的 `src->dst` 或 DAG 的 `src_rank dst_rank` 决定。两个新增字段只指定服务器内的 relay rank：

- `src_relay` 必须与逻辑源 rank 位于同一服务器；
- `dst_relay` 必须与逻辑目的 rank 位于同一服务器；
- 逻辑源和逻辑目的必须位于不同服务器。

### 4.2 严格串行状态机

一个逻辑 flow 被展开为最多三个等字节数的 UEC subflow：

```text
Phase 1 src_local: src_rank -> src_relay
等待全部 transfer_bytes 完成并收到完成确认

Phase 2 fabric: src_relay -> dst_relay
等待全部 transfer_bytes 完成并收到完成确认

Phase 3 dst_local: dst_relay -> dst_rank
等待全部 transfer_bytes 完成并收到完成确认
```

这是真正的 message-level store-and-forward，不是 packet pipeline。Phase 2 不能与 Phase 1 重叠，Phase 3 不能与 Phase 2 重叠。

若 `src_relay == src_rank`，跳过 Phase 1；若 `dst_relay == dst_rank`，跳过 Phase 3。fabric phase 始终存在，并继承当前全局 `flow-ecmp`、`oblivious-spray` 或 `ecmp-rr` 策略。

### 4.3 Flow ID、完成与 DAG barrier

- fabric subflow 使用输入的 flow/task ID。
- 服务器内部 subflow 使用仿真器动态分配的内部 flow ID，避免与用户 ID 冲突。
- 逻辑 flow 只有在最后一个实际 subflow 完成后才完成。
- DAG network task 的完成回调只在整个三阶段状态机结束后触发。
- DAG 的后继 barrier 因此会等待目的服务器内部转发结束。
- CM 的 `send_done_trigger` 和 `recv_done_trigger` 绑定到最后一个实际 subflow。
- CM 的启动 `trigger` 只启动第一个实际 subflow。

首版不支持 `server_forward` 与 `-conn_reuse`/`msg` 组合，也不支持一侧多个 relay 或服务器内部多跳。

### 4.4 日志

状态机输出可机器检查的日志：

```text
SERVER_FORWARD_BEGIN flow=3 src=0 src_relay=3 dst_relay=11 dst=8 bytes=16384 phases=3
SERVER_FORWARD_PHASE_START flow=3 phase=src_local src=0 dst=3 time_us=0
SERVER_FORWARD_PHASE_DONE flow=3 phase=src_local time_us=...
SERVER_FORWARD_PHASE_START flow=3 phase=fabric src=3 dst=11 time_us=...
SERVER_FORWARD_PHASE_DONE flow=3 phase=fabric time_us=...
SERVER_FORWARD_PHASE_START flow=3 phase=dst_local src=11 dst=8 time_us=...
SERVER_FORWARD_PHASE_DONE flow=3 phase=dst_local time_us=...
SERVER_FORWARD_DONE flow=3 time_us=...
```

### 4.5 与 DeepEP 分层 builder 的关系

`server_forward` 可以表达历史上的 destination-side coarse 模型：

```text
src_relay = logical src_rank
dst_relay = 目标服务器中与 src_rank 相同 local index 的 rank
```

此时 source-local phase 跳过。对于逻辑 task `0 -> 9`、每服务器 8 张 GPU：

```text
route server_forward src_relay:0 dst_relay:8

实际 flow 1: fabric   0 -> 8
实际 flow 2: dst_local 8 -> 9
```

两个 flow 严格串行，但仍属于一个 DAG network task。task 只有在 `dst_local`
完成后才通知 barrier；若逻辑目标就是 relay，例如 `0 -> 8`，则只执行 fabric
flow。

当前 DeepEP/EPLB/MoonEP builder 不使用这个 coarse 封装。它们显式生成
`dispatch_fabric -> dispatch_local` 和
`combine_local_reduce -> combine_fabric`，以支持一份 RDMA payload 被多个目标
rank 共享。`server_forward` 继续作为 CM/DAG 的通用源路由功能独立测试。

## 5. 错误处理

route 语法错误或拓扑坐标不合法时，程序必须在事件循环开始前返回非零状态，并给出包含 flow/task 上下文的错误信息。不能静默退回普通 ECMP，也不能只执行路径的一部分。

必须拒绝的典型输入包括：

- 未知 route 模式或未知 token；
- `explicit` 路径首尾 rank 不匹配；
- 跨 rail 路径缺少 L1 或跨 plane；
- bundle、spine、rail、plane 越界；
- relay 不在对应端点服务器内；
- compute task 携带 route；
- `server_forward` 与 connection reuse 混用。

## 6. 验收标准

- 不带 route 的既有 CM/DAG 结果保持兼容。
- `explicit` 的实际链路严格等于输入坐标，且不受三种全局路由策略影响。
- 显式正向与自动反向路径使用成对的 plane、spine 和 bundle。
- `server_forward` 三阶段严格串行，跳过规则正确。
- destination-side coarse 用例固定跳过 source-local，并正确执行一或两个实际 flow。
- CM 与 DAG 都支持两种 route 模式。
- DAG task 只在最终本地 phase 完成后通知 barrier。
- 非法输入在仿真开始前确定性失败。
