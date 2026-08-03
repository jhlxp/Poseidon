# DAG 任务与执行模型

## 1. 输入格式

每行只表示一个 network task 或一个 compute task：

```text
task_id stage_id src_rank dst_rank compute_rank bytes compute_us [predecessor_stage ... | -]
```

合法 network task：

```text
10 1 0 64 - 1048576 0 0
```

合法 compute task：

```text
20 1 - - 64 0 25 0
```

不允许一行同时包含网络和计算，也不允许两者同时为空。

## 2. Stage 语义

- 同一 stage 的全部 task 在同一模拟时刻启动。
- network task 的时间由 UEC 在 MpRail 上产生的真实 FCT 决定。
- compute task 使用固定 `compute_us` 完成事件。
- stage 在最后一个 task 完成时结束。
- 后继 stage 只在全部前驱 stage 完成后启动。

因此：

```text
T_stage = max(task_0_done, task_1_done, ..., task_n_done) - T_stage_start
```

## 3. 多通信与多计算

一个 source 向多个 destination 发送时，为每个 destination 写独立 network task：

```text
1 0 0 64  - 1048576 0 -
2 0 0 128 - 1048576 0 -
3 0 0 192 - 1048576 0 -
```

同一 stage 可以并行启动多个 kernel：

```text
4 0 - - 0 0 20 -
5 0 - - 1 0 30 -
```

MpRail 的网络竞争由 Queue/UEC 决定；计算 task 首版仍是固定时间事件，不模拟 SM occupancy 和同 GPU kernel 排队。

## 4. 与 MpRail 的关系

- DAG 不决定 plane 或 spine。
- network task 只提供 `src_rank`、`dst_rank`、`bytes` 和 flow/task ID。
- MpRail 根据 flow ID 选择 source plane，并在该 plane 中选择 L1-EPS。
- 同 stage 的多个 network task 可以散列到不同 plane。
- compute task 不进入网络拓扑。

## 5. 失败条件

以下输入必须在仿真开始前失败：

- task ID 为 0 或重复；
- network 和 compute 同时存在；
- network 和 compute 同时为空；
- rank 超出节点范围；
- 同一 stage 的 task 声明不同前驱；
- 前驱 stage 不存在；
- stage 图存在环。

事件循环结束时 DAG 尚未完成也必须返回失败。
