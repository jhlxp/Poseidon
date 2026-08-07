# DAG 执行时间线可视化

## 1. 目标

`visualization/dag_timeline.py` 把 workload 生成器的 task 语义与 HTSim 实际执行
时间合并，输出类似 Nsight Systems 的 GPU swimlane：

```text
GPU 0  Compute     [Attention----------][Expert------]
       Network TX        [Dispatch-------------]
       Network RX                         [Combine--------]

GPU 1  Compute       [Router---]          [Expert------]
       Network TX                         [Combine--------]
       Network RX         [Dispatch-------------]
```

每张 GPU 固定三条 lane：

- `Compute`：绑定该 rank 的 compute task；
- `Network TX`：该 rank 是逻辑 network task source；
- `Network RX`：该 rank 是逻辑 network task destination。

`server_forward` 的 `src_relay/dst_relay` 也计入相应 TX/RX lane。由于一个逻辑 task
可能包含多个串行 subflow，relay lane 首版仍画该 task 的完整 FCT，而不是每个
subflow 的精确子区间。

同一时间横坐标下 Compute 与 TX/RX 同时出现，就表示当前 DAG 和 HTSim 执行结果中
存在计算通信 overlap。

## 2. 输入如何关联

工具读取三个文件：

| 文件 | 使用内容 |
|---|---|
| `manifest.json` | `num_ranks` 和 workload 总体信息 |
| `task_map.json` | task ID、逻辑名称、rank、字节、payload、依赖和理论 `compute_us` |
| `htsim.log` | `DAG_TASK_START/DAG_TASK_DONE` 的实际仿真时间戳 |

关联键是唯一 `task_id`：

```text
task_map.json task_id
  + htsim.log DAG_TASK_START task=<id>
  + htsim.log DAG_TASK_DONE  task=<id>
  -> 一个完整 TaskEvent
```

任何 task 缺少 start/done、出现重复事件或日志含未知 task，工具都会报错，不生成
部分时间线。

## 3. 时间和字节口径

### 3.1 Compute

compute task 同时记录：

```text
declared_compute_us = task_map.json 中生成器写入的固定理论时长
actual_duration_us  = DAG_TASK_DONE - DAG_TASK_START
```

正常情况下两者只存在日志打印精度导致的微小差异。横向 bar 使用 HTSim 的实际
start/done 时间，因此依赖等待和并发位置均来自本次仿真结果。

### 3.2 Network

network task 的：

```text
logical bytes = task_map.json.transfer_bytes
logical FCT   = DAG_TASK_DONE - DAG_TASK_START
```

若 task 使用 `server_forward`，FCT 包含该逻辑 task 的 fabric 和 destination-local
等全部串行 subflow，bar 不把内部 phase 再拆开。HTML hover 会显示 `route_spec`。

逐 task CSV 额外给出：

```text
logical_throughput_gbps = transfer_bytes * 8 / FCT_us / 1000
```

它是逻辑 payload 的端到端有效吞吐，不是任意一条物理链路的吞吐。物理
Host/L0/L1/NVLink 链路利用率仍应使用 `mprail_link_load.py` 的 sampler 结果。

### 3.3 每 GPU overlap

对每张 GPU 分别计算：

```text
compute_active = union(all compute task intervals on rank)
network_active = union(all TX/RX task intervals touching rank)
overlap        = intersection(compute_active, network_active)
```

因此多条并发 flow 不会把 network active time 重复相加。`TX bytes` 和 `RX bytes`
仍按逻辑 task 字节分别累计。

这个 overlap 表示 DAG 时间并发。workload 生成器已把每 GPU 的 compute/comm
逻辑 stream 顺序降低为 predecessor edges，但 HTSim 没有实现运行时 CUDA stream、
动态 SM、HBM 或通信 kernel 资源调度。当前 compute 仍是固定时长事件，通信重叠时
可使用生成器约定的 20 SM 静态预留。

## 4. 输出

每次运行产生：

```text
<output-dir>/
├── dag_gpu_timeline.html
├── dag_task_timeline.csv
├── dag_rank_overlap_summary.csv
└── dag_timeline_summary.json
```

| 输出 | 用途 |
|---|---|
| `dag_gpu_timeline.html` | 自包含交互时间线；横向滚动，hover 查看 task、字节、FCT、route 和依赖 |
| `dag_task_timeline.csv` | 每个 task 的 start/end/FCT/bytes/logical throughput |
| `dag_rank_overlap_summary.csv` | 每 GPU compute/network active、overlap、TX/RX 字节 |
| `dag_timeline_summary.json` | makespan、task 数、payload 字节和统计语义 |

HTML 下方还包含逐 task 明细表。大量 task 在同一 network lane 并发时，bar 可能视觉
重合；hover 和明细表仍能逐项恢复。可使用 `--ranks` 只观察部分 GPU。

## 5. 使用方法

```bash
python3 visualization/dag_timeline.py \
  --workload-dir <generator-output-dir> \
  --htsim-log <case-dir>/htsim.log \
  --output-dir <case-dir>/dag_timeline \
  --gpus-per-server 8 \
  --title "DeepEP EP32 timeline"
```

只观察 server 0 和 server 2：

```bash
python3 visualization/dag_timeline.py \
  --workload-dir <generator-output-dir> \
  --htsim-log <case-dir>/htsim.log \
  --ranks 0-7,16-23
```

HTML 横向尺寸由 `--pixels-per-us` 控制，最小为 1400 px。

## 6. 应当观察什么

首轮 workload 分析建议按以下顺序：

1. `makespan_us` 是否由 compute 还是 network 尾部决定；
2. Attention、Dispatch、Expert、Combine 的真实先后与 overlap 是否符合 DAG；
3. 哪些 GPU 的 TX/RX 长时间活跃，哪些 GPU 几乎没有通信；
4. 每 GPU `compute_network_overlap_us` 和 overlap fraction；
5. Dispatch/Combine 的字节是否符合算法去冗余规则；
6. network task 的 FCT 是否存在明显长尾；
7. 再结合链路负载图定位 endpoint、rail 或 spine 瓶颈。

只有 DAG timeline 和物理链路 timeline 使用相同 HTSim run、相同时间轴时，才能把
模型 phase 与网络拥塞对应起来。

## 7. 当前边界

- 不解析单条 flow 内 packet 进度；一个 DAG network task 是一根完整 FCT bar。
- `server_forward` relay 使用现有 TX/RX lane，但首版不拆出 subflow 子区间。
- 不根据重叠 bar 推导真实 SM occupancy；stream 顺序来自生成器的静态 DAG edges。
- 不用逻辑 throughput 替代物理链路 sampler。
- 当前不画 predecessor 箭头；依赖保留在 HTML tooltip 和 task CSV 中，避免 EP32
  图被大量连线覆盖。

这些边界不妨碍当前目标：观察每个 GPU 的计算、通信、实际持续时间、传输字节和
二者的 DAG overlap。

## 8. 测试

独立测试入口：

```bash
python3 tests/run_dag_timeline_visualization.py
```

固定合成 case 包含两个 GPU、三个 compute task 和一个 2-8us network task，验证：

- task map 与 HTSim log 完整 join；
- network FCT 和 logical throughput；
- GPU0 计算通信 overlap 为 6us；
- GPU1 计算通信 overlap 为 2us；
- 不完整日志被拒绝；
- HTML hover、逐 task CSV、逐 rank CSV 和总览 JSON 完整生成，且不产生 PNG。
