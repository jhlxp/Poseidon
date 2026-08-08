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

DeepEP/EPLB/MoonEP 的 fabric/local hierarchy leg 已经是独立 task，按各自真实端点
进入 TX/RX lane。通用 `server_forward` 的 `src_relay/dst_relay` 也会计入相应
TX/RX lane；由于它仍是一个包含多个串行 subflow 的逻辑 task，relay lane 首版画
该 task 的完整 FCT，而不是每个 subflow 的精确子区间。

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
等全部串行 subflow，bar 不把内部 phase 再拆开。HTML hover 显示 task
摘要，点击 bar 后的 Task Inspector 会显示完整 `route_spec`。

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

ProbeEP planner 是离线决策，不生成 task，因此 ProbeEP timeline 只有每张 GPU 的
`Compute / Network TX / Network RX`，不显示 CPU planner lane，也不增加全局
`Expert Transfer` 或 `Expert weights` lane。`expert_weight_scatter/rdma/gather/prefetch`
和 token 通信一样，按真实 source/destination/relay 直接画入对应 GPU 的 TX/RX 行。
权重使用独立的青色，hover/Inspector 继续显示具体 payload、endpoint、bytes 和 FCT；
Python route lowering 耗时不计入 makespan。

每 rank 的 communication stream 顺序固定为：

```text
expert weight prefetch/scatter/RDMA/gather -> token dispatch -> token combine
```

因此同一 rank 先完成本轮需要的权重传输，再启动 token dispatch。不同 rank 的通信仍
允许并行，不增加全局 weight barrier。这样 timeline 直接呈现每张 NIC 的实际顺序，
也不会把同一 weight task 在全局 lane 和 TX/RX lane 重复画两次。

主类别固定使用七种不同色相，不使用同色系深浅表达不同语义：

| 类别 | 颜色 |
|---|---|
| Attention | 蓝 `#2563EB` |
| Router/placement proxy | 金 `#C89400` |
| Expert FFN | 绿 `#16A34A` |
| Reduce | 紫 `#7C3AED` |
| Token Dispatch | 橙 `#EA580C` |
| Token Combine | 红 `#DC2626` |
| Expert Weight | 青 `#0891B2` |

`scatter/rdma/gather/prefetch` 四种 expert-weight leg 使用同一青色，因为它们都属于
权重通信；具体 leg 仍由 hover/Inspector 中的 `payload_kind` 区分。

ProbeEP 单算法 Dashboard 另有 `Cross-server expert migration` 表，逐 layer/microbatch
展示 planned、admitted、deferred remote experts、moved routes 和 RDMA weight bytes；
同时报告每个 sample 的 peak-used/total expert slots。总 Dashboard 的算法摘要显示
admitted expert 总数和权重总量。

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
| `dag_gpu_timeline.html` | 自包含交互时间线；Fit/缩放、hover 摘要、点击 Inspector 和按需展开依赖 |
| `dag_task_timeline.csv` | 每个 task 的 start/end/FCT/bytes/logical throughput |
| `dag_rank_overlap_summary.csv` | 每 GPU compute/network active、overlap、TX/RX 字节 |
| `dag_timeline_summary.json` | makespan、task 数、逐 payload 字节/FCT、expert-weight 汇总和统计语义 |

HTML 下方的 `Task details` 默认收起，只在首次展开时生成逐 task
明细表。大量 task 在同一 network lane 并发时，bar 可能视觉重合；hover、
Inspector 和明细表仍能逐项恢复。独立调试时可使用 `--ranks` 只观察部分 GPU；
统一 EP32 smoke/full 禁止筛选，固定展示 rank 0-31。

### 4.1 HTML 数据组织

timeline 不在每个 SVG bar 中写入一份完整 tooltip。当前数据流为：

```text
task data:          每个 task 只保存一次
SVG bar:            data-task-id + lane
hover:              通过 task_id 动态读取摘要
click:              固定 Task Inspector
predecessor expand: 解压 gzip 数据，按 task_id 恢复完整名称
Task details:       首次展开时从同一份 task data 生成表格
```

`server_forward` task 即使同时画在 TX/RX/relay lane，也只复用一份 task
元数据。predecessor 列表使用 task ID 引用并 gzip 内联，不在 HTML 中重复
数千次长 task key。浏览器需支持标准 `DecompressionStream('gzip')`；当前
Chrome 和 Firefox 可直接离线打开。

这种组织避免 task metadata 随 lane 数重复膨胀。完整 predecessor 信息仍可在
Inspector 中按需恢复；文件大小以当次 ZIP 和 task 数为准，不在文档中固化旧的
局部-rank 样本值。

### 4.2 Gate / expert load before-after 模块

`visualization/gate_load_profile.py` 是独立于时间线的第二个 HTML 模块。它参考
[UltraEP profiler/viewer](https://dots-infra.github.io/UltraEP/zh/#4-效果可视化追踪每个-microbatch-的均衡收益)
的层次化观察方法：先看所有 rank 的总体不均衡，再下钻到
指定 layer、microbatch 和 expert instance。UltraEP 采集运行时 reroute 前后的
真实负载；本项目直接读取 workload `manifest.json` 中生成器已经确定的：

```text
micro_batch_algorithms[*].gate
micro_batch_algorithms[*].expert_load_profile.before
micro_batch_algorithms[*].expert_load_profile.after
```

两种状态的准确含义是：

| 状态 | 含义 |
|---|---|
| `before` | Gate logical assignments 按原始 expert placement 聚合 |
| `after` | 算法把相同 logical assignments 分给实际 physical instance/execution rank 后聚合 |

页面的 before/after rank 图强制使用相同 load 横轴，不能分别 autoscale 后制造
“看起来一样均衡”的错觉。rank bar 内按 logical expert 分段，hover 可查看 instance、
expert、rank 和 load；下方另画不受副本放置影响的 Gate logical-expert 分布，并提供
可按 before/after 和 rank 筛选的实例明细。

输出为：

```text
<output-dir>/
├── gate_load_profile.html
├── gate_load_profile.csv
└── gate_load_profile_summary.json
```

它不是 HTSim 时间结果，也不表示真实 GPU kernel profiler。load 单位是 logical
token route 数；页面负责回答“Gate 有多偏、算法把执行负载搬到哪里”，DAG timeline
负责回答“任务实际何时开始/结束”，链路图负责回答“物理网络何时有多少流量”。

## 5. 使用方法

```bash
python3 visualization/dag_timeline.py \
  --workload-dir <generator-output-dir> \
  --htsim-log <case-dir>/htsim.log \
  --output-dir <case-dir>/dag_timeline \
  --gpus-per-server 8 \
  --ranks 0-31 \
  --title "DeepEP EP32 timeline"
```

以下局部筛选只用于独立调试，不得用于统一 EP32 smoke/full 产物。只观察 server 0
和 server 2：

```bash
python3 visualization/dag_timeline.py \
  --workload-dir <generator-output-dir> \
  --htsim-log <case-dir>/htsim.log \
  --ranks 0-7,16-23
```

### 5.1 横向缩放与时间尺度

时间线打开时默认 `Fit`，完整 makespan 刚好适配当前可视宽度，不需要
先横向拖动。工具栏支持：

- `-` / `+` 按钮和 slider 连续缩放；
- `Ctrl + wheel` 在时间线内缩放；
- `Fit` 恢复全局；
- 动态刻度和 `100 px = X us` 读数，同时显示当前可见时间及总时间。

缩放时 bar、背景、分割线和刻度使用同一 `px/us` 比例重算。因此同一
水平距离对应的实际时间是明确的，不会只是把 SVG 图片放大。

### 5.2 五算法合并页

```bash
python3 tests/run_dsv3_2layer_algorithms.py
python3 tests/run_dsv3_2layer_algorithms.py --full --workers 4
```

标准入口先为每个算法临时生成独立的 `algorithm_dashboard.html`。单算法页面完整
包含可折叠 Gate/expert before-after、独立可缩放 timeline、MpRail 链路负载图、
makespan/task/bytes/overlap 摘要和相关 CSV 下载。直接打开任一算法页面，不需要
再回到五算法总览才能查看该算法的完整信息。

根目录再生成 `dsv3_algorithm_comparison.html`：NCCL、DeepEP、EPLB、MoonEP 和
ProbeEP 各有一个可折叠区域，页头支持全部展开/收起；每个区域嵌入对应的完整算法
dashboard，并提供单独打开链接。五个 timeline 都固定包含 GPU 0-31；
`selected_ranks` 和 overlap 汇总也必须覆盖全部 32 rank。

合并页通过相对路径引用各算法的 HTML、PNG 和 CSV，不把大文件重复内联到
一个文件。同时生成 `dsv3_visualization_bundle.zip`，其内只包含：

```text
dsv3_algorithm_comparison.html
algorithms/<algorithm>/algorithm_dashboard.html
algorithms/<algorithm>/gate_load/*
algorithms/<algorithm>/timeline/*
algorithms/<algorithm>/link_load/*
```

ZIP 不包含 workload、HTSim log、`htsim.dat` 或原始 `output_metrics`。ZIP 写入并
校验成功后，runner 立即删除服务器 run 目录中的总览 HTML、五个算法 dashboard、
五个 Gate HTML 和五个 timeline HTML；CSV、JSON、PNG、workload 和仿真日志仍
按原样保留。因此标准 run 成功后
服务器目录中应有 `0` 个散装 HTML。

下载 ZIP 后解压：打开根目录的 `dsv3_algorithm_comparison.html` 查看五算法总览；
打开 `algorithms/<algorithm>/algorithm_dashboard.html` 只查看一个算法的完整信息。

## 6. 应当观察什么

首轮 workload 分析建议按以下顺序：

1. `makespan_us` 是否由 compute 还是 network 尾部决定；
2. Attention、Dispatch、Expert、Combine 的真实先后与 overlap 是否符合 DAG；
3. 哪些 GPU 的 TX/RX 长时间活跃，哪些 GPU 几乎没有通信；
4. 每 GPU `compute_network_overlap_us` 和 overlap fraction；
5. Dispatch/Combine 的字节是否符合算法去冗余规则；
6. network task 的 FCT 是否存在明显长尾；
7. 再结合链路负载图定位 endpoint、rail 或 spine 瓶颈。
8. 对 EPLB/MoonEP/ProbeEP 比较 Gate before 与 execution after 的 rank `max/mean`，并确认
   total routes 和 logical-expert histogram 没有改变。

只有 DAG timeline 和物理链路 timeline 使用相同 HTSim run、相同时间轴时，才能把
模型 phase 与网络拥塞对应起来。

## 7. 当前边界

- 不解析单条 flow 内 packet 进度；一个 DAG network task 是一根完整 FCT bar。
- `server_forward` relay 使用现有 TX/RX lane，但首版不拆出 subflow 子区间。
- 不根据重叠 bar 推导真实 SM occupancy；stream 顺序来自生成器的静态 DAG edges。
- 不用逻辑 throughput 替代物理链路 sampler。
- 当前不画 predecessor 箭头；完整依赖保留在点击后的 Inspector 和 task
  CSV 中，避免 EP32 图被大量连线覆盖。

这些边界不妨碍当前目标：观察每个 GPU 的计算、通信、实际持续时间、传输字节和
二者的 DAG overlap。

## 8. 测试

独立测试入口：

```bash
python3 tests/run_dag_timeline_visualization.py
python3 tests/run_gate_workload_visualization.py
```

固定合成 case 包含两个 GPU、三个 compute task 和一个 2-8us network task，验证：

- task map 与 HTSim log 完整 join；
- network FCT 和 logical throughput；
- GPU0 计算通信 overlap 为 6us；
- GPU1 计算通信 overlap 为 2us；
- 不完整日志被拒绝；
- task data 只内联一份，SVG 不含重复 `<title>`，完整 predecessor gzip 可恢复；
- HTML hover、点击 Inspector、Fit/缩放控件、动态尺度、可折叠 Task details、逐 task CSV、
  逐 rank CSV 和总览 JSON 完整生成；
- 两个伪算法 case 能合成可折叠总览页，并生成仅含 timeline 和链路图的
  可解压 ZIP。
- EP32 smoke/full 的五个 timeline summary 均报告 `selected_ranks=0..31`，ZIP 内
  页面可恢复 GPU 00 到 GPU 31，服务器 run 目录不保留散装 HTML。
- Gate 测试验证 raw `32 x 58 x 9` physical receive 数据向 256 logical experts
  折叠守恒、五种 provider 可复现、before/after 逻辑路由守恒，以及 HTML/CSV/JSON
  三种产物完整生成。
- 五算法 runner 验证五份 manifest 的 Gate assignment digest 序列完全相同；ZIP
  同时包含五个完整算法 dashboard、各算法 Gate/timeline HTML 和一个总 dashboard。
