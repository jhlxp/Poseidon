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

每个 ProbeEP invocation 的 Weight 与 Dispatch 属于同一 communication stage。对每条
source rail，只保证：

```text
same-source remote Weight RDMA TX -> Dispatch fabric TX
```

它不等待 local prefetch、目的 RX/gather、其他 rail 或全局 Weight 完成。这样 timeline
直接呈现每张 NIC 的实际顺序，
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
不展示没有进入算法决策的虚假 slot/capacity 指标。总 Dashboard 的算法摘要显示
admitted expert 总数和权重总量。

`gate_load_profile.html` 的跨服务器区域按所选 microbatch 的最后一层展示 remote
copy 数与 expert IDs，再将最后一层的 Dispatch/Combine token 和 Expert Weight 按有向
`src_server -> dst_server -> rail/NIC` 展开。这个区域用于观察空间负载，不代替
timeline 的 FCT 和时间序列。

ProbeEP 2-layer ratio-control 实验生成一张总 Dashboard，展示同一次 HTSim 中 4 个
`Weight+Dispatch` control observations 的 `Cref`、`Nmax`、planned/admitted replica
及 Expert Weight RDMA 字节，并把 local NVLink Weight 与 remote RDMA Weight 分列；
同时把 4 个 Combine phases 作为 telemetry 单独展示。
Combine 不进入迁移 controller。Dashboard 链接同一 workload 的 GPU
0-31 完整 timeline、Gate before-after 和链路负载。32 张 GPU 的 lane 必须全部保留，
不得聚合或只画 rank 0-7。Dashboard 同时显示瓶颈 NIC 的实际 Token+Weight 总字节、
按 `0.9*Cmax/Nmax` 得到的 probe total、`Cmax*400Gbps` 理论 total，以及更新前后的
Expert Weight migration budget。这些列必须区分“实际字节”“总字节上限”和“扣除
token 后的专家预算”，不能把链路图中的实际 MB 解释成 budget；页面不展示 replay
字段。`probeep_dispatch_observations.json` 在每个
observation 中保留全部 rank 的 compute/communication 时间；其中
`weight_dispatch_us_by_rank` 是从配对 compute start 到该 rank 最后一个
Weight/Dispatch TX/RX task 完成的 stage elapsed time，不是 NIC active time。
`migration_tx_bytes_by_rank` 和 `migration_rx_bytes_by_rank` 分别记录方向字节；
`migration_endpoint_bytes_by_rank=max(TX,RX)` 是 controller 的 full-duplex endpoint
footprint。只能使用 `remote_weight_rdma_bytes=sum(TX)=sum(RX)` 表示实际跨机权重字节，
不能把 endpoint footprint 跨 rank 求和后当作 RDMA 总量。

Gate 页的 `Per-NIC directed load` 是整次 invocation 的空间流量汇总，柱中包含
`Dispatch + Combine + Expert Weight`；它不是 controller 的 migration budget，也不是
`Weight+Dispatch` observation 的 `Bsample`。Dashboard 必须分别标出：

```text
directed total       = Dispatch + Combine + Expert Weight 空间汇总
Bsample and Nmax     = 实测 Weight+Dispatch 窗口的瓶颈 endpoint 字节和时间
probe total          = alpha * (Cmax / Nmax) * Bsample
theoretical total    = Cmax * 400 Gbps
migration budget     = min(probe total, theoretical total) - token baseline
```

这些值回答不同问题，不能直接互相比较大小来判断功能是否正确，也不应
在文档中固化已删除实验的数字。
`probeep_combine_telemetry.json` 则保存 Combine active/compute/overlap 的 per-rank 数组；
Dashboard 的 max 值只是摘要。Expert Weight 使用同一主色仅表示数据
类别一致，判断 controller 是否减少跨机迁移必须查看 `remote_weight_rdma_bytes`，
不能把 `expert_weight_prefetch/scatter/gather` 的 local NVLink 字节计作 remote migration。

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
| `dag_timeline_summary.json` | makespan、task 数、逐 payload 字节/FCT、expert-weight RDMA/local/logical-leg 分项和统计语义 |

`expert_weight_logical_leg_bytes` 会把 scatter、RDMA、gather 和 prefetch 等逻辑 task
leg 全部相加，只用于解释 timeline 工作量，不能代表跨机迁移量。跨机口径使用
`expert_weight_rdma_bytes`；服务器内部权重 leg 汇总使用
`expert_weight_local_bytes`。兼容字段 `expert_weight_bytes` 与 logical-leg 总量相同，
新分析不得再把它解释为 RDMA bytes。

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
指定 microbatch 和 expert instance。UltraEP 采集运行时 reroute 前后的
真实负载；本项目直接读取 workload `manifest.json` 中生成器已经确定的：

```text
micro_batch_algorithms[*].gate
micro_batch_algorithms[*].expert_load_profile.before
micro_batch_algorithms[*].expert_load_profile.after
```

页面以 microbatch 为唯一选择维度。底层 summary 统一保存两端快照：

| 状态 | 含义 |
|---|---|
| `before` | 第一层 raw Gate logical assignments 按 baseline expert placement 聚合 |
| `after` | 最后一层的实际 physical instance/execution rank 负载 |

页面展示必须按算法解释这两端：

| 算法 | HTML 负载语义 | 专属内容 |
|---|---|---|
| NCCL | first/final layer 原始 expert execution load | 无去重直达的 Dispatch/Combine |
| DeepEP | first/final layer 原始 expert execution load | 去重后的分层 token transport |
| EPLB | first-layer baseline / final-layer physical placement | primary/replica instances |
| MoonEP | first-layer baseline / final-layer server-local replica placement | master/local replica instances |
| ProbeEP | first-layer baseline / final-layer admitted placement | P/A/D、remote copies、Expert Weight |

因此 NCCL/DeepEP 页不显示“migration 0/0/0”或重复的算法 before/after；
Cross-server expert copies 和 Expert Weight 列只存在于 ProbeEP。标题必须同时写出
`Layer N (id n)` 和 MB；页面不提供独立 Layer 选择器。算法仍在 manifest 中为每个
`(layer,microbatch)` 保留完整 before/after profile，CSV/JSON 不丢失逐层原始数据。

页面的 before/after rank 图强制使用相同 load 横轴，不能分别 autoscale 后制造
“看起来一样均衡”的错觉。rank bar 内按 logical expert 分段，hover 可查看 instance、
expert、rank 和 load；server 聚合图必须先于 rank 图并标出均值。Gate 输入分布分别
展示第一层和最后一层 raw logical-expert histogram；它不表示算法 after placement。
实例明细中的 Before/After 也分别绑定首层 raw 和末层 admitted 快照。

网络模块统一绑定最后一层：

- `Cross-server expert copies` 只统计最后一层真正 admitted 的完整 remote replicas；
- planned/admitted/deferred 和 server padded routes 只报告最后一层；
- directed server-pair 与 per-NIC 图对所有算法画最后一层的
  Dispatch/Combine；仅 ProbeEP 再加 Expert Weight；
- 不跨层累计 bytes 或 replica 数，因为这些模块用于观察当前实验最接近收敛的状态。

输出为：

```text
<output-dir>/
├── gate_load_profile.html
├── gate_load_profile.csv
└── gate_load_profile_summary.json
```

它不是 HTSim 时间结果，也不表示真实 GPU kernel profiler。load 单位是 TopK 展开后的
logical expert-token route 数，不是 unique input tokens，也不是执行时间。页面负责回答
“首层 raw Gate 有多偏、末层算法把执行负载搬到哪里”，DAG timeline
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

### 5.2 有效五算法聚合页

```bash
python3 tests/run_dsv3_2layer_algorithms.py \
  --full --workers 4 --algorithms nccl,deepep,eplb,moonep
python3 tests/run_probeep_2layer_ratio_full.py --full
```

第一个入口为四个静态算法生成独立的 `algorithm_dashboard.html`；第二个入口
在单个持续 HTSim PID 中生成动态 ProbeEP dashboard。单算法页面保留
makespan/task/bytes/overlap 摘要和 Gate、timeline、link-load 三个独立入口。
这个导航页不嵌入任何重量 HTML 或 PNG。

聚合步骤再生成 `dsv3_algorithm_comparison.html`：NCCL、DeepEP、EPLB、MoonEP 和
动态 ProbeEP 各有一个摘要行和单一 dashboard 链接。总览不使用 iframe，不默认
打开第一个算法，也不同时请求任何 Gate/timeline/link-load 资源。
五个 timeline 都固定包含 GPU 0-31；
`selected_ranks` 和 overlap 汇总也必须覆盖全部 32 rank。
通用静态 runner 的默认列表中虽然有静态 ProbeEP，但该页面只能用于结构回归，
有效性能聚合必须替换为动态入口的页面。

三级页面通过相对路径逐级跳转，不把大文件重复内联或预加载到
导航页。同时生成 `dsv3_visualization_bundle.zip`，其内只包含：

```text
dsv3_algorithm_comparison.html
algorithms/<algorithm>/algorithm_dashboard.html
algorithms/<algorithm>/gate_load/*
algorithms/<algorithm>/timeline/*
algorithms/<algorithm>/link_load/*
```

ZIP 不包含 workload、HTSim log、`htsim.dat` 或原始 `output_metrics`。ZIP 写入并
校验成功后，打包流程删除服务器 run 目录中的总览 HTML，以及本次实际
包含的每个算法 dashboard、Gate HTML 和 timeline HTML；CSV、JSON、PNG、workload 和仿真日志仍
按原样保留。因此标准 run 成功后
服务器目录中应有 `0` 个散装 HTML。

下载 ZIP 后解压：打开根目录的 `dsv3_algorithm_comparison.html` 查看五算法总览；
点击某个算法后进入 `algorithms/<algorithm>/algorithm_dashboard.html`，再点击
Gate、timeline 或 link-load 才加载对应的唯一重量资源。

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
- 不根据重叠 bar 推导真实 SM occupancy；stream 顺序来自生成器提交的不可变 DAG edges，可以是静态文件或动态 append batch。
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
- 两个伪算法 case 能合成三级导航页，根页和单算法页不包含 iframe/预加载图片，
  并生成可解压 ZIP。
- 有效 EP32 smoke/full 聚合的五个 timeline summary 均报告 `selected_ranks=0..31`，ZIP 内
  页面可恢复 GPU 00 到 GPU 31，服务器 run 目录不保留散装 HTML。
- Gate 测试验证 raw `32 x 58 x 9` physical receive 数据向 256 logical experts
  折叠守恒、五种 provider 可复现、before/after 逻辑路由守恒，以及 HTML/CSV/JSON
  三种产物完整生成；五个算法必须生成各自的 execution/placement 标题和列，
  ProbeEP-only migration/Expert Weight 不得出现在其他算法的可见模块中。
- 静态 runner 先验证四份 manifest 的 Gate assignment digest，聚合时再验证动态
  ProbeEP 的 digest 序列完全相同；ZIP
  同时包含五个完整算法 dashboard、各算法 Gate/timeline HTML 和一个总 dashboard。
