# MpRail 链路负载可视化

## 1. 目标与边界

`visualization/mprail_link_load.py` 从 HTSim 的链路负载采样 CSV 生成 MpRail
专用吞吐时间序列。它只处理当前 MpRail 的 EPS/rail/plane 和服务器内部 FullMesh
语义，不复用旧 OXC 图中的 `L1-OCS`、`Group` 或 `Tray` 分类。

主图使用四行布局：

```text
             L0 -> L1        L1 -> L0

             Host -> L0      L0 -> Host

       Server endpoint output    Server endpoint input
             (sum planes)             (sum planes)

                 Server-local FullMesh
```

这里按拓扑层次从上到下排列。L1 是 fabric 顶层，没有 OCS/Core 面板；服务器内部
FullMesh/NVLink 位于 GPU/NIC 一侧，因此放在最下面，不能放在 L1 上方。

第三行的 server endpoint 按一个 HTSim endpoint/rank 聚合，不会再把同一物理服务器
内的多个 rank 相加。红色容量线按 plane 数聚合：

```text
aggregate_rank_bandwidth = plane_count * 400 Gbps
```

专用测试为 `1 * 400 = 400 Gbps`；8-plane 实验为
`8 * 400 = 3200 Gbps = 3.2 Tbps`。

每条蓝色、橙色或绿色曲线表示一条有向物理链路的 bucket throughput。红色虚线
来自 `link_info.csv` 中该面板的真实 `rate_gbps`，不是脚本写死的 400 Gbps。
因此 fabric 可以显示 100/200/400/800 Gbps，服务器内部 FullMesh 也可以显示独立的
高速率。

## 2. 生成采样输入

运行 HTSim 时必须启用 link-load sampler：

```bash
HTSIM_LINK_LOAD_SAMPLE=1 \
HTSIM_LINK_LOAD_SAMPLE_US=1 \
./htsim/sim/build-mprail/datacenter/htsim_uec \
  -topology mprail \
  ...
```

`HTSIM_LINK_LOAD_SAMPLE_US` 是采样窗口，单位微秒。建议根据仿真持续时间选择：

| workload 时间范围 | 建议窗口 |
|---|---:|
| 数十到数百 us | 1 us |
| 数百 us 到数 ms | 5-10 us |
| 更长实验 | 50-1000 us |

HTSim 在当前工作目录写出：

```text
output_metrics/
├── link_info.csv
└── link_load_1ms.csv
```

`link_load_1ms.csv` 是历史文件名。实际窗口由环境变量决定，不能根据文件名假定为
1 ms。CSV 中 `time_ms` 使用毫秒，绘图时统一换算为微秒。

## 3. UEC CC 与 buffer 配置

### 3.1 默认拥塞控制

MpRail 当前由 `htsim_uec` 驱动，每条 CM/DAG network task 都创建 `UecSrc` 和
`UecSink`。默认配置是：

```text
transport = UEC
CC control point = sender-driven
sender CC algorithm = NSCC
receiver-driven CC = disabled
```

主程序默认令 `sender_driven=true`，随后调用 `UecSrc::initNsccParams()`；该函数启用
sender-based CC。`UecSrc::_sender_cc_algo` 的默认值是 `NSCC`。因此“默认是 UEC”
没有错，但更精确的说法是“UEC transport 上运行 sender-driven NSCC”。

可显式写成：

```text
-sender_cc_only -sender_cc_algo nscc
```

相关选项：

| 参数 | 含义 |
|---|---|
| `-sender_cc_only` | 只启用 sender-driven CC |
| `-sender_cc_algo nscc` | 使用默认 NSCC |
| `-sender_cc_algo dctcp` | 改为 sender-side DCTCP |
| `-sender_cc_algo constant` | 固定窗口，不根据 ACK/NACK 更新 cwnd |
| `-receiver_cc_only` | 关闭 sender CC，只启用 receiver-driven CC |

`-load_balancing_algo` 控制的是 UEC multipath/pathid，不是拥塞控制。当前默认值为
`mixed`，在 MpRail 上会启用 packet-spray 数据面；它与默认 NSCC 是两个独立维度。

### 3.2 Queue buffer

MpRail 每条有向链路都有独立 queue。当前外部链路和 server-local FullMesh 都使用
同一个 `mprail_cfg.queue_size`；区别只有 line rate 和 latency。

固定 buffer 使用：

```text
-q N
```

`N` 的单位是 packet，不是 byte 或 KB。queue byte 数为：

```text
queue_size_bytes = ceil(N * MTU_bytes)
```

当前默认 MTU 是 4150 bytes，因此测试中常用的：

```text
-mtu 4150 -q 32
```

对应：

```text
queue_size_bytes = 32 * 4150 = 132800 bytes
```

不写 `-q` 时启用自动 BDP buffer：

| 模式 | 默认 queue size |
|---|---:|
| 默认 trimming 开启 | `1 * BDP` |
| `-disable_trim` | `5 * BDP` |
| `-queue_size_bdp_factor F` | `F * BDP` |

MpRail 的 BDP 使用跨 rail 最长路径的 unloaded RTT、外部 `linkspeed` 和 MTU 计算，
最终向上取整为 packet 数，再换算成 byte。启动日志会打印：

```text
network_max_unloaded_rtt_us ...
bdp_pkt ...
queue_size_bytes ...
```

当前 MpRail 不使用通用 `-queue_type` 来选择 queue class。ECN 开启时构造
`ECNQueue`，关闭时构造普通 `Queue`。

### 3.3 ECN threshold

ECN 默认开启。不显式传 `-ecn` 时：

```text
ecn_low  = ceil(0.2 * bdp_pkt) * MTU
ecn_high = ceil(0.8 * bdp_pkt) * MTU
```

MpRail 的 `ECNQueue` 当前只使用 `ecn_low` 作为 marking threshold；`ecn_high` 仍会被
解析和校验，但不作为第二级 MpRail threshold。显式参数：

```text
-ecn LOW_PACKETS HIGH_PACKETS
```

两项输入的单位同样是 packet，进入拓扑前转换为 byte。`-disable_ecn` 会让 MpRail
使用普通 Queue。

采样 CSV 的 `max_queue_bytes` 是该时间桶观察到的最大 queue occupancy。判断 buffer
压力时应使用：

```text
occupancy_ratio = max_queue_bytes / queue_size_bytes
```

其中 `queue_size_bytes` 取 HTSim 启动日志，不能拿 400 Gbps line rate 代替。

### 3.4 推荐的显式实验参数

专用测试使用 400 Gbps/link、1 plane、400 Gbps/rank，建议把关键口径写全：

```bash
HTSIM_LINK_LOAD_SAMPLE=1 \
HTSIM_LINK_LOAD_SAMPLE_US=1 \
./htsim/sim/build-mprail/datacenter/htsim_uec \
  -topology mprail \
  -mprail_planes 1 \
  -linkspeed 400000 \
  -local_linkspeed 3200000 \
  -mtu 4150 \
  -q 32 \
  -sender_cc_only \
  -sender_cc_algo nscc \
  ...
```

这里没有显式写 `-ecn`，所以使用自动的 20%/80% BDP threshold，其中 MpRail queue
实际采用 low threshold。若实验需要固定 ECN threshold，应同时记录
`-ecn LOW_PACKETS HIGH_PACKETS`。

## 4. 使用

脚本使用 Python 标准库和 Matplotlib；运行环境需要能够 `import matplotlib`，不要求
Pandas。

```bash
python3 visualization/mprail_link_load.py \
  --metrics-dir <run-dir>/output_metrics \
  --output-dir <run-dir>/visualization \
  --planes 1 \
  --title "DeepEP EP32 MpRail link throughput"
```

可选时间窗口：

```bash
python3 visualization/mprail_link_load.py \
  --metrics-dir <run-dir>/output_metrics \
  --x-min-us 0 \
  --x-max-us 200
```

不指定 `--output-dir` 时，结果直接写回 `--metrics-dir`。

## 5. 输出

```text
<output-dir>/
├── mprail_link_load_by_layer.png
├── mprail_link_load_summary.csv
├── mprail_endpoint_load_summary.csv
└── mprail_link_inventory.csv
```

| 文件 | 内容 |
|---|---|
| `mprail_link_load_by_layer.png` | 四行七面板的物理链路与 endpoint 聚合 throughput |
| `mprail_link_load_summary.csv` | 每面板链路数、总字节、负载 CV、p50/p99/max throughput 和最大队列 |
| `mprail_endpoint_load_summary.csv` | endpoint 输入/输出的聚合容量、p50/p99/peak、utilization 和 headroom |
| `mprail_link_inventory.csv` | 从链路名恢复的 rank、rail、plane、spine 和 bundle 坐标 |

summary 的 `active_samples` 统计 CSV 中实际存在的采样行。`total_bytes` 和 percentile
只覆盖命令选择的时间窗口。

## 6. MpRail 链路分类

脚本不信任旧 sampler 的 `layer/direction` 字段，因为旧版本对 MpRail 链路可能写成
`unknown`。它严格解析 `link_name`：

| 名称形状 | 面板 | 结构化坐标 |
|---|---|---|
| `MPRAIL_LOCAL_MPRAIL_HOST_SRC_a->MPRAIL_HOST_DST_b(bN)` | server-local | src/dst rank、bundle |
| `MPRAIL_HOST_SRC_a->MPRAIL_L0_rR_pP(bN)` | Host -> L0 | rank、rail、plane、bundle |
| `MPRAIL_L0_rR_pP->MPRAIL_HOST_DST_b(bN)` | L0 -> Host | rank、rail、plane、bundle |
| `MPRAIL_L0_rR_pP->MPRAIL_L1_pP_sS(bN)` | L0 -> L1 | rail、plane、spine、bundle |
| `MPRAIL_L1_pP_sS->MPRAIL_L0_rR_pP(bN)` | L1 -> L0 | rail、plane、spine、bundle |

L0/L1 两端 plane 不一致时直接失败。以 `MPRAIL_` 开头但不匹配上述格式的链路也
直接失败，避免新链路被静默归入错误面板。

## 7. 统计语义

### 7.1 一条曲线是什么

第一、二和第四行的一条曲线对应 `link_id` 标识的一条有向 queue/link。它不是：

- 一个 flow 的发送速率；
- 一个 plane 的所有链路之和；
- 双向链路的合计；
- 所有 rail 的平均值。

同一个时间桶没有该 link 的 CSV 行时，图中补零。补零只用于画连续时间序列，
summary 的 active percentile 不把这些补零点加入样本。

第三行是唯一的聚合面板：output 按 `src_rank` 把同一时间桶的全部
`Host -> L0` plane 链路求和；input 按 `dst_rank` 把全部 `L0 -> Host` plane 链路
求和。`--planes` 必须与仿真配置一致，聚合容量线为：

```text
aggregate_line_rate = planes * per_plane_line_rate
peak_utilization = peak_throughput / aggregate_line_rate
peak_headroom = 1 - peak_utilization
```

### 7.2 吞吐包含什么

sampler 记录通过对应 queue 的全部 packet bytes，包括数据包以及实际经过该方向链路
的协议控制包。因此图表示物理链路负载，不等于 DAG 中 `transfer_bytes` 除以时间。

### 7.3 链路数量

MpRail 链路按首次有 packet 经过时注册。图标题中的 `active/discovered` 分母是本次
`link_info.csv` 中发现的链路数，不是拓扑理论端口总数。没有服务器内部流量时，
`Server-local FullMesh` 面板为空是正确结果。

## 8. 测试

```bash
python3 tests/run_mprail_visualization.py
```

测试构造五类 MpRail 链路和多个时间桶，检查：

- 五类名称和坐标解析；
- L0/L1 跨 plane 输入被拒绝；
- 四行七面板 PNG 确实生成且尺寸有效；
- endpoint 容量线按 `plane_count * 400 Gbps/link` 计算；
- 专用测试中 endpoint output 的 400 Gbps peak 得到 100% utilization 和 0% headroom；
- summary 的链路数、总字节、line rate 和最大队列正确；
- inventory 能恢复 rank/rail/plane/spine/bundle。

测试输入、命令、PNG、CSV 和中文报告统一保存在：

```text
test_logs/run_<timestamp>_mprail_visualization/
```
