# 计算时间 JSON 配置

## 1. 目标

计算时间与 DAG 结构、MoE 算法和 HTSim 网络仿真解耦。每个 compute 模块在 JSON
中同时保存一个理论时间和一个 profiling 时间，再通过 `selected_source` 二选一：

```text
theoretical -> 使用 theoretical_us
profiled    -> 使用 profiled_us
```

选中的数值直接成为 `.dag` 的固定 `compute_us`。HTSim 不读取 JSON，也不在运行时
重新计算 FLOP；JSON 由 Python workload generator 读取并降低到 DAG。

## 2. 目录和命名

配置统一放在：

```text
pysrc/compute_profiles/
```

当前配置：

```text
H100_DSV3_EP32_compute_4096tpr.json
```

该文件对应标准 `4096 tokens/rank/microbatch`。文件名必须携带 token-per-rank
scope，禁止再使用无法判断 shape 的
`H100_DSV3_EP32_compute.json`。

文件名应同时体现硬件和必要的 workload scope。以后 H20 profiling 可以新增：

```text
H20_compute.json
H20_DSV3_EP32_compute.json
```

不要修改 Python 代码来切换硬件时间，只新增或修改 JSON。

## 3. JSON 格式

当前文件结构为：

```json
{
  "schema_version": 1,
  "hardware": "H100_SXM",
  "profile_scope": "DSV3 4-algorithm comparison baseline; EP32; tokens_per_rank=4096",
  "selected_source": "theoretical",
  "time_unit": "us",
  "total_sms": 132,
  "communication_sms": 20,
  "modules": {
    "attention": {
      "theoretical_us": 2188.73965337108,
      "profiled_us": null
    },
    "expert_ffn": {
      "theoretical_us": 3439.44802672599,
      "profiled_us": null
    }
  }
}
```

字段语义：

| 字段 | 语义 |
|---|---|
| `schema_version` | 当前固定为 `1` |
| `hardware` | GPU/加速器名称，写入 manifest |
| `profile_scope` | 该组固定时间适用的 shape、batch、算法和并行范围 |
| `selected_source` | `theoretical` 或 `profiled` |
| `time_unit` | 固定为 `us` |
| `total_sms` | 该硬件总 SM，用于 task metadata |
| `communication_sms` | overlap task 的静态通信 SM 预留 |
| `modules` | operation 名到两种固定时间的映射 |

JSON 不允许注释。说明信息写进 `profile_scope` 或本 Markdown，不在 JSON 中加入
非标准注释语法。

## 4. 模块名称

当前 generator 可能产生以下 compute operation：

| 模块 | 含义 |
|---|---|
| `attention` | 当前 coarse Attention 占位 |
| `router_projection` | router/gate projection |
| `expert_ffn` | routed expert FFN |
| `combine_reduce` | NCCL/EPLB/MoonEP combine reduce |
| `combine_final_reduce` | DeepEP combine final reduce |
| `per_server_planning_proxy` | MoonEP server-local planning 占位 |

配置可以包含当前 workload 没有使用的模块。实际生成过程中一旦遇到缺失模块，立即
报错，不允许回退到另一个模块或 H100 默认公式。

## 5. 理论与 profiling 二选一

初始阶段使用：

```json
"selected_source": "theoretical"
```

完成 profiling 后，为所有会执行的模块填入正数：

```json
"profiled_us": 7.5
```

然后切换：

```json
"selected_source": "profiled"
```

选中的字段如果是 `null`、零、负数或非数字，workload 生成立即失败。系统不会从
`profiled_us` 静默退回 `theoretical_us`，否则同一实验可能混合两种计算口径。

CLI 也可以临时覆盖 JSON 内的选择：

```bash
python3 pysrc/generate_moe_dag.py \
  --output generated_workloads/example \
  --compute-config pysrc/compute_profiles/H100_DSV3_EP32_compute_4096tpr.json \
  --compute-time-source theoretical
```

`--compute-time-source` 必须与 `--compute-config` 同时使用。

## 6. 固定时间的适用边界

模块时间不是跨 shape 通用常数。例如 `expert_ffn` 会受到 token route 数、padding、
expert placement 和算法影响。一个 balanced-permuted 基线的固定时间不能直接代表
MoonEP/EPLB 改变 execution placement 后的负载 shape。

因此每份 JSON 必须准确写 `profile_scope`。以下任一参数改变时，应重新确认或新建
配置：

- hidden、MoE hidden、sequence、token 数；
- expert 数、top-k、padding；
- dtype、kernel fusion、通信 overlap 条件；
- GPU 型号、频率、软件版本；
- 算法导致的 per-rank expert token shape。

`H100_DSV3_EP32_compute_4096tpr.json` 服务于正常
`4096 tokens/rank/microbatch` 的训练/prefill 四算法对比基线。当前固定
`expert_ffn` 时间按 balanced-permuted routing 下每 rank 32768 routes 的平衡参考
shape 计算；
因此本轮比较主要隔离通信和 planning 差异，不声称已反映算法改变实际
per-rank expert shape 后的 kernel 时间差异。

JSON 模式仍保留每个 task 的理论 `operation_flops` 供审计，但实际
`compute_us` 完全取所选 JSON 数字。需要比较不同 placement 的真实计算
时间时，应用 profiling 值或为各 shape 建立独立 profile，不应误读该基线。

同一模块的这个数字同时用于普通窗口和 communication-overlap 窗口，不再按可用
SM 比例二次缩放；`available_sms` 仍按 JSON 的 `total_sms` 和
`communication_sms` 写入 task metadata。如果两种窗口必须使用不同 profiling
时间，应先把它们定义成两个明确的 operation，而不是在读取阶段隐式乘系数。

## 7. Manifest 与复现

生成后的 `manifest.json` 在 `metadata.compute_cost` 中记录：

```text
model
config_path
hardware
profile_scope
selected_source
total_sms
communication_sms
modules
```

因此一个实验必须同时保存 workload 生成物和使用的 JSON commit。只保存 `.dag`
虽然可以重放 HTSim，但无法解释计算时间来自理论值还是 profiling 值。
