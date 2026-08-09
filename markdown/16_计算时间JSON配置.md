# 计算时间 JSON 配置

## 1. 核心语义

计算 JSON 保存的是每个 operation 处理一个对应 token 的时间，不再保存某个
balanced workload 的固定模块总时间：

```text
compute_us = task_token_count * selected_us_per_token
```

`selected_source` 仍然在理论值和 profiling 值之间二选一：

```text
theoretical -> theoretical_us_per_token
profiled    -> profiled_us_per_token
```

这样同一个 layer 内不同 rank 收到不同数量的 expert tokens 时，`expert_ffn`
时间会自然不同。HTSim 不读取 JSON；Python generator 在生成 DAG 时完成乘法并把
结果写入 `compute_us`。

## 2. 配置文件

当前标准文件：

```text
pysrc/compute_profiles/H100_DSV3_EP32_compute_4096tpr.json
pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json
```

ProbeEP 两层 full 和静态算法 full runner 的默认完整实验 profile 为
`H20_DSV3_EP32_compute_4096tpr.json`。H100 profile 保留为显式对照，不再是默认。

文件名中的 `4096tpr` 表示它服务于标准 DSV3 4096 tokens/rank 实验，并不表示
JSON 内保存 4096 tokens 的固定总时间。Attention 的单 token 时间仍与固定的
sequence length、hidden shape 和 kernel 定义有关，因此配置不是跨模型通用常数。

H20 理论 profile 固定使用 `148 TFLOP/s` dense BF16、78 个 SM、通信
预留 20 个 SM；因此 Expert FFN 的 overlap 计算峰值为
`148 * (78-20) / 78 = 110.0513 TFLOP/s`。该数值是理论占位，不是
H20 上的 grouped-GEMM profiling 结果。

| profile | dense BF16 | total SM | communication SM | overlap compute peak |
|---|---:|---:|---:|---:|
| H100 SXM | 989 TFLOP/s | 132 | 20 | 839.1515 TFLOP/s |
| H20 SXM | 148 TFLOP/s | 78 | 20 | 110.0513 TFLOP/s |

## 3. Schema v2

```json
{
  "schema_version": 2,
  "hardware": "H100_SXM",
  "selected_source": "theoretical",
  "time_unit": "us",
  "scaling": "duration_us = token_count * selected_us_per_token",
  "total_sms": 132,
  "communication_sms": 20,
  "modules": {
    "expert_ffn": {
      "token_kind": "routed_expert_token",
      "theoretical_us_per_token": 0.10496362386248749,
      "profiled_us_per_token": null
    }
  }
}
```

| 字段 | 语义 |
|---|---|
| `schema_version` | 当前必须为 `2`；旧的固定总时间 schema v1 不再接受 |
| `hardware` | GPU/加速器名称，写入 manifest |
| `profile_scope` | hidden、sequence、top-k 等适用 shape |
| `selected_source` | `theoretical` 或 `profiled` |
| `time_unit` | 固定为 `us` |
| `total_sms` | GPU 总 SM |
| `communication_sms` | overlap 时静态预留给通信的 SM |
| `token_kind` | 当前 operation 的乘数具体代表哪种 token |
| `*_us_per_token` | 每个对应 token 的理论或 profiling 时间 |

所有实际执行到的模块都必须存在。选中字段为 `null`、零、负数或非数字时立即
失败，不允许在 theoretical/profiled 之间静默回退。

## 4. 各模块 token 口径

| operation | `token_kind` | task 使用的 `token_count` |
|---|---|---:|
| `attention` | `source_token` | 当前 rank 的输入 tokens |
| `router_projection` | `source_token` | 当前 rank 的输入 tokens |
| `expert_ffn` | `routed_expert_token` | 当前 execution rank 的 expert routes；MoonEP 使用 padding 后 routes |
| `combine_reduce` | `source_token_with_topk8` | 当前 source rank 的输入 tokens |
| `combine_final_reduce` | `source_token_with_topk8` | 当前 source rank 的输入 tokens |
| `per_server_planning_proxy` | `server_routed_expert_token` | 当前 server 内需要规划的 expert routes |

`per_server_planning_proxy` 仅供 MoonEP 的 GPU 理论占位 task 使用，不是通用 CPU
planner 时间。ProbeEP 不读取该配置项，也不生成对应 task；ProbeEP planner 和 route
lowering 均为离线 workload 生成步骤，不进入 compute profile、DAG 或 makespan。

`expert_ffn` 的一个 routed expert token 是一条 token-to-expert route。DSV3
`topk=8` 时，一个 source token 会产生 8 个 routed expert tokens。NCCL 和 DeepEP
按原始 expert placement 聚合；EPLB 按 physical expert placement 聚合；MoonEP
按实时 execution rank 聚合并加入 kernel padding。

例如理论 FFN 单价为：

```text
0.10496362386248749 us / routed expert token
```

则：

```text
32768 routes -> 3439.448 us
100817 routes -> 10582.118 us
```

这正是热点专家导致计算 straggler 的建模入口。

## 5. 理论值来源

当前 H100 理论单价由对应 operation 的 FLOP 公式除以 H100 BF16 dense 理论峰值，
再除以该 operation 的 token 数得到。它是线性理论占位，不模拟小 GEMM 效率、
grouped GEMM shape、HBM、launch overhead 或真实 occupancy。

JSON 模式沿用当前静态 SM 约定：task metadata 会记录 132 总 SM 和通信预留 20 SM；
单 token 单价本身不因是否 overlap 再做隐式缩放。需要区分 overlap/non-overlap 的
实测效率时，应定义明确的 profile，而不是在读取器里隐藏乘数。

## 6. Profiling 值

profiling 完成后，把同一 operation 的实测时间除以本次测量采用的准确
`token_count`，得到：

```json
"profiled_us_per_token": 0.0875
```

然后将 `selected_source` 改为 `profiled`。这个线性化值只适用于对应
`profile_scope`。真实 grouped GEMM 通常不是严格线性；如果需要更高精度，应另外
开发按 token bucket/shape 查表，而不是把某个总时间伪装成通用 per-token 值。

CLI 可覆盖 JSON 内的选择：

```bash
python3 pysrc/generate_moe_dag.py \
  --output generated_workloads/example \
  --compute-config pysrc/compute_profiles/H100_DSV3_EP32_compute_4096tpr.json \
  --compute-time-source theoretical
```

完整 H20 ProbeEP 动态闭环实验显式传入 profile：

```bash
python3 tests/run_probeep_2layer_ratio_full.py \
  --full --gate-layer-map 0,1 \
  --compute-config pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json
```

不再有 baseline/replay 两份 profile 的混用问题。同一份 JSON 用于单个
HTSim 进程中所有动态追加的 compute task。

## 7. Manifest 和 task 审计

`manifest.json.metadata.compute_cost` 保存：

```text
model = json_linear_per_token_v2
config_path / hardware / profile_scope / selected_source
每个 module 的 token_kind 和两种 us_per_token
```

每个 compute task 的 `task_map.json.metadata` 还保存：

```text
compute_token_count
compute_us_per_token
compute_token_kind
cost_source
```

因此可以直接核对：

```text
task.duration_us == compute_token_count * compute_us_per_token
```

必须同时保存 workload 生成物和使用的 JSON 版本。只保存 `.dag` 可以重放网络，
但无法解释计算时间采用了什么 token 口径。

## 8. 测试要求

`tests/run_workload_generator.py` 必须验证：

1. schema v1 固定总时间配置被拒绝；
2. theoretical/profiled 二选一且不静默回退；
3. `expert_ffn` token 数翻倍时 `compute_us` 翻倍；
4. 不均匀 Gate 下各 rank 的 FFN 时间随真实 route 数变化；
5. MoonEP 使用 padded routes，而不是未 padding 的 real routes；
6. task metadata 中的 count、单价和最终时长乘法自洽。
