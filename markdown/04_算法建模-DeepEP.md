# 算法建模：DeepEP

## 1. 版本边界

DeepEP 不是一个固定通信流程。生成器必须显式选择版本和模式，不能只写 `algorithm=deepep`。

本文以本地仓库 `/home/xuheng/DeepEP` 的 `11d1cab` 为代码阅读基线，区分：

| 模式 | 主要用途 | 数据路径 |
|---|---|---|
| V1 normal | 训练、prefill | NVLink + RDMA hierarchical forwarding |
| V1 low-latency | decode | 官方定义为 pure RDMA，强调低延迟和 hook overlap |
| V2 hybrid | 多节点高吞吐 | scale-out RDMA 后 scale-up NVLink forwarding |
| V2 direct | 直接模式 | rank 间直接通信，不做 hierarchical scale-up forwarding |

当前生成器实现两个 forward 选项：`deepep-hybrid` 按 destination server 去重并
显式展开 fabric/本地转发，`deepep-direct` 按 source/destination rank 直接传输。
它们是本文数据路径的 workload 模型，不冒充 DeepEP kernel 仿真。V1
low-latency、V1 normal 专属细节和 backward 尚未实现。

## 2. Rank 与 domain

假设每个服务器有 `G` 张 GPU：

```text
global_rank = node_id * G + local_rank
scaleout_rank = node_id
scaleup_rank = local_rank
```

DeepEP V2 hybrid 在代码中设置：

```text
num_scaleout_ranks = num_rdma_ranks
num_scaleup_ranks  = num_nvlink_ranks
```

跨节点传输先到目标 scale-out domain 中与发送者相同 scale-up coordinate 的 relay，再由 forward warp 在目标 NVLink domain 内发送到实际 expert rank。

V2 direct 则令逻辑 scale-out 为 1、scale-up 为全部 ranks，使用 full GIN connectivity，不使用上述两级 forwarding。

## 3. Forward phase

### 3.1 Router 与 layout

输入为每个 source rank 的：

```text
x[S, H]
topk_idx[S, K]
topk_weights[S, K]
```

建模 task：

```text
router_projection
topk_selection
dispatch_layout / notify
```

V1 normal 显式计算每 rank、RDMA rank、expert 的 token count。V2 在 dispatch 中生成 receive count、slot、forward metadata 和 handle。layout/notify 是 compute/control cost，不应吞并数据面 FCT。

### 3.2 Hybrid dispatch 的复制规则

对 source token `t`：

```text
D_node(t) = 选择的 expert 所在 destination nodes 集合
D_rank(t, n) = node n 内被选择的 destination ranks 集合
```

跨节点 scale-out hop：

```text
每个 (token, destination node) 发送一次 hidden payload
src_rank -> relay(destination_node, local_rank(src_rank))
```

目标节点 scale-up hop：

```text
relay -> D_rank(t, destination_node) 中每个 unique rank
```

一个 token 在同一 destination node 选择多个 expert 时，fabric 仍只发送一份；若多个 expert 位于同一 destination rank，本地 hop 也只发送一份 hidden，并附带多个 top-k metadata。

因此 fabric dispatch 字节为：

```text
sum_t |D_node(t) - {source_node}| * dispatch_token_bytes
```

而不是：

```text
S * K * dispatch_token_bytes
```

同 node token 不走 fabric，直接执行 scale-up local transfer 或 local bypass。

### 3.3 为什么不能直接为每个 expert 写 `server_forward`

如果为每个 destination expert rank 生成：

```text
src -> relay -> expert_rank
```

那么同一个 token 选择目标节点内多个 rank 时，src 到 relay 的 fabric payload 会被重复多次。这与 DeepEP 的 domain 去重不符。

正确的 IR 是：

```text
每个 destination node: 一个聚合 fabric transfer task
每个 unique destination rank: 一个依赖对应 fabric barrier 的 local fanout task
```

当前生成器按 `chunk_tokens` 拆分，每个 `fabric_chunk[i]` 与对应
`local_fanout_chunk[i]` 使用独立 task/barrier。真实 `hybrid_dispatch.cuh` 可以在
一条传输内部按更细粒度消费 buffer；本模型只在 flow/chunk 完成点释放依赖。

### 3.4 Expert compute

dispatch 完成后每个 rank 按 local expert token count 执行 grouped GEMM：

```text
gate/up GEMM -> activation -> down GEMM
```

当前用每个 rank 的实际 routed assignment 数核算 expert GEMM FLOP，再根据是否与
通信重叠选择单卡 `989e12 FLOP/s` 或预留 20 SM 后的 `839.15e12 FLOP/s`，得到
固定 `compute_us`。当前 DeepEP builder 尚未加入 kernel alignment、expanded mode
或 capacity padding；这些不能从当前结果中反推。

### 3.5 Hybrid combine

combine 不应简单假设“dispatch 的每条 flow 原样反向”。DeepEP V2 可以在 scale-up/scale-out domain 内执行 partial reduction，`allow_multiple_reduction` 还会改变 reduction 次数与传输量。

第一层 planner 应根据 forward handle 和 combine mode 生成：

```text
local expert outputs
  -> scale-up partial reduction / relay gathering
  -> scale-out partial payload
  -> origin domain final reduction
  -> origin token output
```

首版采用保守规则：

1. 每个参与目标 node 对原 token 产生一份 partial output；
2. node 内先归并到 relay；
3. relay 向 origin 发送一份 BF16 partial output；
4. origin 执行最终本地 reduce。

若配置与所读 DeepEP mode 不一致，生成器必须拒绝，而不是套用该近似。

## 4. Route 映射

### 4.1 单一目的 rank 的 coarse 表达

当一次聚合 transfer 确实只有一个 destination rank 时，可使用：

```text
server_forward src_relay:<source-side-relay> dst_relay:<destination-side-relay>
```

例如 dispatch 从 rank 2 到另一个节点的 rank 13，且 `G=8`：

```text
source local coordinate = 2
destination relay = 8 + 2 = 10
logical flow: 2 -> 13
route server_forward src_relay:2 dst_relay:10
```

这表示 fabric 到达 rank 10，再执行 `10 -> 13` 的本地转发。

### 4.2 one-to-many

当 relay 需要向多个 destination rank 扇出时，不使用多条完整 server_forward。应生成一次 `src -> relay` fabric task 和多条 `relay -> dst` local task，并通过 phase graph 关联。

现有 `.dag` 已能把它表达成两组普通 network task：跨服务器的 `src -> relay` 会自然进入 fabric，同服务器的 `relay -> dst` 会自然进入本地 FullMesh。manifest 负责标记两者属于同一个逻辑 forwarding phase，不需要先扩展 route 语法。

不应把所有 fabric transfer 合并成一个 barrier。每个 local fanout 只依赖自身对应的 fabric barrier，因此不同 destination node 可以独立推进。若首版每个 transfer 仍是完整消息，则它不能表达单条消息内部的 chunk pipeline；精确模式需要继续拆分 chunk task。

节点内 scale-up transfer 映射到 MpRail local FullMesh 是近似：它保留 rank、字节和本地链路竞争，但不复现 NVLink TMA/store、barrier 和 GPU memory ordering。首版不单独估算这些本地 helper kernel，后续再用 profiling 或资源模型补齐。

## 5. Dtype 与 metadata

典型配置使用 FP8 dispatch、BF16 combine：

```text
dispatch hidden bytes = H * 1 + scale factor bytes
combine output bytes  = H * 2
```

每个 token 还可能携带：

- source global token index；
- top-k expert indices；
- top-k weights；
- destination slots / linked-list metadata；
- count、tail 和 notification。

当前只实现 `payload_only`：dispatch 为 `H * bytes(dispatch_dtype)`，combine 为
`H * bytes(combine_dtype)`。scale、top-k metadata、barrier、tail、counter 和 atomic
没有创建额外 flow。加入 `data_plus_metadata` 前必须先定义精确 layout，不能用统一
比例放大 hidden payload。

## 6. Communication-compute overlap

### 6.1 首版目标

首版只需要复现 DeepEP 流水线图中的核心关系：通信 flow/chunk 与没有依赖冲突的计算 task 同时推进。

```text
network/chunk task  --------->  UEC/MpRail FCT
compute task        --------->  fixed compute_us
                         overlap
```

目标模式中的 SM 使用固定近似：

```text
H100 SXM total SMs = 132
communication SMs per rank = 20
compute SMs during overlap = 112
overlap BF16 peak ~= 839.15 TFLOP/s
```

20 SM 按一个 rank 上的逻辑 dispatch/combine kernel 预留。该 kernel 即使产生多个 destination flow 或多个显式 chunk flow，也只预留一次 20 SM。网络 FCT 不因该参数改变；它只把重叠 compute task 的固定理论算力从 989 TFLOP/s 降到约 839.15 TFLOP/s。

这是本项目固定的实验假设，不是 DeepEP 的通用常量。当前 DeepEP V2 使用 `get_theoretical_num_sms()` 计算建议值，也允许调用方通过 `num_sms` 覆盖；其 README 性能表中的不同拓扑使用过 12 或 6 SM，V1 legacy 示例使用过 24 SM。首版固定 20 是为了避免开发动态 occupancy 和 kernel scheduler。

### 6.2 Low-latency 模式

V1 low-latency 官方用途是 decode，网络为 pure RDMA，并支持 receiving hook，使网络流量在后台进行且不占用计算 SM。

barrier DAG 可以表达 hook 的 microbatch/task-level 依赖，但首版不模拟 receiving-hook/persistent-kernel 的特殊行为。目标中的固定 20 SM 只用于普通 overlap 配置，不能套到官方声明为零 SM 占用的 low-latency hook。

首版不实现 low-latency；文档和配置中必须明确拒绝：

```text
deepep.mode = v1_low_latency
```

直到确实需要研究该模式时再单独定义；当前网络与普通 DeepEP overlap 不依赖它。

## 7. Backward

DeepEP API 语义：

```text
dispatch backward = combine
combine backward  = dispatch with cached handle
```

训练 graph 需要复用 forward handle 的 placement/slot，而不是重新采样 routing。还要增加 expert GEMM backward、router backward 和框架的 DP/TP gradient communication。

首版 forward 不生成伪 backward。

## 8. 配置草案

```yaml
algorithm: deepep
version: v2
mode: hybrid
dispatch_dtype: fp8
combine_dtype: bf16
expert_alignment: 1
expanded: false
allow_multiple_reduction: true
forwarding_fidelity: full_message_barrier
metadata_mode: payload_only
```

## 9. 校验与测试

- 相同 token 对同一 destination node 的 fabric payload 只出现一次。
- 同 node 多 destination ranks 只增加 local fanout，不增加 fabric copy。
- 同 rank 多 experts 对 hidden payload 去重。
- 所有 relay 的 local coordinate 与算法配置一致。
- dispatch 和 combine 分别使用正确 dtype 字节。
- 每 rank expert compute token count 与 dispatch 输出一致。
- skew routing 下 hot rank 的 compute 时间自然变长。
- coarse forwarding 在报告中明确标注，不与 chunk pipeline 混淆。
- direct/hybrid/low-latency mode 不能静默互相降级。

## 10. 代码依据

- [DeepEP 官方仓库](https://github.com/deepseek-ai/DeepEP)。
- `/home/xuheng/DeepEP/README.md`：V2 API、hybrid/direct、dispatch/combine 和 SM/QP 配置。
- `/home/xuheng/DeepEP/docs/legacy.md`：V1 normal、pure-RDMA low-latency 和 forward/backward API。
- `/home/xuheng/DeepEP/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`：scale-out 去重、RDMA put 和 scale-up forwarding。
- `/home/xuheng/DeepEP/deep_ep/include/deep_ep/impls/hybrid_combine.cuh`：hierarchical combine 与 replay metadata。
- `/home/xuheng/DeepEP/csrc/kernels/backend/nccl.cu`：physical/logical RDMA 与 NVLink domain 映射。
