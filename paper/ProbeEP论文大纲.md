# ProbeEP NSDI 论文大纲

本文给出 ProbeEP 的论文级叙事、motivation、核心 insight、章节结构和证据链。
这里描述的是系统问题与设计思想，不按当前 Python 类、函数或 manifest 字段组织正文。
具体实验参数、运行命令和画图入口统一放在 `experiment/`。

## 写作总则：一句话主线

> 现有 MoE 系统要么优化 token 通信而保留倾斜的专家计算，要么只在高速本地
> fabric 内复制专家，因而无法消除跨服务器计算拖尾。ProbeEP 发现，流水化 MoE
> 执行中存在随阶段和负载变化的 NIC overlap headroom；它把这一 headroom 转化为
> 有预算的跨服务器专家迁移，在不让权重流量成为新关键路径的前提下缩短最慢
> GPU 的专家计算。换言之，ProbeEP 不是继续减少通信，而是主动花费可隐藏的通信
> 来消除不可隐藏的计算拖尾。

英文核心句：

> ProbeEP trades otherwise hidden network headroom for a shorter MoE compute
> critical path.

## 核心 Insight

### Insight 1：局部均衡有一个跨服务器上限

DeepEP 一类通信优化减少 token 重复传输并利用服务器内 relay，但不改变专家在哪台
服务器执行。MoonEP/UltraEP-style 本地复制可以均衡一台服务器内部的 GPU，却仍受
每台服务器原始专家热度总量约束。只要热门专家集中在少数服务器，最快的本地
均衡也无法降低最忙服务器的计算下界。

这个观察必须通过两级负载图讲清楚：

```text
rank imbalance  --local replication-->  lower rank imbalance
server imbalance --------------------->  unchanged lower bound
```

因此，跨服务器迁移不是一个可选优化，而是突破 local-replication ceiling 的必要条件。

### Insight 2：专家权重很大，但它的暴露成本不等于线速传输时间

直觉上，跨服务器复制完整专家会增加大量 RDMA 流量，因此现有设计通常把它排除在
关键路径之外。这个判断忽略了 MoE 的双流流水线：一部分 Weight+Dispatch 通信可以
与另一个 microbatch 的 Attention 或 Expert FFN 重叠。真正应比较的不是“权重字节数
是否大”，而是：

```text
exposed migration cost
  = max(0, Weight+Dispatch stage - overlapped compute window)

critical-path compute saving
  = old straggler compute - new straggler compute
```

只要后者大于前者，跨服务器复制就是净收益。热点专家承载的 route 越多，固定权重
成本被摊销得越充分；padding 越严重，迁移一个专家块可能消除的拖尾越大。

### Insight 3：可用 headroom 不能靠静态带宽模型决定

可隐藏的迁移量同时受 Gate 分布、token dispatch、计算硬件、网络竞争和当前流水线
阶段影响。Attention 窗口相对稳定，MoE 窗口随实际执行放置和 padding 改变；同一个
固定预算不可能在所有层、microbatch 和硬件上都合适。

ProbeEP 因此不预测每条 flow 的完成时间，而是直接测量上一同类 overlap window 中的
`network/compute` 比例，用实际瓶颈 NIC 的 Token+Weight 字节反算下一窗口预算。反馈
只决定“可以迁移多少”，不改变 token dispatch 的正确性，也不把网络模型混入计算
均衡目标。

### Insight 4：稀缺 RDMA 与充裕本地 fabric 要分层使用

跨服务器 RDMA 只负责突破 server-level 下界；进入目标服务器后，专家副本和 token
再由高速本地 fabric 分散到 GPU。由此得到三个职责独立但连续执行的模块：

```text
global compute planner
  -> feedback-bounded migration admission
  -> local padding-aware packing and pipelined transfer
```

这种分层避免了两个极端：既不把所有负载均衡都限制在本地，也不为每个细粒度
route 盲目跨服务器复制权重。

## 与相关工作的定位

| 工作 | 核心做法 | ProbeEP 与它的本质区别 |
|---|---|---|
| NCCL / direct All-to-AllV | 直接交换 token，保持原始专家放置 | 不处理计算倾斜，也不利用重复路由和层级局部性 |
| DeepEP | 用层级 relay、去重和 overlap 降低 token 通信 | 优化通信路径，但保留 server-level expert compute skew |
| EPLB | 用历史负载维护持久专家布局 | 预测可能过期，重布局频率低，不能逐窗口消费实时 NIC headroom |
| MoonEP / UltraEP-style baseline | 在高速本地或 rack-scale fabric 内实时复制并 reroute | 依赖大的 scale-up 域；在普通多服务器 RDMA 集群中仍有 locality ceiling |
| FAST | 用更快的本地 fabric 重塑较慢的 All-to-AllV 通信 | FAST 主要消除通信拖尾；ProbeEP 主动增加可隐藏通信以消除计算拖尾 |
| ProbeEP | 反馈式跨服务器临时复制，联合全局计算均衡和 NIC admission | 将网络 headroom 变成可控的 compute rebalancing resource |

写作时不要把 UltraEP 说成“错误方案”。它回答的是 rack-scale scale-up 域中的精确
均衡；ProbeEP 回答的是另一个问题：在普通分层 GPU 集群里，能否在慢得多的 scale-out
网络上安全地获得跨服务器均衡收益。

## Claim 边界

| 可以主张 | 需要谨慎或暂不能主张 |
|---|---|
| ProbeEP 是跨服务器、临时、反馈约束的专家复制机制 | 不能把当前 Python planner 写成已完成的 GPU 原生在线实现 |
| 单 HTSim 进程内完成 observation、budget 更新和后续 DAG append | 不能把离线 route lowering 时间隐含为零开销 |
| 完整建模 forward 中 Attention、Router、Weight、Dispatch、Expert、Combine 的依赖与 overlap | 当前不是完整训练迭代，尚未包含 backward、梯度规约和 optimizer state |
| 完整专家权重作为离散 admission 单元，token 不被 budget 裁剪 | 当前未建模 HBM 容量、CUDA kernel 资源竞争和真实 weight-copy kernel |
| 包级网络仿真可以支持网络趋势与机制分析 | 不能只凭仿真宣称生产训练吞吐或真实 GPU kernel 性能 |

论文首版建议把主场景写成 **MoE forward path for training and prefill**。若要把标题和
摘要收窄为“training”，必须先补齐 backward 副本梯度规约和训练迭代结果。

## 标题候选

首选：

> ProbeEP: Trading Communication for Computation in Mixture-of-Experts Systems

更具体的备选：

> ProbeEP: Feedback-Guided Cross-Server Expert Replication for MoE

> ProbeEP: Turning Network Headroom into Balanced MoE Compute

标题里优先保留 `ProbeEP`、`communication for computation` 或
`cross-server expert replication`。不要把 planner 的具体启发式名称写进标题。

## 0. Abstract

摘要按四段短句组织。

| 段落 | 内容 |
|---|---|
| 问题 | Large-EP 放大动态专家倾斜；通信优化和本地复制都无法消除跨服务器计算拖尾 |
| 反直觉机会 | 跨服务器权重复制通常被认为过贵，但流水化执行中存在可隐藏、且随阶段变化的 NIC headroom |
| 方法 | ProbeEP 全局规划计算迁移，以实测 overlap 反馈限制完整专家 admission，并在多个 NIC rail 上流水化权重与 token dispatch |
| 结果 | 用仿真主结果、负载均衡质量、网络开销、反馈适应性和 prototype 占位结果支撑；所有数字最后回填 |

摘要不要列算法内部字段，不要写开发状态，不要用“我们设置一个固定预算”描述核心。

## 1. Introduction

Introduction 建议九段，每段只完成一个任务。

| 段落 | 任务 |
|---|---|
| 1 | 说明 MoE 依靠 Expert Parallelism 扩展，forward 中 Dispatch、Expert FFN 和 Combine 共同决定关键路径 |
| 2 | 说明 Gate 引入的动态倾斜同时形成通信热点和计算拖尾，large EP 把 expert skew 直接放大为 rank/server skew |
| 3 | 回顾通信方向：NCCL、DeepEP、FAST 主要降低 All-to-AllV 成本，但不消除专家计算的 server-level 下界 |
| 4 | 回顾 placement 方向：EPLB 使用历史，MoonEP/UltraEP-style 实时复制依赖本地或 rack-scale 高速域 |
| 5 | 提出 gap：普通多服务器集群的单机 scale-up 域太小，而跨服务器权重复制被认为会占满 NIC、恶化关键路径 |
| 6 | 给出关键观察：MoE 双流流水线中，通信的暴露成本由 overlap window 决定，网络 headroom 可以换取更短的计算拖尾 |
| 7 | 给出 ProbeEP：全局 server-first 迁移、反馈式 NIC admission、本地 padding-aware packing、rail-aware chunk pipeline |
| 8 | 解释系统意义：ProbeEP 不追求零通信，而是在 compute-bound 与 network-bound 之间维持受控边界 |
| 9 | 列贡献，并预告仿真和 prototype 证据 |

贡献建议写成四条：

| 贡献 | 内容 |
|---|---|
| C1 | 识别并量化本地专家复制的跨服务器均衡上限，以及跨服务器迁移的 overlap 获利区间 |
| C2 | 提出计算优先的两阶段迁移规划，将 server-level padded compute 与 rank-level packing 分离 |
| C3 | 提出基于真实 Weight+Dispatch observation 的双窗口 NIC controller 和完整专家 admission |
| C4 | 在 dependency-aware、packet-level 仿真中系统评估 ProbeEP，并通过 prototype 章节验证实现路径 |

Introduction 需要一张系统示意图：左侧是 server-local baseline 留下的热服务器，右侧是
ProbeEP 利用多个 rail 迁移完整专家，Weight+Dispatch 与计算窗口重叠，最终缩短最慢
Expert FFN。

## 2. Background and Motivation

本章只建立问题和机会，不提前讲完整算法。

### 2.1 MoE 的通信与计算关键路径

说明 token 经 Gate 路由到逻辑专家，Dispatch 把 token 送到执行实例，Expert FFN 按
实例 batch 执行，Combine 返回结果。强调三点：

- top-k 路由让逻辑 route 数、唯一 token payload 数和专家计算量不是同一个量；
- grouped GEMM 的 padding 使少量 route 变化也可能跨越一个计算块；
- 同步边界由最慢 rank 决定，平均负载不是正确目标。

### 2.2 为什么通信优化仍留下计算拖尾

用 DeepEP-style 例子说明去重和 server relay 可以减少 RDMA 字节，却不会改变热专家的
执行服务器。通信完成后，冷服务器先结束 Expert FFN，热服务器仍在计算。由此引出：

> Reducing token movement does not necessarily shorten the expert-compute
> critical path.

### 2.3 为什么本地实时复制仍不够

用一个多服务器例子说明：每台服务器内部都能做到理想均衡，但各服务器总 route 数
不同。MoonEP/UltraEP-style local replication 只能把每台服务器的总量除以本地 GPU 数，
不能把热服务器的工作交给冷服务器。

本节对应两张 motivation 图：Gate/server skew 分布，以及 local-replication ceiling。

### 2.4 为什么跨服务器复制看起来不可行

公平陈述三个困难：

- 专家权重远大于单个 token payload，复制是离散固定成本；
- Weight 与 Dispatch 共享 NIC，盲目迁移可能制造新的网络关键路径；
- 可用预算随计算阶段、Gate、硬件和拥塞改变，静态 replica cap 不稳健。

### 2.5 Opportunity：通信换计算的获利区间

定义一个概念模型，不绑定实现变量名：

```text
Gain = critical-path compute reduction - exposed migration delay
```

用 heatmap 展示在不同热点强度、专家状态大小和网络/计算比下存在显著正收益区域。
这里的目的不是证明 ProbeEP 已经最优，而是证明“跨服务器复制一律太贵”这个假设不成立。

### 2.6 Design Requirements

| 要求 | 原因 |
|---|---|
| 全局计算感知 | 必须突破 server-local lower bound |
| padding 感知 | raw route 均衡不等于 FFN 时间均衡 |
| 完整专家 admission | 不能用部分权重执行专家，也不能把权重当连续流量 |
| 反馈而非静态预算 | headroom 随窗口动态变化 |
| per-endpoint 双向约束 | 任一 source TX 或 destination RX 都可能成为瓶颈 |
| 无全局 Weight barrier | 慢 rail 不应阻塞无关 Dispatch |
| 与 token correctness 解耦 | migration budget 不允许丢弃或裁剪 token |

## 3. ProbeEP Overview

### 3.1 Communication-for-Computation

先用一张简化时间线说明 ProbeEP 的选择：在原方案中，network 提前结束但 Expert FFN
有长尾；ProbeEP 在 overlap window 中加入专家权重通信，使 network 更接近但不超过
计算窗口，同时缩短 Expert FFN 尾部。整篇论文都围绕这张图解释。

### 3.2 End-to-End Workflow

```text
exact Gate load
  -> global padded-compute migration intents
  -> feedback-bounded full-expert admission
  -> rail-aware weight chunks + hierarchical token dispatch
  -> local replica packing
  -> Expert FFN and Combine
  -> observe Weight+Dispatch window
  -> update the next same-kind budget
```

强调 planner、controller 和 scheduler 的接口，而不是实现语言：planner 只决定计算上
值得迁移什么；controller 决定网络上允许迁移多少；scheduler 决定如何走多 rail。

### 3.3 Two Feedback Chains

Attention 和 Expert FFN 提供不同的可重叠计算窗口。ProbeEP 为两类窗口分别保留状态，
避免用稳定的 Attention 掩盖动态的 MoE，也避免用某次极端 MoE 窗口污染 Attention。

### 3.4 Non-goals

| 非目标 | 边界 |
|---|---|
| 改变 Gate 语义 | logical expert 选择和 top-k 权重保持不变 |
| 裁剪 token | 所有 Dispatch/Combine payload 必须完成 |
| 通用 collective 调度 | ProbeEP 专注 MoE expert state 与 token pipeline |
| 依赖 rack-scale scale-up | 目标是普通 server-local scale-up + scale-out NIC 集群 |
| 在线重训练或改变参数 | 临时副本执行同一专家参数，训练语义需由 gradient reduce 保持 |

## 4. ProbeEP Design

### 4.1 Compute-Centric Migration Objective

目标不是最小化迁移字节，而是最小化 admission 后最慢服务器、再最慢 rank 的 padded
expert compute。以 `(logical expert, destination server)` 为复制单元，以 route chunk
表达可迁移工作，以完整 expert state 表达固定网络成本。

正文需要区分三种量：real routes、padded routes、compute time。算法依据 padded
compute 做决策，real routes 用于 token conservation，compute time 用于跨硬件比较。

### 4.2 Global Server-First Planning

先从所有服务器构造 expert load histogram，识别 donor surplus 和 receiver deficit。
优先处理能降低全局最大 padded compute 的热点 expert，并允许多个 route move 复用同一
远程副本。该阶段输出 migration intents，不假设它们一定能通过网络 admission。

需要给出一个反例说明：按 raw route 精确均分可能增加 padding，而 padding-aware move
可以用略不相等的 route 数获得更低的最大 FFN 时间。

### 4.3 Feedback-Bounded Full-Expert Admission

Admission 逐个检查 intent 是否仍改善当前 compute objective，并验证完整 expert weight
能否同时满足 source TX 和 destination RX budget。若完整权重放不下，就延后该副本；
不能传一部分权重后把它算作有效迁移。

这个设计自然产生 anytime behavior：预算小时接纳少数高收益副本，预算增加时继续接纳
下一批；任何中间状态都保持 token 和模型语义正确。

### 4.4 Intra-Server Padding-Aware Packing

全局 server mapping 确定后，再利用本地高速 fabric 把 `(expert, padding block)` 分配到
GPU。优先复用 home rank 或远程 seed rank，必要时建立本地副本。目标是最小化 rank
计算长尾，而不是让每个 GPU 的 real route 数完全相同。

### 4.5 Overlap-Window Feedback Controller

每个 observation 从共同 release 到该 rank 的 Weight+Dispatch TX/RX 全部结束，包含网络
等待和空隙。控制器比较全局最慢 network stage 与对应的 compute reference，使用瓶颈
endpoint 实际字节反算下一窗口可承载总量，再扣除不可裁剪的 token baseline，得到只供
migration 使用的 budget。

这里要讲四条性质：

- 比例反馈可以一次响应较大的 workload 变化，而不是固定步长慢慢追；
- 理论 line-rate 只作为硬上限，实际可用速率来自 observation；
- token baseline 优先，超预算时首先将 migration 降为零；
- Attention 与 MoE 状态独立更新，Combine 只做 telemetry。

### 4.6 Rail-Aware Weight Scheduling

完整专家切成网络 chunk，仅用于传输流水化，不表示 tensor parallel。每个 chunk 在可用
rail 中选择同时满足 source TX 和 destination RX 的路径，并平衡 directed server-pair
load 与 endpoint total load。

源侧需要本地 scatter 时先送到对应 rail，跨服务器采用 same-rail RDMA，目标侧再 gather
到 seed rank；额外本地实例由 seed 通过高速 fabric 分发。

### 4.7 Per-Source-Rail Immediate Dispatch

同一 source rail 的 remote Weight TX 完成后立即释放该 rail 的 Dispatch fabric TX。
它不等待其他 rail、目的端 gather 或全局权重完成。Expert FFN 仍等待自身所需的 token
和权重，从而同时满足流水化和消费端正确性。

### 4.8 Correctness and Complexity

需要给出以下不变量：

- 每个 logical route 恰好映射到一个 physical expert instance；
- 所有 remote instance 在执行前拥有完整一致的 expert state；
- Dispatch 与 Combine 保持 token multiplicity 和 source identity；
- migration TX/RX 字节守恒，任何 endpoint budget 不超过硬上限；
- 已提交并开始执行的 DAG task 不被反馈过程修改。

复杂度按 histogram planning、padding refinement、local packing、controller 和 chunk
scheduling 分开分析。正文只给渐进复杂度和在线元数据规模，不出现 Python 函数名。

## 5. Implementation

### 5.1 Workload and Execution Model

说明 dependency-aware DAG 如何表达每 GPU 一条 compute stream、一条 communication
stream、双 microbatch overlap 和跨层 wavefront。强调 barrier 是依赖关系，不是人为的
阶段时间。

### 5.2 Dynamic Closed-Loop Simulation

ProbeEP 使用一个持续运行的 packet-level simulator 实例。每个 observation 完成后，
controller 更新状态并原子追加后续层；EventList、拥塞控制和网络队列不重置，也不回写
已经提交的任务。这样反馈结果来自同一条真实仿真时间线，而不是离线 replay。

### 5.3 Data Plane Modeling

说明 token 使用 hierarchical Dispatch/Combine，weight 使用 scatter/RDMA/gather/local
prefetch，所有传输经过同一网络和拥塞控制模型。计算时间来自独立 profile schema，
网络 FCT 由包级仿真产生。

### 5.4 Current Implementation Boundary

正文或 artifact 必须透明说明：当前 planner/reference lowering 在仿真外执行，未计入
makespan；prototype 需要补一个在线 C++/CUDA planner microbenchmark。当前模型是
forward path，backward state restoration、replica gradient reduce、HBM capacity 和
真实 kernel resource contention 留给 prototype 或 future work。

## 6. Evaluation

评估按 FAST 和 LoongX 的粒度组织：一个小节是一类实验，一类实验从端到端性能、机制
指标、资源代价和边界条件等多个角度共同回答一个问题，而不是把每个 subplot 当成一类
实验。具体参数、扫描点和命令不写进论文大纲，全部见 `experiment/实验大纲.md`。

### 6.1 Evaluation Questions

| 问题 | 证据 |
|---|---|
| Q1：ProbeEP 是否缩短端到端关键路径？ | 五算法主结果和不同 compute/network balance |
| Q2：收益是否确实来自突破 local-replication ceiling？ | Gate、server/rank padding、迁移效率和网络暴露分解 |
| Q3：在线反馈是否能把迁移限制在真实 overlap window 内？ | 两条反馈链的时序和 fixed-budget 对照 |
| Q4：面对不同 MoE 负载，ProbeEP 是否稳健？ | Gate family、倾斜、token volume、layer/seed 和持续性 |
| Q5：面对不同硬件 balance，ProbeEP 是否稳健？ | compute、NIC、本地 fabric 和 expert state |
| Q6：面对不同层次拓扑与规模，结论是否仍成立？ | server boundary、path diversity、oversubscription 和 EP scale |
| Q7：每个设计选择是否必要？ | planner、admission、controller 和 chunk pipeline 消融 |
| Q8：代价和适用边界是什么？ | planner/control overhead、metadata scaling 和 break-even region |

### 6.2 Methodology

像 LoongX 一样清楚区分三类证据：

- packet-level simulation：端到端 makespan、queue、link load 和反馈行为；
- analytical opportunity model：只解释 break-even 区域，不替代端到端结果；
- prototype：验证 planner、weight movement 和控制路径可落地。

所有 baseline 使用同一 Gate assignments、计算 profile、拓扑、transport 和 stream
schedule。静态 baseline 与动态 ProbeEP 必须来自等价 workload；ProbeEP 主结果必须使用
同一 simulator PID 内的闭环运行，不能用静态 ProbeEP 代替。

仿真运行在 128-core server 上。每个 HTSim 进程使用一个 CPU core，批量实验最多并发
100 个 HTSim 进程，其余 28 cores 留给构建、数据处理和系统服务。并发只缩短 sweep 的
wall-clock time，不改变单个 case 的仿真语义或报告指标。

所有实验从同一个 Base Case 出发。Base Case 对应现有 EP32/1Plane 的完整 prefill/训练
forward 配置和已经完成的五算法 H20/H100 运行。每类实验只改变一组因素；其余模型、
Gate assignment、算法语义和仿真方法保持不变。论文只描述控制变量思想，精确配置见实验
artifact。

### 6.3 End-to-End Performance

比较 direct All-to-AllV、hierarchical token transport、历史持久布局、本地实时复制和
完整 ProbeEP。该类实验报告端到端 makespan/throughput，并同时覆盖至少两类 compute/network
balance，证明 ProbeEP 不是只对一个硬件点有效。

主结果的解释顺序：

1. direct baseline 同时受 token duplication 和 compute skew 影响；
2. hierarchical transport 降低通信但留下 compute straggler；
3. persistent/local replication 改善 rank balance，却受 server boundary 限制；
4. ProbeEP 支付少量可隐藏 weight RDMA，进一步降低全局 compute tail。

这一实验类型内部同时报告绝对 makespan、相对最强 baseline 的 speedup、有效吞吐以及
不同硬件点上的排序稳定性。不能只给归一化结果而隐藏绝对量级。

### 6.4 Mechanism Analysis

从相同 Base Case 沿完整因果链解释收益：现实 Gate 先形成 server-level skew；MoonEP 式
本地复制降低 rank skew 却不能改变 server aggregate lower bound；ProbeEP 的 global
server-first plan 再降低 padded compute；最终只有被 NIC budget 接纳的 intents 进入数据
面。组内同时报告 before/after rank 与 server imbalance、padded routes、每个远程副本搬移
的有效 routes、weight RDMA、endpoint load 与 compute/network overlap。不能把所有 task
FCT 相加伪装成 critical-path breakdown。

### 6.5 Feedback and Adaptivity

展示连续同类窗口中的 `N/C`、migration budget、admitted replicas 和 moved routes。
重点不是要求每个点精确落在目标比例，而是验证：负载变化后预算方向正确、token
baseline 不被裁剪、Attention/MoE 两条状态链不串扰、预算受 endpoint hard cap 限制。

在相同 Gate 序列上加入零迁移、多个固定预算和完整 feedback。固定预算对照既报告
makespan，也报告迁移字节和暴露时间，用来说明静态保守值会错失计算收益、静态激进值会
制造网络债务，而反馈能够随窗口改变工作点。

### 6.6 Workload Sensitivity

改变 Gate family 与实测倾斜程度、层/seed、token volume、top-k 和热点持续性。报告端到端
性能、server/rank imbalance、迁移副本数、每副本 moved routes 和迁移字节。实验要找出
两个边界：接近均衡时 ProbeEP 应自动退化为少迁移；极端倾斜时 network budget 限制可
消除的 compute skew。raw receive 数据只能称为 decode-derived empirical distribution，
不能包装成精确的训练 trace。

### 6.7 Hardware Sensitivity

分别改变 compute profile、scale-out NIC、本地 scale-up fabric 和 expert state cost，再做
compute/network 交叉扫描。结果应形成清楚的适用区域：compute 更慢、热点更集中或 NIC
headroom 更大时，通信换计算更有利；专家状态过大或 token traffic 已经填满网络时，
controller 应减少 admission。

### 6.8 Topology and Scale Sensitivity

保持 workload 语义一致，改变 server boundary、server 数量、rail/plane/path diversity、
oversubscription 和 EP scale。该组实验回答两件事：local-replication ceiling 是否随 server
边界变化，以及 ProbeEP 的 endpoint-aware admission/rail-aware scheduling 是否在不同
层次拓扑下仍能避免把 compute hotspot 转成 network hotspot。无法保持 raw assignment
语义的规模点使用合成 Gate，并与 raw-trace 结论分开报告。

### 6.9 Design Ablation

至少包含：无跨服务器迁移、固定保守预算、固定激进预算、无反馈的 compute-only plan、
完整 ProbeEP。若 prototype 完成，再加入无 rail-aware chunking 和全局 Weight barrier 两个
数据面消融。

### 6.10 Overheads and Operating Boundary

分别报告 planner、controller、chunk scheduling、动态 DAG append 和 metadata 的开销。
仿真中的 Python reference planner 只用于算法验证；最终在线开销 claim 必须来自
prototype 实现。规模维度包括 EP size、expert 数、route histogram 大小和 weight chunk 数。
同时用 analytical break-even model 解释 token volume、expert state 与 exposed migration
cost 的边界，但不能用模型结果代替 packet-level end-to-end 结果。

## 7. Prototype

## 8. Discussion

### 8.1 Training Semantics

说明 forward 副本如何在 backward 复用同一 plan，副本梯度如何规约回 master，optimizer
只更新 master。当前未实现部分必须明确列出。

### 8.2 Memory Capacity

当前研究重点是网络与计算 tradeoff，未将 HBM slot 作为 admission 约束。实际系统需要
预留复用 buffer、加入容量 cap，并研究与 checkpoint/optimizer state 的关系。

### 8.3 Interaction with Communication Libraries

ProbeEP 不替代 DeepEP；它改变 physical expert execution placement，token 数据面仍可
使用 DeepEP 或其他 EP backend。FAST 与 ProbeEP 也可组合：前者优化剩余 token
All-to-AllV，后者决定是否花额外网络换计算。

### 8.4 When ProbeEP Does Not Help

主动列出均衡 Gate、网络饱和、专家状态过大、计算极快、短窗口和显存不足等场景。
正确行为应是少接纳或零接纳，而不是保证始终优于所有 baseline。

### 8.5 Generality

讨论 prefill、training forward、更多 microbatch 和其他 hierarchical topology。不要在没有
实验时扩展到 decode、任意 collective 或跨数据中心场景。

## 9. Related Work

按问题分类，不逐篇复述。

| 类别 | 写作重点 |
|---|---|
| MoE communication | NCCL、DeepEP、FAST 等优化 token movement；ProbeEP 改变 compute placement |
| Expert load balancing | EPLB、LPLB、MoonEP/UltraEP 等预测或本地复制；ProbeEP 聚焦普通 scale-out 域中的反馈式跨服务器迁移 |
| Computation-communication overlap | 既有工作隐藏给定通信；ProbeEP 反过来把可隐藏窗口作为主动迁移预算 |
| Runtime feedback control | 对比静态模型、历史预测和在线 measurement-driven control |
| AI workload simulation | 说明 DAG-aware packet simulation 的必要性和局限，不把仿真器当作系统贡献本身 |

## 10. Conclusion

结论回到一个观点：MoE 优化不应只问如何减少通信，也应问哪些通信可以安全地换掉更
昂贵的计算拖尾。ProbeEP 用全局计算规划、反馈式 admission 和分层 weight/token
pipeline，把动态 NIC headroom 转化为更均衡的 expert execution。

## 写作检查清单

- 每个 claim 都能指向一张图、一个表、一个定理或一个 prototype measurement。
- “通信换计算”始终落到 exposed communication 与 critical-path compute 的比较。
- 不把 server-local baseline 写成弱 baseline；明确它在 rack-scale 场景中的优势。
- 不把 static ProbeEP 结构测试放进主性能结果。
- 不把 task FCT 求和当作 makespan breakdown。
- 不把 raw receive 数据称为精确训练 trace；准确描述其来源和 fidelity。
- 不把仿真外 planner 时间算入端到端收益，也不宣称其在线开销已经解决。
- Prototype 没有结果前只保留一级标题，不写占位数字。
