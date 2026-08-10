# Fig_01 End-to-End Performance

这一类型回答：ProbeEP 是否在 Base Case 上优于 NCCL、DeepEP、EPLB 和 MoonEP，以及结论
是否跨 H20/H100 compute/network balance 成立。

组内证据包括绝对 makespan、相对最强 baseline 的 normalized makespan/speedup、有效 token
throughput 和 case 审计信息。默认直接消费已完成的 Base Case；设置 `REUSE_BASE=0` 才会
重新运行四个静态 baseline 和动态 ProbeEP。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```

输出 `fig_01a_makespan`、`fig_01b_normalized`、`fig_01c_speedup` 和
`fig_01d_throughput`。所有子结果来自同一类端到端实验，不拆成新的 Fig 类型。
