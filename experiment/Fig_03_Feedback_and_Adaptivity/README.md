# Fig_03 Feedback and Adaptivity

这一类型在相同 Gate 序列上比较零迁移、固定保守/激进预算和完整 feedback，并用更多连续
layer 展开 controller 时序。它同时观察 `N/C`、budget、admitted/deferred intents、moved
routes、迁移字节和端到端 makespan。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```

输出 `fig_03a_ratio`、`fig_03b_budget`、`fig_03c_admission` 和
`fig_03d_fixed_vs_feedback`。Attention 与 MoE 状态按 `compute_kind` 分开，不能把两条反馈
链拼成一条伪时间序列。
