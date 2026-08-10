# Fig_08 Overheads and Operating Boundary

这一类型包含两类明确分开的证据：

- `reference_planner`：Python `ProbeEPBuilder.plan()` 随 EP、expert、route histogram 和
  weight chunk 数变化的时间/内存趋势；
- `analytical_model`：compute saving 与 exposed expert migration cost 的 break-even 区域。

两者都不能替代 packet-level end-to-end 结果，也不能作为最终 prototype 在线延迟 claim。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```

输出 `fig_08a_ep_scaling`、`fig_08b_expert_scaling`、`fig_08c_route_scaling` 和
`fig_08d_break_even`。
