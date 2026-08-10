# Fig_04 Workload Sensitivity

这一类型固定 Base Case hardware/topology，改变 Gate family、实测 skew、seed/layer 与 token
volume。每个点成对运行 MoonEP 和动态 ProbeEP；关键点可扩展为五算法。

横轴优先使用从实际 assignment 计算出的 rank/server max-to-mean，而不是 `ultra`、`fast`、
`raw` 等 provider 名称。raw receive 的 fidelity 保持为 decode-derived empirical
distribution，不称为训练 trace。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```

输出 `fig_04a_skew_speedup`、`fig_04b_migration`、`fig_04c_token_volume` 和
`fig_04d_seed_variation`。
