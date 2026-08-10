# Fig_02 Mechanism Analysis

这一类型复用 Fig_01 Base Case，不新增 workload。它沿因果链分析 Gate skew、MoonEP 的
server-local ceiling、ProbeEP 的 planned/admitted padded-compute reduction，以及为此支付的
remote Weight RDMA 和 overlap 代价。

`collect.py` 直接读取 Base Case visualization ZIP 中的 Gate profile、timeline 和 link-load
结构化摘要，不解析 HTML，不运行 HTSim。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```

输出包含 `fig_02a_gate_skew`、`fig_02b_local_ceiling`、`fig_02c_padding`、
`fig_02d_network_cost` 和 `fig_02e_migration_efficiency`。
