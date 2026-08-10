# Fig_07 Design Ablation

这一类型保持相同的动态 runner 和 Base Case Gate，逐项改变 migration admission、feedback
和 Weight chunk granularity。所有消融都走 observation/update/append 路径，避免把 static
ProbeEP 与动态 ProbeEP 混比。

当前可执行项：No-Remote、固定保守/激进预算、feedback、细粒度/单块 Weight。尚未实现的
No-Per-Rail-Release 与 No-Pair-Aware-Placement 由 `bash/run_pipeline_ablation.sh` 显式
guard；在真正的 config switch 完成前不生成伪数据。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```
