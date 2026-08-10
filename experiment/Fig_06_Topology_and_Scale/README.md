# Fig_06 Topology and Scale Sensitivity

这一类型首先固定 EP32，改变 GPUs/server（即 server boundary）、plane/spine/bundle path
diversity 和 oversubscription，观察 MoonEP ceiling、ProbeEP speedup、endpoint/link peak load。

当前可执行 runner 的 rank 数固定为 32，因此 `bash/run.sh` 覆盖 EP32 内的 topology matrix。
EP16/EP64 需要同时泛化动态 runner、compute profile 和 Gate assignment；
`bash/run_ep_scale.sh` 是显式 guard，防止把不可比的静态 DAG 或重复 raw trace 混进结果。
当前 ProbeEP Weight route metadata 固定在 plane 0，因此 multi-plane case 用于暴露未利用的
额外容量，不包装成 multi-plane-aware scheduling 结果。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```
