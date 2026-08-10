# Fig_05 Hardware Sensitivity

这一类型保持 Base Case workload 与 assignment 不变，改变 GPU compute profile、scale-out
NIC、本地 fabric 和 expert state scale。先做单因素，再用 H20/H100 × NIC 构造必要的
compute/network operating region。

每个点成对运行 MoonEP 和动态 ProbeEP，输出 speedup、migration/admission 以及 exposed
network 代价。expert state scale 只改变 ProbeEP 的完整专家搬移字节，不改变 token traffic
或 baseline 计算量。

```bash
MODE=full PLAN_ONLY=1 bash bash/run.sh
MODE=full bash bash/run.sh
```
