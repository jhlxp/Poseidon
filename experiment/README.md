# ProbeEP 实验

这里的 `Fig_XX` 表示一类 evaluation experiment，不表示一张单独的图。每个类型从多个
角度回答同一个系统问题，并由一个 `plot.py` 生成 `fig_xxa`、`fig_xxb`、`fig_xxc` 等
子结果。完整实验思想见 `experiment/实验大纲.md`。

## Base Case

所有实验默认从 `markdown/14_专用测试拓扑-EP32-1Plane.md` 定义的完整 EP32/1Plane
prefill/训练 forward 配置出发，Gate 语义遵守
`markdown/17_Gate分布与Routing生成.md`。已完成的 H20/H100 五算法 Base Case 位于：

```text
test_logs/run_20260809_234100_h20_h100_2layer_5algo_full
```

每个敏感性实验只改变自己声明的一组因素。其余配置必须继承 Base Case，并在
`data/metadata.json` 中记录差异。

## 目录约定

```text
experiment/
├── README.md
├── 实验大纲.md
├── common/
│   ├── collect_results.py
│   ├── package_artifacts.py
│   ├── plotting.py
│   └── run_helpers.sh
└── Fig_XX_Experiment_Type/
    ├── README.md          # 问题、控制变量、组内证据、解释边界
    ├── bash/run.sh        # 该实验类型的完整 case matrix
    ├── collect.py         # 类型专用数据提取；可选
    └── plot.py            # 一次生成该类型的全部子结果
```

运行入口统一为：

```bash
MODE=full bash experiment/Fig_XX_Experiment_Type/bash/run.sh
```

设置 `PLAN_ONLY=1` 只打印 case matrix，不启动仿真。`quick` 只用于检查执行链，不能进入
论文；`full` 才能标记为 paper-eligible。本轮只构造脚本，没有执行这些入口。

## CPU 资源约束

实验服务器有 128 个 CPU cores。HTSim 是单核进程，每个 case 固定占用 1 个 core；所有
实验 bash 默认最多同时运行 100 个 HTSim 进程。执行器把仿真固定到 CPU 0--99，并通过
跨脚本 slot lock 保证多个实验类型同时启动时也不突破全局 100-core 上限。CPU 100--127
保留给 HTSim 构建、数据采集、压缩、绘图和系统服务。

```bash
MAX_HTSIM_PROCESSES=100 MODE=full bash experiment/Fig_XX_Experiment_Type/bash/run.sh
```

`MAX_HTSIM_PROCESSES` 可以调小，但不能超过 100。各 runner 使用 `--workers 1` 和
`--skip-build`，公共入口在提交并发 case 前只构建一次 simulator。OpenMP/BLAS 线程数也
被限制为 1，避免单个 case 暗中占用多个 core。

## 输出约定

每次真实运行后生成：

```text
data/       # 绘图所需 CSV/JSON，长期保留
png/        # 600 dpi PNG
pdf/        # vector PDF
artifact/   # source_runs.csv、logs.zip、html.zip
```

`data/` 不混放 HTML 或日志。命令日志和源运行日志进入 `logs.zip`；交互式页面进入
`html.zip`。`source_runs.csv` 保存每个 case 对应的原始 `test_logs/run_*`，保证结果可追溯。

## 绘图规范

- `figsize=(5, 4)`，不设置 title；
- PNG 600 dpi，同时输出 PDF；
- `bbox_inches="tight"`，`pad_inches=0.02`；
- 字体优先 Arial，缺失时回退 Times New Roman；
- label 22 pt，tick 19 pt，legend 20 pt；
- legend 放不进主图时单独输出 `legend.png` 和 `legend.pdf`；
- 画图脚本只能消费 `data/`，禁止内置论文数字或生成随机趋势。

## 证据边界

- `simulation`：HTSim/DAG 的 packet-level 执行结果；
- `trace_analysis`：对 Gate 或既有仿真 artifact 的统计；
- `analytical_model`：解释 break-even 区域，不代替端到端结果；
- `reference_planner`：Python reference implementation 的结构与开销，不代表 prototype。

当前端到端模型是 forward path。backward、梯度归并、HBM 容量和真实 kernel contention
必须由 Prototype 补证，不能从仿真结果外推。
