# AE 实验复现目录规范

本目录用于放置论文/报告中每一张图对应的可复现实验材料。组织原则是：
每张图一个独立目录，目录内包含一键运行脚本、实验日志、最终图片、画图代码和该图的复现说明。
AE 审查时可以只进入目标 `Fig_xx/` 目录，运行其中的 `bash/run.sh` 并检查生成的 log 与图。

## 目录结构

```text
experiment/
├── README.md
├── Fig_01/
│   ├── bash/
│   │   └── run.sh
│   ├── log/
│   │   └── README.md
│   ├── png/
│   │   └── fig_01.png
│   ├── pdf/
│   │   └── fig_01.pdf
│   ├── plot.py
│   └── README.md
├── Fig_02/
│   ├── bash/
│   │   └── run.sh
│   ├── log/
│   ├── png/
│   ├── pdf/
│   ├── plot.py
│   └── README.md
└── Fig_xx/
    ├── bash/
    │   └── run.sh
    ├── log/
    ├── png/
    ├── pdf/
    ├── plot.py
    └── README.md
```

## 每个 Fig 目录的内容

| 文件/目录 | 作用 | AE 检查点 |
|---|---|---|
| `bash/run.sh` | 一键运行本图实验，完成仿真/统计/画图，并把 stdout/stderr 写入 `log/` | `bash bash/run.sh` 可以从干净状态重建该图 |
| `log/` | 保存本实验运行日志、关键中间统计和错误信息 | 日志文件名包含日期或 case 名，能追踪每个子实验 |
| `png/` | 保存本实验最终 PNG 结果图 | 图名与论文图号一致，例如 `fig_01.png` |
| `pdf/` | 保存本实验最终 PDF 结果图 | 图名与论文图号一致，例如 `fig_01.pdf` |
| `plot.py` | 本实验专用画图脚本，只依赖本目录或仓库内明确路径 | 重新执行后能覆盖生成 `png/` 与 `pdf/` 中的结果 |
| `README.md` | 本实验说明，解释目标、命令、输入、输出和预期结果 | AE 不读源码也能知道如何复现和验证 |

## `bash/run.sh` 约定

每个 `bash/run.sh` 建议满足以下约定：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FIG_DIR}/../.." && pwd)"

mkdir -p "${FIG_DIR}/log" "${FIG_DIR}/png" "${FIG_DIR}/pdf"

# 1. 生成或运行本图需要的实验。
# 2. 将日志写入 "${FIG_DIR}/log/"。
# 3. 调用 "${FIG_DIR}/plot.py" 生成 "${FIG_DIR}/png/" 和 "${FIG_DIR}/pdf/" 下的图。
```

建议脚本支持从仓库根目录或 `Fig_xx/` 目录内运行，所有输出路径都使用 `SCRIPT_DIR`
或 `FIG_DIR` 作为锚点，避免 AE 因当前工作目录不同而复现失败。

## 单图 `README.md` 模板

每个 `Fig_xx/README.md` 建议使用以下结构：

````markdown
# Fig_XX: <图标题>

## 目标

说明本图验证的问题、对应论文/报告中的图号，以及主要对比对象。

## 一键复现

```bash
bash bash/run.sh
```

## 输入与配置

- workload:
- topology:
- algorithm/case:
- key parameters:

## 输出

- log: `log/<日志文件>`
- png: `png/fig_xx.png`
- pdf: `pdf/fig_xx.pdf`

## 预期结果

说明 AE 应该检查的趋势、数值范围或关键结论。

## 备注

记录运行时间、机器需求、随机种子、已知限制或可选的快速/完整模式。
````

## 命名建议

- 图目录统一使用 `Fig_01`、`Fig_02`、`Fig_03` 这样的两位编号。
- 最终图统一命名为 `png/fig_01.png`、`pdf/fig_01.pdf`，子图可使用
  `png/fig_01a.png`、`pdf/fig_01a.pdf`。
- 日志建议命名为 `run.log`、`simulation.log`、`plot.log`，多 case 可使用
  `case_name.log`。
- 如果某张图有多个仿真 case，可以在 `Fig_xx/` 下增加 `data/` 或 `cases/`，
  但最终 AE 入口仍保持为 `bash/run.sh`。

## AE 审查流程

1. 进入目标图目录：`cd experiment/Fig_XX`
2. 运行一键脚本：`bash bash/run.sh`
3. 检查 `log/` 中是否有完整运行日志且无异常退出。
4. 检查 `png/` 与 `pdf/` 中是否生成对应结果图。
5. 阅读本图 `README.md`，确认结果趋势与预期结果一致。
