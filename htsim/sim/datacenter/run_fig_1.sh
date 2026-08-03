#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments_output}"
OUT_DIR="${OUT_DIR:-figures}"

mkdir -p "$OUT_DIR"

python3 plot_alternative.py --output-root "$OUTPUT_ROOT" --experiment permutation_global_4MiB --sf-topo p7q9 --detail-type ooo_avg --outfile "$OUT_DIR/sf_perm.pdf" --no-show --no-y-titles

python3 plot_alternative.py --output-root "$OUTPUT_ROOT" --experiment adv_all_4MiB --sf-topo p7q9 --detail-type ooo_avg --outfile "$OUT_DIR/sf_adv.pdf" --no-show --no-y-titles

python3 plot_alternative.py --output-root "$OUTPUT_ROOT" --experiment permutation_global_4MiB --df-topo p4a8h4 --detail-type ooo_avg --outfile "$OUT_DIR/df_perm.pdf" --no-show --fig-width 3.35

python3 plot_alternative.py --output-root "$OUTPUT_ROOT" --experiment adv_i5_4MiB --df-topo p4a8h4 --detail-type ooo_avg --outfile "$OUT_DIR/df_adv.pdf" --no-show --no-y-titles
