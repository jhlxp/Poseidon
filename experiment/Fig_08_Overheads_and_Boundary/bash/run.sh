#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"

command=(
    taskset -c 100 python3 "${TYPE_DIR}/benchmark.py"
    --mode "${MODE}" --data-dir "${TYPE_DIR}/data"
)
echo "[reference planner and analytical boundary]"
print_command "${command[@]}"
if [[ "${PLAN_ONLY}" == "1" ]]; then
    echo "Plan only: benchmark and plotting were not executed."
    exit 0
fi

log="${TYPE_DIR}/artifact/command_logs/benchmark.log"
"${command[@]}" >"${log}" 2>&1
python3 "${TYPE_DIR}/plot.py"
(
    cd "${TYPE_DIR}/artifact/command_logs"
    zip -q "${TYPE_DIR}/artifact/logs.zip" ./*.log
)
rm -rf "${TYPE_DIR}/artifact/command_logs"
echo "${TYPE_DIR}"
