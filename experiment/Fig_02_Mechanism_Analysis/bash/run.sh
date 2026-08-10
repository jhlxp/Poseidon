#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id hardware algorithm source_run

BASE_RUN="${BASE_RUN:-${REPO_ROOT}/test_logs/run_20260809_234100_h20_h100_2layer_5algo_full}"
if [[ ! -f "${BASE_RUN}/summary.json" ]]; then
    echo "invalid BASE_RUN: ${BASE_RUN}" >&2
    exit 2
fi
echo "[base] analyze ${BASE_RUN}"
manifest_add base "" "" "${BASE_RUN}"

if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
python3 "${TYPE_DIR}/collect.py" \
    --manifest "${MANIFEST}" --data-dir "${TYPE_DIR}/data" --mode "${MODE}"
finish_type
