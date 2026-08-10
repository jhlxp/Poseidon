#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id controller budget_mib hardware algorithm layers source_run

profile="${REPO_ROOT}/pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json"
full_args=()
if [[ "${MODE}" == "full" ]]; then
    full_args=(--full --compute-config "${profile}")
fi
layers="${LAYERS:-6}"
layer_map="$(seq -s, 0 $((layers - 1)))"

run_dynamic_case() {
    local case_id="$1"
    local controller="$2"
    local budget="$3"
    run_case "${case_id}" \
        python3 "${REPO_ROOT}/tests/run_probeep_2layer_ratio_full.py" \
        "${full_args[@]}" --num-layers "${layers}" \
        --gate-provider raw_receive_cdf --gate-layer-map "${layer_map}" \
        --gate-seed 17 --controller-mode "${controller}" \
        --initial-budget-mib "${budget}"
    manifest_add "${case_id}" "${controller}" "${budget}" H20 probeep "${layers}" "${LAST_RUN}"
}

run_dynamic_case fixed_0 fixed 0
run_dynamic_case fixed_16 fixed 16
run_dynamic_case fixed_64 fixed 64
run_dynamic_case feedback feedback 16

if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
collect_standard
finish_type
