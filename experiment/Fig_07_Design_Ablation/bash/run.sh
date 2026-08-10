#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id variant controller budget_mib weight_chunk_mib target_ratio hardware algorithm source_run

profile="${REPO_ROOT}/pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json"
full_args=()
if [[ "${MODE}" == "full" ]]; then
    full_args=(--full --compute-config "${profile}")
fi

run_variant() {
    local case_id="$1"
    local variant="$2"
    local controller="$3"
    local budget="$4"
    local chunk_mib="$5"
    local target="$6"
    local chunk_bytes
    chunk_bytes="$(python3 -c "print(int(float('${chunk_mib}') * 1024 * 1024))")"
    run_case "${case_id}" \
        python3 "${REPO_ROOT}/tests/run_probeep_2layer_ratio_full.py" \
        "${full_args[@]}" --num-layers 2 \
        --gate-provider raw_receive_cdf --gate-layer-map 0,1 --gate-seed 17 \
        --controller-mode "${controller}" --initial-budget-mib "${budget}" \
        --weight-chunk-bytes "${chunk_bytes}" --target-overlap-ratio "${target}"
    manifest_add "${case_id}" "${variant}" "${controller}" "${budget}" \
        "${chunk_mib}" "${target}" H20 probeep "${LAST_RUN}"
}

run_variant no_remote no_remote fixed 0 4 0.9
run_variant fixed_8 fixed_conservative fixed 8 4 0.9
run_variant fixed_64 compute_only_aggressive fixed 64 4 0.9
run_variant fine_1 fine_weight_chunks feedback 16 1 0.9
run_variant full full_probeep feedback 16 4 0.9
run_variant coarse_128 monolithic_weight feedback 16 128 0.9
run_variant target_05 conservative_target feedback 16 4 0.5

if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
collect_standard
finish_type
