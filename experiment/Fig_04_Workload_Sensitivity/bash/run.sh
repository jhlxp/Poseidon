#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id gate_provider skew_label seed tokens hardware algorithm source_run

profile="${REPO_ROOT}/pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json"
full_args=()
if [[ "${MODE}" == "full" ]]; then
    full_args=(--full --compute-config "${profile}")
fi

run_pair() {
    local case_id="$1"
    local provider="$2"
    local skew_label="$3"
    local seed="$4"
    local tokens="$5"
    shift 5
    local provider_args=("$@")
    local raw_args=()
    if [[ "${provider}" == "raw_receive_cdf" ]]; then
        raw_args=(--gate-layer-map 0,1)
    fi

    run_case "${case_id}_moonep" \
        python3 "${REPO_ROOT}/tests/run_dsv3_2layer_algorithms.py" \
        "${full_args[@]}" --workers 1 --algorithms moonep --num-layers 2 \
        --gate-provider "${provider}" --gate-seed "${seed}" \
        "${raw_args[@]}" "${provider_args[@]}" \
        --tokens-per-rank "${tokens}" --chunk-tokens "${tokens}" \
        --moonep-replicas-per-rank 256
    manifest_add "${case_id}" "${provider}" "${skew_label}" "${seed}" \
        "${tokens}" H20 moonep "${LAST_RUN}"

    run_case "${case_id}_probeep" \
        python3 "${REPO_ROOT}/tests/run_probeep_2layer_ratio_full.py" \
        "${full_args[@]}" --num-layers 2 \
        --gate-provider "${provider}" --gate-seed "${seed}" \
        "${raw_args[@]}" "${provider_args[@]}" \
        --tokens-per-rank "${tokens}" --chunk-tokens "${tokens}" \
        --controller-mode feedback
    manifest_add "${case_id}" "${provider}" "${skew_label}" "${seed}" \
        "${tokens}" H20 probeep "${LAST_RUN}"
}

run_pair balanced balanced_permuted balanced 17 4096
run_pair uniform uniform_random uniform 17 4096
run_pair ultra_1_25 ultra_rank_zipf rank_1.25 17 4096 --gate-target-rank-imbalance 1.25
run_pair ultra_2 ultra_rank_zipf rank_2 17 4096 --gate-target-rank-imbalance 2
run_pair ultra_4 ultra_rank_zipf rank_4 17 4096 --gate-target-rank-imbalance 4
run_pair fast_05 fast_matrix_zipf fast_0.5 17 4096 --gate-fast-skew 0.5
run_pair fast_08 fast_matrix_zipf fast_0.8 17 4096 --gate-fast-skew 0.8
run_pair fast_095 fast_matrix_zipf fast_0.95 17 4096 --gate-fast-skew 0.95
run_pair raw raw_receive_cdf empirical 17 4096
run_pair raw_t1024 raw_receive_cdf empirical 17 1024
run_pair raw_t8192 raw_receive_cdf empirical 17 8192
run_pair ultra_seed23 ultra_rank_zipf rank_2 23 4096 --gate-target-rank-imbalance 2
run_pair ultra_seed41 ultra_rank_zipf rank_2 41 4096 --gate-target-rank-imbalance 2

if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
collect_standard
python3 "${TYPE_DIR}/collect.py" --manifest "${MANIFEST}" --data-dir "${TYPE_DIR}/data"
finish_type
