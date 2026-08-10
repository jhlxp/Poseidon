#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id hardware algorithm workload compute_profile nic_gbps source_run

BASE_RUN="${BASE_RUN:-${REPO_ROOT}/test_logs/run_20260809_234100_h20_h100_2layer_5algo_full}"
REUSE_BASE="${REUSE_BASE:-1}"

if [[ "${REUSE_BASE}" == "1" ]]; then
    if [[ ! -f "${BASE_RUN}/summary.json" ]]; then
        echo "invalid BASE_RUN: ${BASE_RUN}" >&2
        exit 2
    fi
    echo "[base] reuse ${BASE_RUN}"
    manifest_add base "" "" raw_receive_cdf H20_and_H100 400 "${BASE_RUN}"
else
    ensure_simulator
    for hardware in H20 H100; do
        profile="${REPO_ROOT}/pysrc/compute_profiles/${hardware}_DSV3_EP32_compute_4096tpr.json"
        full_args=()
        if [[ "${MODE}" == "full" ]]; then
            full_args=(--full --compute-config "${profile}")
        fi
        for algorithm in nccl deepep eplb moonep; do
            queue_case "${hardware}_${algorithm}" \
                "${hardware}_${algorithm}" "${hardware}" "${algorithm}" \
                raw_receive_cdf "${profile}" 400 -- \
                python3 "${REPO_ROOT}/tests/run_dsv3_2layer_algorithms.py" \
                --skip-build "${full_args[@]}" --workers 1 \
                --algorithms "${algorithm}" --num-layers 2 \
                --gate-provider raw_receive_cdf --gate-layer-map 0,1 \
                --gate-seed 17 --moonep-replicas-per-rank 256
        done

        queue_case "${hardware}_probeep" \
            "${hardware}_probeep" "${hardware}" probeep \
            raw_receive_cdf "${profile}" 400 -- \
            python3 "${REPO_ROOT}/tests/run_probeep_2layer_ratio_full.py" \
            --skip-build "${full_args[@]}" --num-layers 2 \
            --gate-provider raw_receive_cdf --gate-layer-map 0,1 \
            --gate-seed 17 --controller-mode feedback
    done
fi

wait_for_cases
if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
collect_standard
finish_type
