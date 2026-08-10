#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id sweep hardware nic_gbps local_gbps weight_scale algorithm source_run
ensure_simulator

run_pair() {
    local case_id="$1"
    local sweep="$2"
    local hardware="$3"
    local nic="$4"
    local local_bw="$5"
    local weight_scale="$6"
    local profile="${REPO_ROOT}/pysrc/compute_profiles/${hardware}_DSV3_EP32_compute_4096tpr.json"
    local full_args=()
    if [[ "${MODE}" == "full" ]]; then
        full_args=(--full --compute-config "${profile}")
    fi

    queue_case "${case_id}_moonep" \
        "${case_id}" "${sweep}" "${hardware}" "${nic}" "${local_bw}" \
        "${weight_scale}" moonep -- \
        python3 "${REPO_ROOT}/tests/run_dsv3_2layer_algorithms.py" \
        --skip-build "${full_args[@]}" --workers 1 --algorithms moonep \
        --num-layers 2 --gate-provider raw_receive_cdf --gate-layer-map 0,1 \
        --gate-seed 17 --moonep-replicas-per-rank 256 \
        --nic-line-rate-gbps "${nic}" --local-line-rate-gbps "${local_bw}"

    queue_case "${case_id}_probeep" \
        "${case_id}" "${sweep}" "${hardware}" "${nic}" "${local_bw}" \
        "${weight_scale}" probeep -- \
        python3 "${REPO_ROOT}/tests/run_probeep_2layer_ratio_full.py" \
        --skip-build "${full_args[@]}" --num-layers 2 \
        --gate-provider raw_receive_cdf --gate-layer-map 0,1 --gate-seed 17 \
        --controller-mode feedback --nic-line-rate-gbps "${nic}" \
        --local-line-rate-gbps "${local_bw}" --expert-weight-scale "${weight_scale}"
}

for hardware in H20 H100; do
    for nic in 100 200 400 800; do
        run_pair "nic_${hardware}_${nic}" nic "${hardware}" "${nic}" 7200 1
    done
done
for scale in 0.25 0.5 1 2 4; do
    run_pair "weight_${scale}" weight H20 400 7200 "${scale}"
done
for local_bw in 900 1800 3600 7200; do
    run_pair "local_${local_bw}" local H20 400 "${local_bw}" 1
done

wait_for_cases
if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
collect_standard
finish_type
