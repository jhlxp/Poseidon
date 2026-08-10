#!/usr/bin/env bash
set -euo pipefail

TYPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${TYPE_DIR}/../common/run_helpers.sh"
prepare_type "${TYPE_DIR}"
manifest_init case_id sweep gpus_per_server servers planes spines links algorithm source_run

profile="${REPO_ROOT}/pysrc/compute_profiles/H20_DSV3_EP32_compute_4096tpr.json"
full_args=()
if [[ "${MODE}" == "full" ]]; then
    full_args=(--full --compute-config "${profile}")
fi

run_pair() {
    local case_id="$1"
    local sweep="$2"
    local gpus="$3"
    local planes="$4"
    local spines="$5"
    local links="$6"
    local servers=$((32 / gpus))
    local topology_args=(
        --gpus-per-server "${gpus}" --planes "${planes}"
        --spines-per-plane "${spines}" --links-per-spine "${links}"
    )

    run_case "${case_id}_moonep" \
        python3 "${REPO_ROOT}/tests/run_dsv3_2layer_algorithms.py" \
        "${full_args[@]}" --workers 1 --algorithms moonep \
        --num-layers 2 --gate-provider raw_receive_cdf --gate-layer-map 0,1 \
        --gate-seed 17 --moonep-replicas-per-rank 256 "${topology_args[@]}"
    manifest_add "${case_id}" "${sweep}" "${gpus}" "${servers}" "${planes}" \
        "${spines}" "${links}" moonep "${LAST_RUN}"

    run_case "${case_id}_probeep" \
        python3 "${REPO_ROOT}/tests/run_probeep_2layer_ratio_full.py" \
        "${full_args[@]}" --num-layers 2 \
        --gate-provider raw_receive_cdf --gate-layer-map 0,1 --gate-seed 17 \
        --controller-mode feedback "${topology_args[@]}"
    manifest_add "${case_id}" "${sweep}" "${gpus}" "${servers}" "${planes}" \
        "${spines}" "${links}" probeep "${LAST_RUN}"
}

for gpus in 4 8 16; do
    run_pair "boundary_${gpus}" boundary "${gpus}" 1 4 1
done
for spines in 1 2 4 8; do
    run_pair "spines_${spines}" paths 8 1 "${spines}" 1
done
for links in 1 2 4; do
    run_pair "bundles_${links}" paths 8 1 4 "${links}"
done
for planes in 1 2; do
    run_pair "planes_${planes}" planes 8 "${planes}" 4 1
done

if [[ "${PLAN_ONLY}" == "1" ]]; then
    finish_type
    exit 0
fi
collect_standard
python3 "${TYPE_DIR}/collect.py" --manifest "${MANIFEST}" --data-dir "${TYPE_DIR}/data"
finish_type
