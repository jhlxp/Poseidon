#!/usr/bin/env bash

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
MODE="${MODE:-full}"
PLAN_ONLY="${PLAN_ONLY:-0}"
HOST_CPU_CORES="${HOST_CPU_CORES:-128}"
HTSIM_CORES_PER_PROCESS=1
MAX_HTSIM_PROCESSES="${MAX_HTSIM_PROCESSES:-100}"
BUILD_JOBS="${BUILD_JOBS:-28}"

if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
    echo "MODE must be quick or full" >&2
    exit 2
fi
if [[ "${PLAN_ONLY}" != "0" && "${PLAN_ONLY}" != "1" ]]; then
    echo "PLAN_ONLY must be 0 or 1" >&2
    exit 2
fi
if (( HOST_CPU_CORES < 128 )); then
    echo "HOST_CPU_CORES must be at least 128 for the configured CPU partition" >&2
    exit 2
fi
if (( MAX_HTSIM_PROCESSES <= 0 || MAX_HTSIM_PROCESSES > 100 )); then
    echo "MAX_HTSIM_PROCESSES must be in [1, 100]" >&2
    exit 2
fi
if (( MAX_HTSIM_PROCESSES > HOST_CPU_CORES )); then
    echo "MAX_HTSIM_PROCESSES cannot exceed HOST_CPU_CORES" >&2
    exit 2
fi
if (( BUILD_JOBS <= 0 || BUILD_JOBS > 28 )); then
    echo "BUILD_JOBS must be in [1, 28]" >&2
    exit 2
fi

export HOST_CPU_CORES HTSIM_CORES_PER_PROCESS MAX_HTSIM_PROCESSES
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
RUN_USER="${USER:-$(id -un)}"
HTSIM_SLOT_DIR="${HTSIM_SLOT_DIR:-/tmp/probeep_htsim_slots_${RUN_USER}}"
CONTROL_CPU_RANGE="100-$((HOST_CPU_CORES - 1))"

prepare_type() {
    TYPE_DIR="$1"
    rm -rf "${TYPE_DIR}/data" "${TYPE_DIR}/png" "${TYPE_DIR}/pdf" "${TYPE_DIR}/artifact"
    mkdir -p "${TYPE_DIR}/data" "${TYPE_DIR}/png" "${TYPE_DIR}/pdf"
    mkdir -p "${TYPE_DIR}/artifact/command_logs"
    mkdir -p "${TYPE_DIR}/artifact/manifest_rows"
    mkdir -p "${TYPE_DIR}/artifact/job_status"
    MANIFEST="${TYPE_DIR}/artifact/source_runs.csv"
}

manifest_init() {
    python3 - "${MANIFEST}" "$@" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(sys.argv[2:])
PY
    MANIFEST_VALUE_COUNT=$(($# - 1))
}

manifest_add() {
    python3 - "${MANIFEST}" "$@" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(sys.argv[2:])
PY
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

ensure_simulator() {
    echo "[build] HTSim (BUILD_JOBS=${BUILD_JOBS})"
    if [[ "${PLAN_ONLY}" == "1" ]]; then
        return 0
    fi
    local log="${TYPE_DIR}/artifact/command_logs/build.log"
    (
        flock 9
        taskset -c "${CONTROL_CPU_RANGE}" cmake -S "${REPO_ROOT}/htsim/sim" \
            -B "${REPO_ROOT}/htsim/sim/build-mprail" \
            -DCMAKE_BUILD_TYPE=Release
        taskset -c "${CONTROL_CPU_RANGE}" cmake --build \
            "${REPO_ROOT}/htsim/sim/build-mprail" \
            --target htsim_uec -j "${BUILD_JOBS}"
    ) 9>"/tmp/probeep_htsim_build_${RUN_USER}.lock" >"${log}" 2>&1
}

_source_run_from_log() {
    local log="$1"
    local source_run
    source_run="$(sed -n 's/^log directory:[[:space:]]*//p' "${log}" | tail -n 1)"
    if [[ -z "${source_run}" ]]; then
        source_run="$(sed -n '\|/test_logs/run_|p' "${log}" | tail -n 1)"
    fi
    printf '%s\n' "${source_run}"
}

_execute_case() {
    local label="$1"
    shift
    local row_values=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        row_values+=("$1")
        shift
    done
    if [[ $# -eq 0 ]]; then
        echo "${label}: missing command separator" >&2
        return 2
    fi
    shift

    local log="${TYPE_DIR}/artifact/command_logs/${label}.log"
    local status="${TYPE_DIR}/artifact/job_status/${label}.status"
    local slot slot_fd
    mkdir -p "${HTSIM_SLOT_DIR}"
    while true; do
        for slot in $(seq 0 99); do
            exec {slot_fd}>"${HTSIM_SLOT_DIR}/slot_${slot}.lock"
            if flock -n "${slot_fd}"; then
                break 2
            fi
            eval "exec ${slot_fd}>&-"
        done
        sleep 0.2
    done
    if ! {
        printf '# htsim_slot=%s cpu=%s cores=1\n' "${slot}" "${slot}"
        printf '$ '
        printf 'taskset -c %q ' "${slot}"
        printf '%q ' "$@"
        printf '\n'
        taskset -c "${slot}" "$@"
    } >"${log}" 2>&1; then
        printf 'failed\n' >"${status}"
        eval "exec ${slot_fd}>&-"
        return 1
    fi
    eval "exec ${slot_fd}>&-"
    local source_run
    source_run="$(_source_run_from_log "${log}")"
    if [[ -z "${source_run}" || ! -d "${source_run}" ]]; then
        echo "${label}: command created no test_logs/run_* directory" >&2
        printf 'failed\n' >"${status}"
        return 1
    fi
    python3 - "${TYPE_DIR}/artifact/manifest_rows/${label}.csv" \
        "${row_values[@]}" "${source_run}" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open("w", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow(sys.argv[2:])
PY
    printf 'passed\n' >"${status}"
}

queue_case() {
    local label="$1"
    shift
    local row_count=0
    local value
    for value in "$@"; do
        [[ "${value}" == "--" ]] && break
        row_count=$((row_count + 1))
    done
    if (( row_count != MANIFEST_VALUE_COUNT )); then
        echo "${label}: expected ${MANIFEST_VALUE_COUNT} manifest values, got ${row_count}" >&2
        return 2
    fi
    echo "[${label}]"
    local command_start=$((row_count + 2))
    print_command "${@:${command_start}}"
    if [[ "${PLAN_ONLY}" == "1" ]]; then
        return 0
    fi
    while (( $(jobs -rp | wc -l) >= MAX_HTSIM_PROCESSES )); do
        wait -n || true
    done
    printf 'queued\n' >"${TYPE_DIR}/artifact/job_status/${label}.status"
    _execute_case "${label}" "$@" &
}

wait_for_cases() {
    if [[ "${PLAN_ONLY}" == "1" ]]; then
        echo "Resource plan: ${MAX_HTSIM_PROCESSES} HTSim processes x ${HTSIM_CORES_PER_PROCESS} core; ${HOST_CPU_CORES} host cores."
        return 0
    fi
    wait || true
    local failed=0
    local status
    for status in "${TYPE_DIR}"/artifact/job_status/*.status; do
        [[ -e "${status}" ]] || continue
        if [[ "$(<"${status}")" != "passed" ]]; then
            failed=$((failed + 1))
            local label
            label="$(basename "${status}" .status)"
            echo "${label}: failed; see artifact/command_logs/${label}.log" >&2
        fi
    done
    if (( failed > 0 )); then
        echo "${failed} experiment case(s) failed" >&2
        return 1
    fi
    local row
    for row in "${TYPE_DIR}"/artifact/manifest_rows/*.csv; do
        [[ -e "${row}" ]] || continue
        cat "${row}" >>"${MANIFEST}"
    done
    rm -rf "${TYPE_DIR}/artifact/manifest_rows" \
        "${TYPE_DIR}/artifact/job_status"
}

collect_standard() {
    python3 "${COMMON_DIR}/collect_results.py" \
        --manifest "${MANIFEST}" \
        --data-dir "${TYPE_DIR}/data" \
        --experiment-type "$(basename "${TYPE_DIR}")" \
        --mode "${MODE}"
}

finish_type() {
    if [[ "${PLAN_ONLY}" == "1" ]]; then
        echo "Plan only: no experiment, collection, plotting, or packaging was executed."
        return 0
    fi
    python3 "${TYPE_DIR}/plot.py"
    python3 "${COMMON_DIR}/package_artifacts.py" \
        --manifest "${MANIFEST}" \
        --artifact-dir "${TYPE_DIR}/artifact"
    echo "${TYPE_DIR}"
}
