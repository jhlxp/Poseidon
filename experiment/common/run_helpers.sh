#!/usr/bin/env bash

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
MODE="${MODE:-full}"
PLAN_ONLY="${PLAN_ONLY:-0}"

if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
    echo "MODE must be quick or full" >&2
    exit 2
fi
if [[ "${PLAN_ONLY}" != "0" && "${PLAN_ONLY}" != "1" ]]; then
    echo "PLAN_ONLY must be 0 or 1" >&2
    exit 2
fi

prepare_type() {
    TYPE_DIR="$1"
    rm -rf "${TYPE_DIR}/data" "${TYPE_DIR}/png" "${TYPE_DIR}/pdf" "${TYPE_DIR}/artifact"
    mkdir -p "${TYPE_DIR}/data" "${TYPE_DIR}/png" "${TYPE_DIR}/pdf"
    mkdir -p "${TYPE_DIR}/artifact/command_logs"
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

run_case() {
    local label="$1"
    shift
    echo "[${label}]"
    print_command "$@"
    if [[ "${PLAN_ONLY}" == "1" ]]; then
        LAST_RUN="PLAN_ONLY/${label}"
        return 0
    fi

    local marker
    marker="$(mktemp)"
    touch "${marker}"
    local log="${TYPE_DIR}/artifact/command_logs/${label}.log"
    {
        printf '$ '
        printf '%q ' "$@"
        printf '\n'
        "$@"
    } >"${log}" 2>&1 || {
        cat "${log}" >&2
        rm -f "${marker}"
        return 1
    }
    LAST_RUN="$(
        find "${REPO_ROOT}/test_logs" -maxdepth 1 -mindepth 1 -type d \
            -newer "${marker}" -printf '%T@ %p\n' \
            | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
    rm -f "${marker}"
    if [[ -z "${LAST_RUN}" || ! -d "${LAST_RUN}" ]]; then
        echo "${label}: command created no test_logs/run_* directory" >&2
        return 1
    fi
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
