#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
No-Per-Rail-Release and No-Pair-Aware-Placement are not executable yet.
Add explicit ProbeEPConfig switches and preserve the same dynamic runner before
enabling these variants. Handwritten or post-processed substitutes are rejected.
EOF
exit 2
