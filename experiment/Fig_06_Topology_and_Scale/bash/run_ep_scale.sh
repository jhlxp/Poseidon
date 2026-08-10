#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
EP-size execution is intentionally disabled: the current dynamic runner, raw Gate
assignment, and H20/H100 compute profiles are EP32-specific. Generalize all three
before enabling EP16/EP64 so the scale experiment remains semantically comparable.
EOF
exit 2
