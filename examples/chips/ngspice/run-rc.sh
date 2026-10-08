#!/usr/bin/env bash
# Operator shortcut; installed under <private-root>/harness alongside chips.pyz.
set -euo pipefail
umask 077
if [ "$#" -lt 1 ] || [[ ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "usage: bash run-rc.sh run-name [ngspice-rc options, e.g. --resume]" >&2
  exit 2
fi
run_name=$1
shift
root=$(cd "$(dirname "$0")/.." && pwd)
python=$(command -v python3.12)
exec env -i PATH=/usr/bin:/bin LC_ALL=C "$python" "$root/harness/chips.pyz" ngspice-rc \
  --input "$root/harness/rc.json" --ngspice "$root/envs/ngspice/47/bin/ngspice" \
  --output "$root/runs/$run_name" --timeout 60 "$@"
