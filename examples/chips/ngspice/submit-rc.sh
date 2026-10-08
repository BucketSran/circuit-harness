#!/usr/bin/env bash
# Submit once; the server owns simulation, grading and evidence finalization.
set -euo pipefail
umask 077
if [[ $# -lt 1 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]]; then
  echo 'usage: submit-rc.sh JOB_ID [--input task.json] [--timeout SECONDS] [--root WORK_ROOT --archive-root ARCHIVE_ROOT]' >&2
  exit 2
fi
job_id=$1
shift
root=$(cd "$(dirname "$0")/.." && pwd)
python=$(command -v python3.12)
exec env -i PATH=/usr/bin:/bin LC_ALL=C "$python" "$root/harness/chips.pyz" submit-rc \
  --input "$root/harness/rc.json" --ngspice "$root/envs/ngspice/47/bin/ngspice" \
  --root "$root/jobs" --job-id "$job_id" --timeout 60 "$@"
