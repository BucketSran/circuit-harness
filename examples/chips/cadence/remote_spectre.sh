#!/usr/bin/env bash
set -euo pipefail
run_dir=${1:?remote run directory required}
cadence_setup=${2:?Cadence csh setup file required}
# Deliberately restrict interpolation at the bash→csh boundary.
for value in "$run_dir" "$cadence_setup"; do
  [[ "$value" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo 'Use simple absolute lab paths' >&2; exit 2; }
done
cd "$run_dir"
[[ -f rc.scs && -f "$cadence_setup" ]]
cat > run.csh <<EOF
source $cadence_setup
cd $run_dir
spectre -64 rc.scs +log spectre.log -format psfascii -raw psf +lqtimeout 60
exit \$status
EOF
exec /bin/csh -f run.csh
