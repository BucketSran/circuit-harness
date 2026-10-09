#!/usr/bin/env bash
# Compatibility entry; the canonical downloader lives beside the AnalogBench guide.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../../../analogbench/fetch-sources.sh" "$@"
