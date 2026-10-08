#!/usr/bin/env bash
# Download original tasks into a private external cache; nothing is vendored.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PRIVATE_CACHE_DIRECTORY" >&2
  exit 2
fi
cache="$1"
mkdir -p "$cache"
chmod 700 "$cache"
for revision in \
  fb0ec30463d005d3e463caf4e48ab9a26008e869 \
  c23f124de1e461655d2e02ce6cfae2654ccea0d3; do
  target="$cache/analog-design-bench-$revision"
  if [[ ! -e "$target" ]]; then
    archive="$(mktemp "$cache/source.XXXXXX")"
    trap 'rm -f "$archive"' EXIT
    curl --fail --location --retry 2 \
      "https://codeload.github.com/Arcadia-1/analog-design-bench/tar.gz/$revision" \
      --output "$archive"
    tar -xzf "$archive" -C "$cache"
    rm -f "$archive"
    trap - EXIT
  fi
  printf '%s\n' "$target"
done
# prepare checks the complete selected task tree against Harness's existing pin.
