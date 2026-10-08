#!/usr/bin/env bash
#
# installation.sh -- create a virtualenv able to run the AlphaApollo v3 test
# suite and a workflow.
#
# pyproject.toml is the single source of truth for dependencies. This script
# never restates a package list; it only calls `pip install -e ".[extra]"` and
# lets pip read the extras. If a dependency changes, this file does not.
#
#   ./installation.sh                  # dev extra (api + test + ruff) into ./.venv
#   ./installation.sh --extras test    # just the test suite
#   ./installation.sh --hf-mirror      # CN hosts: huggingface.co is unreachable
#   ./installation.sh --skip-verify    # install only, no pytest
#
# No sudo. Writes nothing outside the repo and the venv it creates.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
EXTRAS="dev"
USE_HF_MIRROR=0
SKIP_VERIFY=0
PYTHON_BIN="${PYTHON_BIN:-}"

HF_MIRROR_URL="https://hf-mirror.com"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m error:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --extras)      EXTRAS="${2:?--extras needs a value}"; shift 2 ;;
    --venv)        VENV_DIR="${2:?--venv needs a value}"; shift 2 ;;
    --python)      PYTHON_BIN="${2:?--python needs a value}"; shift 2 ;;
    --hf-mirror)   USE_HF_MIRROR=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    -h|--help)     sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)             die "unknown argument: $1 (try --help)" ;;
  esac
done

# --------------------------------------------------------------------------
# 1. Preflight -- refuse early rather than half-install.
# --------------------------------------------------------------------------

[ -f "$REPO_ROOT/pyproject.toml" ] || die "pyproject.toml not found; run this from the repo root."

# Read the floor from pyproject rather than hardcoding it, so this check cannot
# drift away from requires-python.
MIN_PY="$(sed -n 's/^requires-python *= *"[><=~]*\([0-9][0-9.]*\)".*/\1/p' pyproject.toml | head -1)"
MIN_PY="${MIN_PY:-3.10}"

if [ -z "$PYTHON_BIN" ]; then
  # Prefer an explicitly-versioned interpreter; bare `python3` on these hosts can
  # be older than the floor.
  for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$(command -v "$cand")"; break; fi
  done
fi
[ -n "$PYTHON_BIN" ] || die "no python interpreter found on PATH."

# Compare as integer tuples; string compare gets 3.9 vs 3.10 wrong.
"$PYTHON_BIN" - "$MIN_PY" <<'PY' || die "python at '$PYTHON_BIN' is older than requires-python in pyproject.toml."
import sys
want = tuple(int(p) for p in sys.argv[1].split("."))
if sys.version_info[:len(want)] < want:
    sys.stderr.write("found %s, need >=%s\n" % (".".join(map(str, sys.version_info[:3])), sys.argv[1]))
    raise SystemExit(1)
PY

log "python: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')), requires-python >=$MIN_PY"

command -v git >/dev/null 2>&1 || die "git is required (the learning extra installs a package from a git URL)."

# ensurepip is missing on Debian/Ubuntu without python3-venv, and the failure
# mode without this check is a half-made venv with no pip in it.
"$PYTHON_BIN" -c 'import ensurepip, venv' 2>/dev/null \
  || die "python venv/ensurepip missing. On Debian/Ubuntu install python3-venv (needs your sysadmin; this script does not use sudo)."

# --------------------------------------------------------------------------
# 2. Network knobs -- proxy and HF mirror.
# --------------------------------------------------------------------------

# The OpenAI client rides on httpx. httpx only speaks SOCKS when socksio is
# present, and it inherits the proxy from the environment, so a shell with
# ALL_PROXY=socks5://... produces `ImportError: Using SOCKS proxy, but
# 'socksio' is not installed` at client construction -- long after install.
# Detect the inherited proxy and pull the extra in up front.
NEED_SOCKS=0
for var in ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy; do
  val="${!var:-}"
  case "$val" in socks*://*) NEED_SOCKS=1 ;; esac
done
[ "$NEED_SOCKS" -eq 1 ] && log "SOCKS proxy detected in the environment; will install httpx[socks]."

# huggingface.co is unreachable from the CN hosts. Not hardcoded: honoured if
# already exported, opt-in via --hf-mirror, otherwise left alone.
if [ -n "${HF_ENDPOINT:-}" ]; then
  log "HF_ENDPOINT already set to $HF_ENDPOINT; leaving it."
elif [ "$USE_HF_MIRROR" -eq 1 ]; then
  export HF_ENDPOINT="$HF_MIRROR_URL"
  log "HF_ENDPOINT=$HF_ENDPOINT for this run (and appended to the venv activate)."
fi

# --------------------------------------------------------------------------
# 3. Virtualenv -- idempotent: reuse a healthy one, replace a broken one.
# --------------------------------------------------------------------------

if [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c 'import sys' 2>/dev/null; then
  log "reusing existing venv at $VENV_DIR"
elif [ -e "$VENV_DIR" ]; then
  # A directory that exists but has no working interpreter is a half-made venv
  # from an interrupted run. Recreating is safe; repairing is not.
  warn "$VENV_DIR exists but has no working interpreter; recreating."
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
[ -x "$VPY" ] || die "venv creation failed: no interpreter at $VPY"

# Always call pip as `python -m pip`: the console script bakes in an absolute
# shebang and breaks if the venv is ever moved.
PIP="$VPY -m pip"

log "upgrading pip/setuptools/wheel"
$PIP install --quiet --upgrade pip setuptools wheel

# --------------------------------------------------------------------------
# 4. Install the project.
# --------------------------------------------------------------------------

log "installing -e .[$EXTRAS] (dependency list comes from pyproject.toml)"
$PIP install -e ".[$EXTRAS]"

if [ "$NEED_SOCKS" -eq 1 ]; then
  log "installing httpx[socks] for the inherited SOCKS proxy"
  $PIP install "httpx[socks]"
fi

# Persist the mirror into the venv so `source .venv/bin/activate` carries it.
# Guarded by grep so a second run does not append a duplicate line.
if [ "$USE_HF_MIRROR" -eq 1 ] && ! grep -q 'HF_ENDPOINT' "$VENV_DIR/bin/activate"; then
  printf '\n# huggingface.co is unreachable from the CN hosts; hf-mirror.com mirrors it.\nexport HF_ENDPOINT=%s\n' \
    "$HF_MIRROR_URL" >> "$VENV_DIR/bin/activate"
fi

# --------------------------------------------------------------------------
# 5. Check Hydra/OmegaConf parser compatibility.
# --------------------------------------------------------------------------
# Hydra and OmegaConf require the matching generated-parser/runtime pair.
# Check the installed version, then exercise interpolation below.

log "checking the antlr4/Hydra resolution"

ANTLR_VER="$($VPY - <<'PY'
try:
    from importlib.metadata import version
    print(version("antlr4-python3-runtime"))
except Exception:
    print("")
PY
)"
case "$ANTLR_VER" in
  4.9.*) log "antlr4-python3-runtime $ANTLR_VER (pulled in by hydra-core/omegaconf) -- correct" ;;
  "")    warn "antlr4-python3-runtime not installed; hydra-core may be missing from this extra." ;;
  *)     die "antlr4-python3-runtime is $ANTLR_VER, expected 4.9.*. Something re-pinned it
        with an incompatible runtime. Hydra config composition will fail at run time." ;;
esac

# Functional canary, not just a version string: OmegaConf resolves
# interpolations through the antlr4-generated parser, which is the exact code
# path that raises on a runtime/codegen version mismatch.
$VPY - <<'PY' || die "OmegaConf interpolation failed -- this is the antlr4 version mismatch in action."
from omegaconf import OmegaConf
cfg = OmegaConf.create({"a": 1, "b": "${a}"})
assert cfg.b == 1, cfg.b
print("omegaconf interpolation OK (antlr4 parser healthy)")
PY

# --------------------------------------------------------------------------
# 6. Verify.
# --------------------------------------------------------------------------

if [ "$SKIP_VERIFY" -eq 1 ]; then
  log "--skip-verify given; stopping before the test suite."
else
  # Verify the way a user runs it -- with the venv activated. The workspace
  # tools shell out to a bare `python`, which on hosts that ship only
  # `python3` exists solely as $VENV_DIR/bin/python. Calling pytest by
  # absolute path without this leaves the venv off PATH and 19 workspace-tool
  # tests fail with "/bin/sh: python: command not found".
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  log "importing the package and the workflow entry point"
  $VPY -c 'import alphaapollo, alphaapollo.workflows.main as m; print("alphaapollo", alphaapollo.__version__ if hasattr(alphaapollo,"__version__") else "(no __version__)", "workflow entry:", m.__name__)'

  if $VPY -c 'import pytest' 2>/dev/null; then
    log "running the test suite"
    # Not `set -e`-fatal on its own line: report the count, then exit non-zero
    # so a red suite cannot be mistaken for a good install.
    if "$VENV_DIR/bin/pytest" -q; then
      log "test suite passed"
    else
      die "test suite failed -- the install completed but this tree is not green."
    fi
  else
    warn "pytest not present in the '$EXTRAS' extra; skipping the suite. Use --extras dev or test."
  fi
fi

# --------------------------------------------------------------------------

cat <<EOF

$(log "done")

  activate:  source ${VENV_DIR#$REPO_ROOT/}/bin/activate
  tests:     pytest -q
  workflow:  python -m alphaapollo.workflows.main --config <your.yaml>

  CN hosts: huggingface.co is unreachable. Re-run with --hf-mirror, or export
  HF_ENDPOINT=$HF_MIRROR_URL before pulling any model or dataset.

  SOCKS proxy: if you later export ALL_PROXY=socks5://..., install httpx[socks]
  into this venv or the OpenAI client raises ImportError on 'socksio'.
EOF
