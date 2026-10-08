#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  ""|--cluster|--cluster-source) ;;
  *) echo 'Usage: setup.sh [--cluster|--cluster-source]' >&2; exit 2 ;;
esac
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${CLINICAL_VENV:-$repo_dir/.venv}"
uv venv --python 3.8.20 --allow-existing "$venv_dir"
uv pip sync --python "$venv_dir/bin/python" "$repo_dir/requirements.txt"
if [[ "${1:-}" == "--cluster" ]]; then
  # CPU-only binaries match the pinned PyTorch 2.2 ABI.
  uv pip install --python "$venv_dir/bin/python" pyg-lib==0.4.0 \
    --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html --only-binary :all:
fi
if [[ "${1:-}" == "--cluster-source" ]]; then
  source_dir="${CLINICAL_PYG_SOURCE:-${TMPDIR:-/tmp}/clinical-pyg-lib-0.4.0}"
  source_commit=84d48b5553a10d787c730467d4dc4a35bdc380c5
  if [[ ! -e "$source_dir" ]]; then
    git clone --branch 0.4.0 --depth 1 https://github.com/pyg-team/pyg-lib.git "$source_dir"
  fi
  [[ "$(git -C "$source_dir" rev-parse HEAD)" == "$source_commit" ]] || { echo 'Unexpected pyg-lib revision; use a new source directory.' >&2; exit 1; }
  git -C "$source_dir" diff --quiet || { echo 'Preserve the modified pyg-lib source and use a new directory.' >&2; exit 1; }
  git -C "$source_dir" submodule update --init --recursive third_party/METIS third_party/parallel-hashmap
  toolchain_file="$source_dir/python-toolchain.cmake"
  "$venv_dir/bin/python" - "$toolchain_file" <<'PY'
import pathlib, sys, sysconfig
include = sysconfig.get_path('include')
library = str(pathlib.Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY'))
pathlib.Path(sys.argv[1]).write_text(
    f'set(Python3_INCLUDE_DIR "{include}" CACHE PATH "" FORCE)\n'
    f'set(Python3_LIBRARY "{library}" CACHE FILEPATH "" FORCE)\n')
PY
  CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_POLICY_VERSION_MINIMUM=3.5 CMAKE_TOOLCHAIN_FILE="$toolchain_file" \
    uv pip install --python "$venv_dir/bin/python" --no-build-isolation --no-deps "$source_dir"
fi
