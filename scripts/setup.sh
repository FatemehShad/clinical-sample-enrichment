#!/usr/bin/env bash
set -euo pipefail
gpu=0
cluster=0
cluster_source=0
for option in "$@"; do
  case "$option" in
    --gpu) gpu=1 ;;
    --cluster) cluster=1 ;;
    --cluster-source) cluster_source=1 ;;
    *) echo 'Usage: setup.sh [--gpu] [--cluster | --cluster-source]' >&2; exit 2 ;;
  esac
done
if (( cluster_source && (gpu || cluster) )); then
  echo '--cluster-source supports CPU only; use --gpu --cluster for CUDA wheels.' >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${CLINICAL_VENV:-$repo_dir/.venv}"
uv venv --python 3.8.20 --allow-existing "$venv_dir"
requirements_file="$repo_dir/requirements.txt"
pyg_variant=cpu
if (( gpu )); then
  requirements_file="$repo_dir/requirements-gpu.txt"
  pyg_variant=cu121
fi
uv pip sync --python "$venv_dir/bin/python" "$requirements_file"
if (( cluster )); then
  # Extension binaries must match the selected PyTorch 2.2/CUDA ABI.
  uv pip install --python "$venv_dir/bin/python" pyg-lib==0.4.0 \
    --find-links "https://data.pyg.org/whl/torch-2.2.0+${pyg_variant}.html" --only-binary :all:
fi
if (( cluster_source )); then
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
