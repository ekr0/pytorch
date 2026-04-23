#!/usr/bin/env bash
# Build a macOS arm64 wheel for every CPython version in DESIRED_PYTHONS on
# a single runner. After the first full build, subsequent iterations only
# recompile libtorch_python + _C for the new Python ABI (libtorch_cpu is
# ABI-free and reused) via the cross-Python cache invalidation in
# tools/setup_helpers/cmake.py.
#
# Delocation and smoke-testing are handled in separate workflow steps/jobs
# — this script only produces raw wheels under
# "${RUNNER_TEMP}/artifacts/<build_name>".
#
# Inputs (env):
#   PYTORCH_ROOT     Path to the PyTorch checkout.
#   DESIRED_PYTHONS  Space-separated versions, e.g. "3.10 3.11 3.12 3.13 3.13t".
#   RUNNER_TEMP      Work dir (defaults to /tmp).
#   BINARY_ENV_FILE  Rewritten per iteration by binary_populate_env.sh;
#                    defaults to "${RUNNER_TEMP}/env".

set -eux -o pipefail

: "${PYTORCH_ROOT:?PYTORCH_ROOT must be set}"
: "${DESIRED_PYTHONS:?DESIRED_PYTHONS must be set (space-separated list)}"
: "${RUNNER_TEMP:=/tmp}"
export BINARY_ENV_FILE="${BINARY_ENV_FILE:-${RUNNER_TEMP}/env}"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

for desired in ${DESIRED_PYTHONS}; do
    echo "===== Building PyTorch wheel for Python ${desired} ====="
    uv python install "${desired}"
    py_bin_dir="$(dirname "$(uv python find "${desired}")")"
    export PATH="${py_bin_dir}:${PATH}"

    build_name="wheel-py${desired//./_}-cpu"
    export DESIRED_PYTHON="${desired}"
    export PYTORCH_FINAL_PACKAGE_DIR="${RUNNER_TEMP}/artifacts/${build_name}"
    export MAC_PACKAGE_WORK_DIR="${RUNNER_TEMP}/work/${build_name}"
    mkdir -p "${PYTORCH_FINAL_PACKAGE_DIR}" "${MAC_PACKAGE_WORK_DIR}"

    "${PYTORCH_ROOT}/.ci/pytorch/binary_populate_env.sh"
    # shellcheck disable=SC1090
    source "${BINARY_ENV_FILE}"

    USE_PYTORCH_METAL_EXPORT=1
    USE_COREML_DELEGATE=1
    TORCH_PACKAGE_NAME="${TORCH_PACKAGE_NAME//-/_}"
    export USE_PYTORCH_METAL_EXPORT USE_COREML_DELEGATE TORCH_PACKAGE_NAME
    "${PYTORCH_ROOT}/.ci/wheel/build_wheel.sh"
done
