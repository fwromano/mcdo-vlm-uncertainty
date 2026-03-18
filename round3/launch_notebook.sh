#!/usr/bin/env bash
# One-command bootstrap + launch for round3 notebooks.
#
# Usage:
#   bash round3/launch_notebook.sh
#   bash round3/launch_notebook.sh round3/another_notebook.ipynb
#   bash round3/launch_notebook.sh round3/some_subfolder

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUND3_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${ROUND3_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
DEFAULT_ENV_NAME="$(
  awk -F: '/^name:[[:space:]]*/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "${ENV_FILE}" 2>/dev/null || true
)"
ENV_NAME="${MCDO_NOTEBOOK_ENV:-${DEFAULT_ENV_NAME:-mcdo}}"
DEFAULT_TARGET="${ROUND3_DIR}/part1_baseline_embeddings/compute_embeddings.ipynb"
TARGET_INPUT="${1:-${DEFAULT_TARGET}}"
case "${MCDO_OPEN_BROWSER:-1}" in
  0|false|FALSE|False|no|NO|No) JUPYTER_OPEN_BROWSER="False" ;;
  *) JUPYTER_OPEN_BROWSER="True" ;;
esac

resolve_path() {
  local value="$1"
  if [[ "${value}" = /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}

TARGET_PATH="$(resolve_path "${TARGET_INPUT}")"

conda_env_exists() {
  local name="$1"
  conda env list | awk '{print $1}' | grep -qx "${name}"
}

ensure_conda_env() {
  if conda_env_exists "${ENV_NAME}"; then
    return
  fi

  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Error: environment file not found: ${ENV_FILE}" >&2
    exit 1
  fi

  echo "Creating conda env '${ENV_NAME}' from ${ENV_FILE}..."
  conda env create -f "${ENV_FILE}" -n "${ENV_NAME}"
}

ensure_project_install() {
  if conda run -n "${ENV_NAME}" python -c \
    "import importlib.util, sys; import torch, open_clip, PIL, numpy; sys.exit(0 if importlib.util.find_spec('mcdo_clip') else 1)" \
    >/dev/null 2>&1; then
    return
  fi

  echo "Installing project package into '${ENV_NAME}'..."
  conda run -n "${ENV_NAME}" python -m pip install --quiet -e "${REPO_ROOT}"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not installed or not on PATH." >&2
  exit 1
fi

if [[ ! -e "${TARGET_PATH}" ]]; then
  echo "Error: target not found: ${TARGET_PATH}" >&2
  exit 1
fi

case "${TARGET_PATH}" in
  "${ROUND3_DIR}"/*) ;;
  *)
    echo "Error: target must live under ${ROUND3_DIR}" >&2
    exit 1
    ;;
esac

ensure_conda_env
ensure_project_install

export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export IPYTHONDIR="${IPYTHONDIR:-${REPO_ROOT}/.cache/ipython}"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-${REPO_ROOT}/.cache/jupyter}"
mkdir -p "${MPLCONFIGDIR}" "${IPYTHONDIR}" "${JUPYTER_CONFIG_DIR}"

if ! conda run -n "${ENV_NAME}" python -c \
  "import jupyterlab, notebook, jupyter_server, ipykernel, ipywidgets, nbclient, nbconvert, sklearn, matplotlib" \
  >/dev/null 2>&1; then
  echo "Installing notebook packages into '${ENV_NAME}'..."
  conda run -n "${ENV_NAME}" python -m pip install --quiet --upgrade \
    jupyterlab \
    notebook \
    jupyter_server \
    ipykernel \
    ipywidgets \
    nbclient \
    nbconvert \
    scikit-learn \
    matplotlib
fi

echo "Registering kernel '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${ENV_NAME}" >/dev/null

echo "Launching Jupyter Lab"
echo "  root  : ${ROUND3_DIR}"
echo "  target: ${TARGET_PATH}"
echo "  kernel: ${ENV_NAME}"
echo "  browser: ${JUPYTER_OPEN_BROWSER}"
echo "If the browser does not open, use the printed localhost URL."

conda run --no-capture-output -n "${ENV_NAME}" python -m jupyter lab "${TARGET_PATH}" \
  --ServerApp.root_dir="${ROUND3_DIR}" \
  --ServerApp.open_browser="${JUPYTER_OPEN_BROWSER}"
