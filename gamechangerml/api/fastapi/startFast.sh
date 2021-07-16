#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

readonly SCRIPT_PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
readonly REPO_DIR="$( cd "$SCRIPT_PARENT_DIR/../../../"  >/dev/null 2>&1 && pwd )"
readonly MLAPP_VENV_DIR="${MLAPP_VENV_DIR:-/opt/gc-venv-current}"
readonly DS_SETUP_PATH="${REPO_DIR}/gamechangerml/setup_env.sh"

ENV_TYPE="${1:-${ENV_TYPE:-}}"
DOWNLOAD_DEP="${2:-${DOWNLOAD_DEP:-}}"

[[ -z "${ENV_TYPE}" ]] && {
  >&2 echo "[WARNING] No ENV_TYPE - 1st arg - specified, setting to 'PROD' ..."
  ENV_TYPE="PROD"
}

[[ -z "${DOWNLOAD_DEP}" ]] && {
  >&2 echo "[WARNING] No DOWNLOAD_DEP - 2nd arg - specified, setting to 'true' ..."
  DOWNLOAD_DEP="true"
}

case "$DOWNLOAD_DEP" in
  true|false)
    export DOWNLOAD_DEP
    ;;
  *)
    >&2 echo "[ERROR] Invalid DOWNLOAD_DEP specified: '$DOWNLOAD_DEP'"
    exit 1
    ;;
esac

function download_dependencies() {
    [[ "${DOWNLOAD_DEP}" == "true" ]] && {
      echo "[INFO] Attempting to download models from S3 ..."
      echo "[INFO] GC_ML_API_MODEL_NAME=${GC_ML_API_MODEL_NAME:-[DEFAULT]}"
      echo "[INFO] Attempting to download transformer cache and sentence index from S3 ..."
      source "${REPO_DIR}/gamechangerml/scripts/download_dependencies.sh"
    } || {
      echo "[INFO] Skipping model download"
    }
}

function activate_venv() {
  set +o xtrace
  echo "[INFO] Activating venv"
  source ${MLAPP_VENV_DIR}/bin/activate

  # if gamechangerml wasn't installed as module in the venv, just alter pythonpath
  if ! (pip freeze | grep -q gamechangerml); then
    >&2 echo "[WARNING] gamechangerml package not found, setting PYTHONPATH to repo root"
    export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${REPO_DIR}"
  fi
  set -o xtrace
}

function start_gunicorn() {
  echo "[INFO] Starting gunicorn workers for the API ..."
  gunicorn "$@"
}

function start_env_prod() {
  source "${DS_SETUP_PATH}" "${ENV_TYPE}"
  activate_venv
  [[ "${DOWNLOAD_DEP}" = "true" ]] && download_dependencies
  start_gunicorn gamechangerml.api.fastapi.mlapp:app \
    --bind 0.0.0.0:5000 \
    --workers 1 \
    --graceful-timeout 900 \
    --timeout 1200 \
    -k uvicorn.workers.UvicornWorker \
    --log-level debug
}

function start_env_dev() {
  source "${DS_SETUP_PATH}" "${ENV_TYPE}"
  activate_venv
  download_dependencies
  start_gunicorn gamechangerml.api.fastapi.mlapp:app \
      --bind 0.0.0.0:5000 \
      --workers 1 \
      --graceful-timeout 1000 \
      --timeout 1200 \
      --keep-alive 30 \
      --reload \
      -k uvicorn.workers.UvicornWorker \
      --log-level debug
}

function start_env_devlocal() {
  source "${DS_SETUP_PATH}" "${ENV_TYPE}"
  activate_venv
  download_dependencies
  start_gunicorn gamechangerml.api.fastapi.mlapp:app \
      --bind 0.0.0.0:5000 \
      --workers 1 \
      --graceful-timeout 900 \
      --timeout 1600 \
      -k uvicorn.workers.UvicornWorker \
      --log-level debug \
      --reload
}

function start_env_k8s_dev() {
  source "${DS_SETUP_PATH}" "${ENV_TYPE}"
  activate_venv
  download_dependencies
  start_gunicorn gamechangerml.api.fastapi.mlapp:app \
      --bind 0.0.0.0:5000 \
      --workers 1 \
      --graceful-timeout 1000 \
      --timeout 1200 \
      --keep-alive 30 \
      --reload \
      -k uvicorn.workers.UvicornWorker \
      --log-level debug
}

function start_env_k8s_test() {
  source "${DS_SETUP_PATH}" "${ENV_TYPE}"
  activate_venv
  download_dependencies
  start_gunicorn gamechangerml.api.fastapi.mlapp:app \
      --bind 0.0.0.0:5000 \
      --workers 1 \
      --graceful-timeout 900 \
      --timeout 1200 \
      -k uvicorn.workers.UvicornWorker \
      --log-level debug
}

function start_env_k8s_prod() {
  source "${DS_SETUP_PATH}" "${ENV_TYPE}"
  activate_venv
  download_dependencies
  start_gunicorn gamechangerml.api.fastapi.mlapp:app \
      --bind 0.0.0.0:5000 \
      --workers 1 \
      --graceful-timeout 900 \
      --timeout 1200 \
      -k uvicorn.workers.UvicornWorker \
      --log-level debug
}


case "${ENV_TYPE}" in
  PROD)
    start_env_prod
    ;;
  DEV)
    start_env_dev
    ;;
  DEVLOCAL)
    start_env_devlocal
    ;;
  K8S_DEV)
    start_env_k8s_dev
    ;;
  K8S_TEST)
    start_env_k8s_test
    ;;
  K8S_PROD)
    start_env_k8s_prod
    ;;
  *)
    >&2 echo "[ERROR] Attempted to start with invalid ENV_TYPE specified: '$ENV_TYPE'"
    exit 1
    ;;
esac
