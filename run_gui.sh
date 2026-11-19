#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
REQUIREMENTS="${ROOT_DIR}/requirements.txt"
INSTALL_MARKER="${VENV_DIR}/.requirements-installed"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Ошибка: не найден ${PYTHON}." >&2
    exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Создание виртуального окружения: ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

if [[ ! -f "${INSTALL_MARKER}" || "${REQUIREMENTS}" -nt "${INSTALL_MARKER}" ]]; then
    echo "Установка зависимостей..."
    "${VENV_PYTHON}" -m pip install --upgrade pip
    "${VENV_PYTHON}" -m pip install -r "${REQUIREMENTS}"
    touch "${INSTALL_MARKER}"
fi

echo "Запуск GUI: http://localhost:8501"
exec "${VENV_PYTHON}" -m streamlit run "${ROOT_DIR}/app.py" "$@"
