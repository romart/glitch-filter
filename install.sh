#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' 'Usage: ./install.sh' \
        'Requires Python >=3.10 with venv support and FFmpeg on PATH.' \
        'Creates/reuses .venv and installs the project, downloading Python dependencies as needed.' \
        'Set PHOTO_FILTER_PYTHON to select a Python executable.'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf '%s\n' 'Unexpected arguments. Use ./install.sh --help.' >&2
    exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PHOTO_FILTER_PYTHON:-python3}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    exit 1
fi
"$python_bin" -c 'import sys; sys.exit("Python 3.10 or later is required") if sys.version_info < (3, 10) else None'
if ! command -v ffmpeg >/dev/null 2>&1; then
    printf '%s\n' 'FFmpeg is required. Install it through your operating system package manager.' \
        'Debian/Ubuntu: sudo apt install ffmpeg' >&2
    exit 1
fi

venv_dir="$project_root/.venv"
if [[ ! -e "$venv_dir" ]]; then
    if ! "$python_bin" -m venv "$venv_dir"; then
        printf '%s\n' 'Could not create .venv. Ensure Python venv support is installed.' \
            'Debian/Ubuntu: sudo apt install python3-venv' >&2
        exit 1
    fi
fi
if [[ ! -x "$venv_dir/bin/python" ]]; then
    printf '%s\n' 'Existing .venv has no usable Python. Repair or relocate it before retrying.' >&2
    exit 1
fi
"$venv_dir/bin/python" -c 'import sys; sys.exit("Existing .venv requires Python 3.10 or later") if sys.version_info < (3, 10) else None'
"$venv_dir/bin/python" -m pip install -e "$project_root"
"$venv_dir/bin/photo-filter" --version
printf '\nInstalled. Activate the environment:\n  source %q\nThen run:\n  photo-filter --help\n' "$venv_dir/bin/activate"
