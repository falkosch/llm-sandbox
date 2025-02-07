#!/usr/bin/env bash
set -x

[ -d .venv ] && rm -rf .venv

PYTHON_TOOL="$(which -a python | grep -E "3\.?12")"
${PYTHON_TOOL} -m venv --copies --clear --upgrade-deps .venv || exit 1

. .venv/Scripts/activate || exit 1

pip install -r requirements.txt || exit 1
