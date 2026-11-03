#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# PyRadiomics is installed from the AIM-Harvard repository.
python -m pip install "git+https://github.com/AIM-Harvard/pyradiomics"

echo "Ambiente pronto. Ative com:"
echo "source .venv/bin/activate"
