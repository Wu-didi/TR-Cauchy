#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${SCRIPT_DIR}"
mkdir -p ./save ./trainlogs

# Preserve the source snapshot's defaults; override with environment variables or CLI.
exec "${PYTHON:-python3}" trainer.py \
    --method "${METHOD:-famo}" \
    --seed "${SEED:-42}" \
    --gamma "${GAMMA:-0.0}" \
    "$@"
