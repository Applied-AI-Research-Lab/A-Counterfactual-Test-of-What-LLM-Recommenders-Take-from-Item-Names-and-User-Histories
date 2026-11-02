#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PROJECT_ROOT="$PWD"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
PY=.venv/bin/python
N_USERS="${N_USERS:-50}"

$PY -m recmem.prepare --dataset ml-1m
$PY -m recmem.split --dataset ml-1m --n-users "$N_USERS"
$PY -m recmem.conditions --dataset ml-1m
$PY -m recmem.run --dataset ml-1m --tag preflight --backend mock --dump-prompts 1

rm -rf results/raw/ml-1m logs
$PY -m recmem.run --dataset ml-1m --tag mock --backend mock --condition all --workers 4
$PY -m recmem.baselines --dataset ml-1m

if [ -n "${OLLAMA_MODEL:-}" ]; then
  $PY -m recmem.run --dataset ml-1m --tag "ollama-${OLLAMA_MODEL//[:\/]/-}" --model "$OLLAMA_MODEL" \
      --base-url http://localhost:11434/v1 --condition all --orders 1 --limit 20 --workers 2
fi

$PY -m recmem.evaluate --dataset ml-1m
