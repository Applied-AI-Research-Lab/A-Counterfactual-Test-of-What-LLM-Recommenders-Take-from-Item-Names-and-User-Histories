#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
DATASETS="${PILOT_DATASETS:-ml-1m amazon-videogames}"
N_USERS="${PILOT_USERS:-500}"
BACKEND="${PILOT_BACKEND:-openai}"
DONE=logs/done; mkdir -p "$DONE"

wait_for() {
  for f in "$@"; do
    until [ -e "$f" ]; do echo "waiting for $f ..."; sleep 60; done
  done
}

wait_for_server() {
  for _ in $(seq 1 90); do
    curl -sf "http://localhost:$1/v1/models" 2>/dev/null | grep -qF "\"$2\"" && return 0
    echo "waiting for $2 on port $1 ..."; sleep 20
  done
  echo "$2 not served on port $1 after 30 min; check logs/vllm-*.log and that nothing else uses the port"; exit 1
}

case "$1" in
  data)
    if [ -e "$DONE/data" ]; then
      echo "data already prepared (delete $DONE/data to rebuild)"; exit 0
    fi
    for ds in $DATASETS; do
      python -m recmem.prepare --dataset "$ds"
      python -m recmem.split --dataset "$ds" --n-users "$N_USERS"
      python -m recmem.conditions --dataset "$ds"
      python -m recmem.baselines --dataset "$ds"
    done
    touch "$DONE/data"
    ;;
  model)
    tag=$2 model=$3 port=$4
    wait_for "$DONE/data"
    if [ "$BACKEND" = mock ]; then extra=(--backend mock); else wait_for_server "$port" "$model"; extra=(--model "$model" --base-url "http://localhost:$port/v1"); fi
    for ds in $DATASETS; do
      python -m recmem.run --dataset "$ds" --tag "$tag" --condition all --workers "${PILOT_WORKERS:-64}" "${extra[@]}"
      python -m recmem.probes --dataset "$ds" --tag "$tag" "${extra[@]}"
    done
    touch "$DONE/model-$tag"
    ;;
  eval)
    shift
    wait_for $(for t in "$@"; do echo "$DONE/model-$t"; done)
    for ds in $DATASETS; do
      python -m recmem.evaluate --dataset "$ds"
      echo; echo "== $ds: item probes (reid = names the item from its description alone; know = knows the release year)"
      probes=$(ls -t results/summary/"$ds"_test_n*_c20_s*_probes.csv 2>/dev/null | head -1)
      if [ -n "$probes" ]; then column -t -s, "$probes" 2>/dev/null || cat "$probes"; fi
    done
    echo "PILOT FINISHED. Summaries in results/summary/"
    ;;
  *) echo "usage: $0 data | model TAG MODEL_ID PORT | eval TAG..."; exit 1 ;;
esac
