#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
DONE=logs/done-grid; mkdir -p "$DONE"
N_MAIN="${GRID_USERS:-2500}"
N_ROB="${GRID_ROBUST_USERS:-500}"
P2_USERS="${GRID_P2_USERS:-500}"
SASREC_GPU="${SASREC_GPU:-2}"

main_set() { echo "data/processed/$1/test_n${N_MAIN}_c20_s42.jsonl"; }
c100_set() { echo "data/processed/$1/test_n${N_ROB}_c100_s42.jsonl"; }
a0_set()   { echo "data/processed/$1/test_n${N_ROB}_c20_s42_a0.jsonl"; }

step() {
  local marker="$DONE/$1"; shift
  if [ -e "$marker" ]; then echo "skip $(basename "$marker") (done)"; return 0; fi
  echo "== $(date +%H:%M:%S) $(basename "$marker"): $*"
  "$@"
  touch "$marker"
}

wait_for() {
  for f in "$@"; do until [ -e "$f" ]; do echo "waiting for $f ..."; sleep 60; done; done
}

wait_for_server() {
  for _ in $(seq 1 90); do
    curl -sf "http://localhost:$1/v1/models" 2>/dev/null | grep -qF "\"$2\"" && return 0
    echo "waiting for $2 on port $1 ..."; sleep 20
  done
  echo "$2 not served on port $1 after 30 min; check logs/vllm-*.log"; exit 1
}

case "$1" in
  data)
    shift
    for ds in "$@"; do
      if [[ $ds == wait:* ]]; then wait_for "${ds#wait:}"; continue; fi
      [ -e "data/processed/$ds/views_s42.parquet" ] || { echo "$ds not prepared (recmem.prepare + conditions)"; exit 1; }
      [ -e "$(main_set "$ds")" ] || step "split-main-$ds" python -m recmem.split --dataset "$ds" --n-users "$N_MAIN"
      step "split-c100-$ds" python -m recmem.split --dataset "$ds" --n-users "$N_ROB" --n-candidates 100
      step "split-a0-$ds" python -m recmem.split --dataset "$ds" --n-users "$N_ROB" --alpha 0
      touch "$DONE/sets-$ds"
      for set in "$(main_set "$ds")" "$(c100_set "$ds")" "$(a0_set "$ds")"; do
        name=$(basename "$set" .jsonl)
        [ -e "results/raw/$ds/$name/ease/idonly.jsonl" ] ||
          step "ease-$ds-$name" python -m recmem.baselines --dataset "$ds" --eval-set "$set"
        step "sasrec-$ds-$name" env CUDA_VISIBLE_DEVICES="$SASREC_GPU" python -m recmem.sasrec --dataset "$ds" --eval-set "$set"
      done
      touch "$DONE/data-$ds"
    done
    ;;
  model)
    tag=$2 model=$3 port=$4; shift 4
    wait_for_server "$port" "$model"
    touch "$DONE/server-up-$port"
    srv=(--model "$model" --base-url "http://localhost:$port/v1" --workers "${GRID_WORKERS:-64}")
    for ds in "$@"; do
      wait_for "$DONE/sets-$ds"
      step "run-$tag-$ds-main" python -m recmem.run --dataset "$ds" --tag "$tag" --eval-set "$(main_set "$ds")" "${srv[@]}"
      step "probes-$tag-$ds" python -m recmem.probes --dataset "$ds" --tag "$tag" --eval-set "$(main_set "$ds")" "${srv[@]}"
    done
    for ds in "$@"; do
      step "run-$tag-$ds-c100" python -m recmem.run --dataset "$ds" --tag "$tag" --eval-set "$(c100_set "$ds")" "${srv[@]}"
      step "run-$tag-$ds-a0" python -m recmem.run --dataset "$ds" --tag "$tag" --eval-set "$(a0_set "$ds")" "${srv[@]}"
      step "run-$tag-$ds-p2" python -m recmem.run --dataset "$ds" --tag "$tag" --eval-set "$(main_set "$ds")" \
        --prompt-variant 2 --limit "$P2_USERS" "${srv[@]}"
    done
    touch "$DONE/model-$tag"
    ;;
  lora)
    model=$2; shift 2
    for ds in "$@"; do
      wait_for "$DONE/sets-$ds"
      set=$(main_set "$ds")
      step "lora-$model-$ds-base" python -m recmem.lora --dataset "$ds" --model "$model" --base --eval-set "$set"
      for cond in real masked; do
        step "lora-$model-$ds-$cond" python -m recmem.lora --dataset "$ds" --model "$model" --train-condition "$cond" --eval-set "$set"
      done
    done
    touch "$DONE/lora-$model"
    ;;
  eval)
    shift
    wait_for $(for t in "$@"; do echo "$DONE/$t"; done)
    for ds in $(ls data/processed); do
      for set in "$(main_set "$ds")" "$(c100_set "$ds")" "$(a0_set "$ds")"; do
        [ -e "$set" ] && [ -d "results/raw/$ds/$(basename "$set" .jsonl)" ] || continue
        python -m recmem.evaluate --dataset "$ds" --eval-set "$set"
      done
    done
    echo "GRID FINISHED. Summaries in results/summary/"
    ;;
  *) echo "usage: $0 data DS... | model TAG MODEL_ID PORT DS... | lora MODEL DS... | eval TAG..."; exit 1 ;;
esac
