#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PROJECT_ROOT:=$PWD}"; export PROJECT_ROOT
mkdir -p logs logs/done-grid
D=logs/done-grid
MAIN=test_n2500_c20_s42

lane() {
  local session=$1 venv=$2; shift 2
  if tmux has-session -t "$session" 2>/dev/null; then echo "$session already running"; return; fi
  local vars
  vars=$(env | grep -E '^(GRID_[A-Z_]+|SASREC_GPU|KEEP_SERVER)=' | sed 's/=\(.*\)/="\1"/' | tr '\n' ' ' || true)
  tmux new-session -d -s "$session" \
    "tcsh -c 'source $PROJECT_ROOT/$venv && env $vars bash $* |& tee -a $PROJECT_ROOT/logs/$session.log'"
  echo "started $session"
}

lane grid-data activate_project.csh scripts/grid_steps.sh data \
  "wait:results/raw/amazon-beauty/$MAIN/fullrank.json" amazon-beauty \
  "wait:results/raw/amazon-grocery/$MAIN/fullrank.json" amazon-grocery \
  wait:logs/done/model-qwen3.8-27b wait:logs/done/model-gemma4-31b ml-1m amazon-videogames
lane grid-gpu0 activate_project.csh scripts/gpu_lane.sh 0 wait:logs/done/model-qwen3.8-27b \
  model:qwen3.8-27b:Qwen/Qwen3.8-27B-FP8:8100 lora:qwen3.5-4b
lane grid-gpu1 activate_project.csh scripts/gpu_lane.sh 1 wait:logs/done/model-gemma4-31b \
  model:gemma4-31b:google/gemma-4-31B-it:8101 lora:gemma4-e4b
lane grid-gpu2 activate_project.csh scripts/gpu_lane.sh 2 \
  model:qwen3.5-27b:Qwen/Qwen3.5-27B-FP8:8102 model:qwen3.5-4b:Qwen/Qwen3.5-4B:8105
lane grid-gpu3a activate_project.csh scripts/gpu_lane.sh 3 \
  model:qwen3.5-9b:Qwen/Qwen3.5-9B:8103:0.45 model:gemma4-e4b:google/gemma-4-E4B-it:8106:0.45
lane grid-gpu3b activate_project.csh scripts/gpu_lane.sh 3 wait:$D/server-up-8103 \
  model:gemma4-12b:google/gemma-4-12B-it:8104:0.45
lane grid-eval activate_project.csh scripts/grid_steps.sh eval \
  data-amazon-beauty data-amazon-grocery data-ml-1m data-amazon-videogames \
  model-qwen3.8-27b model-gemma4-31b model-qwen3.5-27b model-qwen3.5-9b model-gemma4-12b \
  model-qwen3.5-4b model-gemma4-e4b lora-qwen3.5-4b lora-gemma4-e4b
tmux ls | grep -E '^(grid|vllm)-'
