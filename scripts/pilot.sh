#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PROJECT_ROOT:=$PWD}"; export PROJECT_ROOT
mkdir -p logs

QWEN=Qwen/Qwen3.8-27B-FP8
GEMMA=google/gemma-4-31B-it
PORT_QWEN="${PORT_QWEN:-8100}"
PORT_GEMMA="${PORT_GEMMA:-8101}"

start() {
  local session=$1; shift
  if tmux has-session -t "$session" 2>/dev/null; then echo "$session already running"; return; fi
  local vars
  vars=$(env | grep -E '^PILOT_[A-Z_]+=' | sed 's/=\(.*\)/="\1"/' | tr '\n' ' ' || true)
  tmux new-session -d -s "$session" \
    "tcsh -c 'source $PROJECT_ROOT/activate_project.csh && env $vars bash $PROJECT_ROOT/scripts/pilot_steps.sh $* |& tee $PROJECT_ROOT/logs/$session.log'"
  echo "started $session"
}

if [ "${PILOT_BACKEND:-openai}" != mock ]; then
  scripts/serve.sh 0 "$PORT_QWEN" "$QWEN"
  scripts/serve.sh 1 "$PORT_GEMMA" "$GEMMA"
fi
start pilot-data data
start pilot-qwen model qwen3.8-27b "$QWEN" "$PORT_QWEN"
start pilot-gemma model gemma4-31b "$GEMMA" "$PORT_GEMMA"
start pilot-eval eval qwen3.8-27b gemma4-31b
tmux ls
