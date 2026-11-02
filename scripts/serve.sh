#!/usr/bin/env bash
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 GPU PORT MODEL_ID [vllm args...]"; exit 1; }
: "${PROJECT_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
gpu=$1 port=$2 model=$3; shift 3
session="vllm-gpu${gpu}-p${port}"
mkdir -p "$PROJECT_ROOT/logs"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "$session already running (tmux kill-session -t $session to restart it)"; exit 0
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
  echo "port $port is already in use by another process; choose another port"; exit 1
fi
tmux new-session -d -s "$session" \
  "tcsh -c 'source $PROJECT_ROOT/activate_project.csh && setenv CUDA_VISIBLE_DEVICES $gpu && setenv VLLM_USE_DEEP_GEMM 0 && vllm serve $model --port $port --max-model-len 8192 $* |& tee $PROJECT_ROOT/logs/$session.log'"
echo "started $session: $model on GPU $gpu, http://localhost:$port/v1"
