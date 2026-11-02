#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PROJECT_ROOT:=$PWD}"; export PROJECT_ROOT
gpu=$1 port=$2 model=$3 tag=$4 need=$5
session="vllm-gpu${gpu}-p${port}"
total=$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits)
util=$(python3 -c "print(f'{$need * 1024 / $total:.3f}')")
trap 'tmux kill-session -t "=$session" 2>/dev/null || true' EXIT

bash scripts/serve.sh "$gpu" "$port" "$model" --gpu-memory-utilization "$util" --max-num-seqs 256 --max-logprobs 100 --logprobs-mode processed_logprobs
for _ in $(seq 1 90); do
  curl -sf "http://localhost:$port/v1/models" 2>/dev/null | grep -qF "\"$model\"" && break
  tmux has-session -t "=$session" 2>/dev/null || { echo "server $session died; see logs/$session.log"; exit 1; }
  echo "waiting for $model on port $port ..."; sleep 20
done
curl -sf "http://localhost:$port/v1/models" | grep -qF "\"$model\"" || { echo "$model not served after 30 min"; exit 1; }

DS="ml-1m amazon-videogames amazon-beauty amazon-grocery"
for ds in $DS; do
  python -m recmem.run_ext --dataset "$ds" --tag "$tag" --model "$model" --base-url "http://localhost:$port/v1" \
    --eval-set "data/processed/$ds/test_n500_c20_s42_a0.jsonl" --workers 64
done
for ds in $DS; do
  python -m recmem.readout_ext --dataset "$ds" --tag "$tag" --model "$model" --base-url "http://localhost:$port/v1" \
    --eval-set "data/processed/$ds/test_n2500_c20_s42.jsonl" --workers 96 \
    --conditions real,masked,content,described,both,realshuf,swapped,maskshuf
done
echo "REV VLLM DONE $tag"
