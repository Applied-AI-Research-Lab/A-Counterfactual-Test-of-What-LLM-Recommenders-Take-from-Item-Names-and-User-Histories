#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
gpu=$1; shift
DATASETS="${GRID_DATASETS:-amazon-beauty amazon-grocery ml-1m amazon-videogames}"

for job in "$@"; do
  IFS=: read -r kind a b c d <<<"$job"
  echo "######## $(date '+%F %T') GPU $gpu: $job"
  case "$kind" in
    model)
      tag=$a model=$b port=$c
      session="vllm-gpu${gpu}-p${port}"
      extra=(); [ -n "${d:-}" ] && extra=(--gpu-memory-utilization "$d")
      bash scripts/serve.sh "$gpu" "$port" "$model" "${extra[@]}"
      tcsh -c "source $PROJECT_ROOT/activate_project.csh && bash scripts/grid_steps.sh model $tag $model $port $DATASETS"
      if [ "${KEEP_SERVER:-0}" != 1 ]; then tmux kill-session -t "$session" 2>/dev/null || true; sleep 20; fi
      ;;
    lora)
      tcsh -c "source $PROJECT_ROOT/activate_train.csh && setenv CUDA_VISIBLE_DEVICES $gpu && bash scripts/grid_steps.sh lora $a $DATASETS"
      ;;
    wait)
      until [ -e "$a" ]; do echo "waiting for $a ..."; sleep 60; done
      ;;
    *) echo "unknown job $job"; exit 1 ;;
  esac
done
echo "######## $(date '+%F %T') GPU $gpu: LANE DONE"
