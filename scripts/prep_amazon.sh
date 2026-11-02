#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
for ds in "${@:-amazon-beauty amazon-grocery}"; do
  for ds1 in $ds; do
    { python -m recmem.prepare --dataset "$ds1" &&
      python -m recmem.split --dataset "$ds1" --n-users 2500 &&
      python -m recmem.conditions --dataset "$ds1" &&
      python -m recmem.baselines --dataset "$ds1"; } || echo "PREP-FAILED $ds1"
  done
done
echo PREP-AMAZON-DONE
