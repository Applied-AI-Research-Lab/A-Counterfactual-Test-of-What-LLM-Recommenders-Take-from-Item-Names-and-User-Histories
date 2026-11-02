#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_scheduler as base

base.STATE = base.ROOT / "logs" / "rev"
VIEWS = ("real", "masked", "described", "content", "both")


def jobs() -> list[dict]:
    out = []
    for tag, model, need, port in base.VLLM:
        out.append({"id": f"rev-vllm-{tag}", "need": need, "venv": "activate_project.csh",
                    "cmd": f"bash scripts/rev_vllm_job2.sh $GPU {port + 80} {model} {tag} {need}"})
    bx = "python -m recmem.baselines_ext"
    for ds in base.DATASETS:
        out.append({"id": f"rev-embed-{ds}", "need": 8, "venv": "activate_project.csh",
                    "cmd": f"{bx} embed --dataset {ds}"})
        out.append({"id": f"rev-ease20-{ds}", "need": 40, "venv": "activate_project.csh",
                    "cmd": f"{bx} ease20 --dataset {ds}"})
        out.append({"id": f"rev-sasrec20-{ds}", "need": 8, "venv": "activate_project.csh",
                    "cmd": f"{bx} sasrec20 --dataset {ds}"})
        for variant in ("textid", "text"):
            out.append({"id": f"rev-textsasrec-{variant}-{ds}", "need": 10, "venv": "activate_project.csh",
                        "after": f"rev-embed-{ds}",
                        "cmd": "\n".join(f"{bx} textsasrec --dataset {ds} --view {v} --variant {variant}" for v in VIEWS)})
    return out


base.jobs = jobs

if __name__ == "__main__":
    base.main()
