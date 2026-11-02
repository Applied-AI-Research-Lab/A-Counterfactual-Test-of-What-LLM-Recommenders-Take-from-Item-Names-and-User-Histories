#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext_scheduler as base

base.STATE = base.ROOT / "logs" / "rev2"
base.SMALL_FOREIGN_GB = 0
base.MAX_PER_GPU = 1
base.MAX_GPUS = 2
base.POLL_S = 900


def jobs() -> list[dict]:
    out = []
    bx = "python -m recmem.baselines_ext"
    for ds in base.DATASETS:
        out.append({"id": f"rev2-embed-{ds}", "need": 8, "venv": "activate_project.csh",
                    "cmd": f"{bx} embed --dataset {ds}"})
        out.append({"id": f"rev2-catsasrec-{ds}", "need": 10, "venv": "activate_project.csh",
                    "after": f"rev2-embed-{ds}",
                    "cmd": "\n".join(f"{bx} textsasrec --dataset {ds} --view catonly --variant {v}"
                                     for v in ("textid", "text"))})
    return out


base.jobs = jobs

if __name__ == "__main__":
    base.main()
