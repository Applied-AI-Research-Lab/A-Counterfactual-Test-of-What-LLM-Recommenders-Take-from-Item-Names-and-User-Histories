from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

DATASETS = {
    "ml-1m": "movies",
    "amazon-videogames": "video game products",
    "amazon-beauty": "beauty and personal care products",
    "amazon-grocery": "grocery and gourmet food products",
}

CONDITIONS = ("real", "masked", "swapped", "described")


def proc_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        sys.exit(f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}")
    path = DATA_PROC / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {url}\n  -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.rename(dest)
    return dest


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
