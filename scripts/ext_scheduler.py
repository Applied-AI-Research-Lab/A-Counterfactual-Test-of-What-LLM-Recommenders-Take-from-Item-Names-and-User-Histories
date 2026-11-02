#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "logs" / "ext"
ME = os.environ.get("USER", "")
MAX_GPUS, MAX_PER_GPU, MAX_TRIES = 3, 2, 3
STABLE_MIN, COOLDOWN_MIN, POLL_S = 10, 4, 60
SMALL_FOREIGN_GB, MARGIN_GB, MARGIN_FOREIGN_GB = 8, 6, 20
DATASETS = ["ml-1m", "amazon-videogames", "amazon-beauty", "amazon-grocery"]
MAIN = "data/processed/{ds}/test_n2500_c20_s42.jsonl"

VLLM = [
    ("qwen3.8-27b", "Qwen/Qwen3.8-27B-FP8", 62, 8210),
    ("qwen3.5-27b", "Qwen/Qwen3.5-27B-FP8", 62, 8211),
    ("gemma4-31b", "google/gemma-4-31B-it", 84, 8212),
    ("gemma4-12b", "google/gemma-4-12B-it", 42, 8213),
    ("qwen3.5-9b", "Qwen/Qwen3.5-9B", 34, 8214),
    ("gemma4-e4b", "google/gemma-4-E4B-it", 30, 8215),
    ("qwen3.5-4b", "Qwen/Qwen3.5-4B", 26, 8216),
]
LORA_MODELS = {"qwen3.5-4b": (26, 36), "gemma4-e4b": (30, 42)}


def jobs() -> list[dict]:
    out = []
    check = ("python -m recmem.lora_ext --dataset amazon-grocery --model {m} {mode} --limit 3 --orders 1 "
             "--eval-set " + MAIN.format(ds="amazon-grocery"))
    for m in LORA_MODELS:
        out.append({"id": f"lora-check-{m}", "need": LORA_MODELS[m][1], "venv": "activate_train.csh", "cmd": "\n".join([
            "export PROJECT_ROOT=$(mktemp -d)", "mkdir -p $PROJECT_ROOT/data",
            f"ln -s {ROOT}/data/processed $PROJECT_ROOT/data/processed", f"ln -s {ROOT}/checkpoints $PROJECT_ROOT/checkpoints",
            check.format(m=m, mode="--existing"),
            check.format(m=m, mode="--train-condition real --seed 2 --max-steps 3"),
            "rm -rf $PROJECT_ROOT"])})
    for tag, model, need, port in VLLM:
        lines = [f"bash scripts/ext_vllm_job.sh $GPU {port} {model} {tag} {need}"]
        out.append({"id": f"vllm-{tag}", "need": need, "venv": "activate_project.csh", "cmd": "\n".join(lines)})
    for m in LORA_MODELS:
        for ds in DATASETS:
            out.append({"id": f"lora-existing-{m}-{ds}", "need": LORA_MODELS[m][0], "venv": "activate_train.csh",
                        "after": f"lora-check-{m}",
                        "cmd": f"python -m recmem.lora_ext --dataset {ds} --model {m} --existing --eval-set {MAIN.format(ds=ds)}"})
    for k in (0, 1):
        for m in LORA_MODELS:
            for ds in DATASETS:
                for cond in ("real", "masked"):
                    seeds = (3, 4) if (m, ds, cond) == ("gemma4-e4b", "amazon-grocery", "masked") else (2, 3)
                    out.append({"id": f"lora-seed-{m}-{ds}-{cond}-s{seeds[k]}", "need": LORA_MODELS[m][1],
                                "venv": "activate_train.csh", "after": f"lora-check-{m}",
                                "cmd": f"python -m recmem.lora_ext --dataset {ds} --model {m} --train-condition {cond} "
                                       f"--seed {seeds[k]} --eval-set {MAIN.format(ds=ds)}"})
    return out


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(STATE / "scheduler.log", "a") as fh:
        fh.write(line + "\n")


def gpu_state() -> list[dict]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in q.strip().splitlines():
        i, uuid, total, free = [x.strip() for x in line.split(",")]
        gpus.append({"i": int(i), "uuid": uuid, "total": int(total) / 1024, "free": int(free) / 1024, "foreign": {}})
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout
    by_uuid = {g["uuid"]: g for g in gpus}
    for line in apps.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, mem = [x.strip() for x in line.split(",")]
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if user and user != ME and uuid in by_uuid:
            by_uuid[uuid]["foreign"][pid] = int(mem) / 1024
    return gpus


def session(jid: str) -> str:
    return "ext-" + jid.replace(".", "_").replace(":", "_")


def tmux_alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode == 0


def launch(job: dict, gpu: int) -> None:
    jid = job["id"]
    script = STATE / "jobs" / f"{jid}.sh"
    status = STATE / "status" / jid
    status.unlink(missing_ok=True)
    script.write_text("\n".join([
        "#!/usr/bin/env bash", "set -eo pipefail", f"cd {ROOT}", f"export GPU={gpu} CUDA_VISIBLE_DEVICES={gpu}",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"trap 'echo $? > {status}' EXIT", job["cmd"], ""]))
    cmd = f"tcsh -c 'source {ROOT}/{job['venv']} && bash {script} |& tee -a {STATE}/{jid}.log'"
    subprocess.run(["tmux", "new-session", "-d", "-s", session(jid), cmd], check=True)
    (STATE / "running" / jid).write_text(str(gpu))
    log(f"START {jid} on GPU {gpu} (needs {job['need']} GB)")


def main() -> None:
    for d in ("jobs", "status", "done", "tries", "running"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    queue = jobs()
    history: dict[int, list] = {}
    cooldown: dict[int, float] = {}
    log(f"scheduler started: {len(queue)} jobs, {sum((STATE / 'done' / j['id']).exists() for j in queue)} done")
    while True:
        now = time.time()
        for f in list((STATE / "running").iterdir()):
            jid = f.name
            if tmux_alive(session(jid)):
                continue
            f.unlink()
            st = (STATE / "status" / jid).read_text().strip() if (STATE / "status" / jid).exists() else "?"
            if st == "0":
                (STATE / "done" / jid).touch()
                log(f"DONE  {jid}")
            else:
                tries = STATE / "tries" / jid
                n = int(tries.read_text()) + 1 if tries.exists() else 1
                tries.write_text(str(n))
                log(f"FAIL  {jid} (exit {st}, try {n}/{MAX_TRIES}) - see logs/ext/{jid}.log")
        running = {f.name: int(f.read_text()) for f in (STATE / "running").iterdir()}
        pending = [j for j in queue if not (STATE / "done" / j["id"]).exists() and j["id"] not in running
                   and (int((STATE / "tries" / j["id"]).read_text()) if (STATE / "tries" / j["id"]).exists() else 0) < MAX_TRIES
                   and (not j.get("after") or (STATE / "done" / j["after"]).exists())]
        if not pending and not running:
            log("ALL JOBS FINISHED (or out of tries)")
            break
        gpus = gpu_state()
        for g in gpus:
            h = history.setdefault(g["i"], [])
            h.append((now, set(g["foreign"]), sum(g["foreign"].values())))
            history[g["i"]] = [x for x in h if now - x[0] <= STABLE_MIN * 60 + POLL_S]
        ours = {}
        for jid, gi in running.items():
            ours.setdefault(gi, []).append(jid)
        started = False
        for job in pending:
            for g in sorted(gpus, key=lambda g: (g["i"] not in ours, -g["free"])):
                h = history[g["i"]]
                observed = h[-1][0] - h[0][0] >= STABLE_MIN * 60
                stable = observed and all(x[1] == h[-1][1] and x[2] <= SMALL_FOREIGN_GB for x in h)
                if not stable or now < cooldown.get(g["i"], 0):
                    continue
                if g["i"] not in ours and len(ours) >= MAX_GPUS:
                    continue
                if len(ours.get(g["i"], [])) >= MAX_PER_GPU:
                    continue
                margin = MARGIN_FOREIGN_GB if g["foreign"] else MARGIN_GB
                if g["free"] < job["need"] + margin:
                    continue
                biggest = max(j["need"] for j in pending)
                if job["need"] < biggest and g["free"] >= biggest + margin:
                    continue
                job["gpu_total"] = g["total"]
                launch(job, g["i"])
                cooldown[g["i"]] = now + COOLDOWN_MIN * 60
                ours.setdefault(g["i"], []).append(job["id"])
                started = True
                break
        if not started and int(now) % 1800 < POLL_S:
            log("waiting: " + "; ".join(f"GPU{g['i']} free {g['free']:.0f} GB, foreign {sum(g['foreign'].values()):.0f} GB"
                                        for g in gpus) + f"; pending {len(pending)}, running {len(running)}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
