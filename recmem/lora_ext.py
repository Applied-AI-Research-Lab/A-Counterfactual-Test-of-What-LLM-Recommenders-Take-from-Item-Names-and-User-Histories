from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .common import CONDITIONS, DATASETS, ROOT, read_jsonl
from .lora import (MODELS, LORA, TRAIN, label_token_ids, last_logits, load_model, prompt_ids, train,
                   training_examples)
from .prompts import labels
from .run import make_job
from .run_ext import EXT_CONDITIONS, EXT_RESULTS, ext_jobs, ext_views

ALL_CONDITIONS = (*CONDITIONS, *EXT_CONDITIONS)


def jobs_for(rows, views, cond, noun, history_len, orders, done) -> list[dict]:
    if cond in EXT_CONDITIONS:
        return ext_jobs(rows, views, cond, noun, history_len, orders, done)
    return [make_job(row, k, views, cond, noun, history_len)
            for row in rows for k in range(min(orders, len(row["candidates"]))) if (row["user_id"], k) not in done]


def evaluate(model, tok, dataset, eval_path: Path, tag, conditions, history_len, orders, bs, device, model_name, log,
             limit=None):
    import torch
    rows = read_jsonl(eval_path)[:limit]
    views = ext_views(dataset)
    for cond in conditions:
        path = EXT_RESULTS / dataset / eval_path.stem / tag / f"{cond}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(r["user_id"], r["order_idx"]) for r in read_jsonl(path)} if path.exists() else set()
        jobs = jobs_for(rows, views, cond, DATASETS[dataset], history_len, orders, done)
        log(f"[{tag}] {dataset}/{cond}: {len(jobs):,} to score ({len(done):,} already done)")
        if not jobs:
            continue
        n = len(jobs[0]["candidates"])
        lab_ids, labs = label_token_ids(tok, n), labels(n)
        enc = [prompt_ids(tok, j["messages"], TRAIN["max_len"]) for j in jobs]
        order = sorted(range(len(jobs)), key=lambda i: len(enc[i]))
        with open(path, "a", encoding="utf-8") as fh:
            for b in range(0, len(order), bs):
                idx = order[b:b + bs]
                t0 = time.perf_counter()
                with torch.inference_mode():
                    lp = torch.log_softmax(last_logits(model, [enc[i] for i in idx], tok.pad_token_id, device).float(), -1)
                sel = lp[:, lab_ids].cpu().numpy()
                lat = (time.perf_counter() - t0) / len(idx)
                lines = []
                for j, i in enumerate(idx):
                    rec = {k: v for k, v in jobs[i].items() if k not in ("messages", "max_tokens")}
                    rec.update(model=model_name, eval_set=eval_path.name,
                               response=", ".join(labs[k] for k in np.argsort(-sel[j], kind="stable")),
                               latency_s=round(lat, 4), prompt_tokens=len(enc[i]), completion_tokens=0,
                               label_mass=float(np.exp(sel[j]).sum()))
                    lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.write("".join(lines))
                fh.flush()


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--existing", action="store_true")
    group.add_argument("--train-condition", choices=["real", "masked"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="quick check: only the first N users")
    ap.add_argument("--max-steps", type=int, default=None, help="quick check: stop training early (not saved)")
    args = ap.parse_args()
    if args.train_condition and args.seed in (None, 1):
        raise SystemExit("--seed must be given and differ from 1 (seed 1 is the original run)")

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    device, name = "cuda", MODELS[args.model]
    if args.existing:
        from peft import PeftModel
        tok, base = load_model(name, device)
        evaluate(base, tok, args.dataset, args.eval_set, f"{args.model}-ro", EXT_CONDITIONS, args.history_len,
                 args.orders, args.eval_bs, device, name, log, args.limit)
        for cond in ("real", "masked"):
            tag = f"{args.model}-lora-{cond}"
            model = PeftModel.from_pretrained(base, ROOT / "checkpoints" / args.dataset / tag).eval()
            log(f"loaded adapter checkpoints/{args.dataset}/{tag} (read-only)")
            evaluate(model, tok, args.dataset, args.eval_set, tag, EXT_CONDITIONS, args.history_len, args.orders,
                     args.eval_bs, device, name, log, args.limit)
            base = model.unload()
        log(f"done existing on {args.dataset}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tag = f"{args.model}-lora-{args.train_condition}-s{args.seed}"
    ckpt = ROOT / "checkpoints_ext" / args.dataset / tag
    tok, model = load_model(name, device)
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ckpt).eval()
        log(f"loaded adapter {ckpt}")
    else:
        examples = training_examples(args.dataset, args.train_condition, TRAIN["examples"], args.history_len,
                                     len(read_jsonl(args.eval_set)[0]["candidates"][0]), seed=1000 + args.seed)
        log(f"{tag}: {len(examples):,} training examples on {args.dataset}")
        model, tinfo = train(model, tok, examples, device, args.seed, args.max_steps, log)
        info = {"tag": tag, "dataset": args.dataset, "model": name, "eval_set": args.eval_set.name, "seed": args.seed,
                **tinfo, "lora": LORA, "train": TRAIN | {"examples": len(examples)}}
        if not args.max_steps:
            model.save_pretrained(ckpt)
            (ckpt / "train_info.json").write_text(json.dumps(info, indent=1))
    evaluate(model, tok, args.dataset, args.eval_set, tag, ALL_CONDITIONS, args.history_len, args.orders,
             args.eval_bs, device, name, log, args.limit)
    log(f"done {tag} on {args.dataset}")


if __name__ == "__main__":
    main()
