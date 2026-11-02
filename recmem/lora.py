from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CONDITIONS, DATASETS, RESULTS, ROOT, proc_dir, read_jsonl
from .prompts import build_messages, labels
from .run import default_split, make_job
from .split import sample_negatives

MODELS = {"qwen3.5-4b": "Qwen/Qwen3.5-4B", "gemma4-e4b": "google/gemma-4-E4B-it"}
LORA = {"r": 16, "alpha": 32, "dropout": 0.05}
TRAIN = {"examples": 10_000, "epochs": 1, "lr": 2e-4, "batch_size": 16, "warmup_ratio": 0.03, "max_len": 4096}
MIN_HISTORY = 5


def training_examples(dataset: str, condition: str, n: int, history_len: int, n_cand: int, seed: int) -> list[dict]:
    out = proc_dir(dataset)
    inter = pd.read_parquet(out / "interactions.parquet")
    items = pd.read_parquet(out / "items.parquet")
    seqs = inter.sort_values(["user_id", "ts"], kind="stable").groupby("user_id")["item_id"].apply(list)
    views_df = pd.read_parquet(out / "views_s42.parquet").set_index("item_id")
    views = {col: views_df[col].to_dict() for col in ("category", *CONDITIONS)}
    item_ids = items["item_id"].to_numpy()
    probs = items["popularity"].to_numpy(dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    users = [u for u in sorted(seqs.index) if len(seqs[u]) >= MIN_HISTORY + 3]
    users = rng.choice(users, size=min(n, len(users)), replace=False)
    rows = []
    for u in users:
        seq = seqs[u]
        pos = int(rng.integers(MIN_HISTORY, len(seq) - 2))
        negs = sample_negatives(rng, item_ids, probs, set(seq), n_cand - 1)
        row = {"user_id": u, "history": seq[:pos], "target": seq[pos],
               "candidates": [list(rng.permutation([seq[pos]] + negs))]}
        rows.append(make_job(row, 0, views, condition, DATASETS[dataset], history_len))
    return rows


def load_model(name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
    model.to(device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def label_token_ids(tok, n: int) -> list[int]:
    ids = []
    for lab in labels(n):
        enc = tok.encode(lab, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"label {lab!r} is not a single token for this tokenizer: {enc}")
        ids.append(enc[0])
    return ids


def prompt_ids(tok, messages: list[dict], max_len: int) -> list[int]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > max_len:
        raise ValueError(f"prompt of {len(ids)} tokens exceeds max_len {max_len}")
    return ids


def last_logits(model, batch: list[list[int]], pad_id: int, device: str):
    import torch
    L = max(map(len, batch))
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, p in enumerate(batch):
        ids[j, :len(p)] = torch.tensor(p)
        att[j, :len(p)] = 1
    lasts = [len(p) - 1 for p in batch]
    keep = sorted(set(lasts))
    out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                logits_to_keep=torch.tensor(keep, device=device))
    col = {p: i for i, p in enumerate(keep)}
    return out.logits[torch.arange(len(batch), device=out.logits.device),
                      torch.tensor([col[p] for p in lasts], device=out.logits.device)]


def train(model, tok, examples: list[dict], device: str, seed: int, max_steps: int | None, log) -> tuple:
    import torch
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    for prm in model.parameters():
        if prm.requires_grad:
            prm.data = prm.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lab_ids = label_token_ids(tok, len(examples[0]["candidates"]))
    data = [(prompt_ids(tok, ex["messages"], TRAIN["max_len"]), lab_ids[labels(len(ex["candidates"])).index(ex["target_label"])])
            for ex in examples]
    rng = random.Random(seed)
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]) + rng.random())
    bl = [order[i:i + TRAIN["batch_size"]] for i in range(0, len(order), TRAIN["batch_size"])]
    rng.shuffle(bl)
    steps = math.ceil(len(bl) * TRAIN["epochs"])
    steps = min(steps, max_steps) if max_steps else steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TRAIN["lr"], weight_decay=0.0)
    warm = max(1, int(TRAIN["warmup_ratio"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    model.train()
    losses, accs, t0 = [], [], time.time()
    for step in range(steps):
        idx = bl[step % len(bl)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = last_logits(model, [data[i][0] for i in idx], tok.pad_token_id, device)
        y = torch.tensor([data[i][1] for i in idx], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        losses.append(float(loss.detach()))
        accs.append(float((logits[:, lab_ids].argmax(-1) == torch.tensor(
            [lab_ids.index(data[i][1]) for i in idx], device=logits.device)).float().mean()))
        if step % max(1, min(50, steps // 4)) == 0 or step == steps - 1:
            log(f"step {step}/{steps} loss {np.mean(losses[-50:]):.4f} top1 {np.mean(accs[-50:]):.3f} "
                f"({time.time() - t0:.0f}s)")
    model.eval()
    return model, {"steps": steps, "train_seconds": round(time.time() - t0, 1), "n_examples": len(data),
                   "loss_first50": float(np.mean(losses[:50])), "loss_last50": float(np.mean(losses[-50:])),
                   "top1_last50": float(np.mean(accs[-50:]))}


def evaluate(model, tok, dataset: str, eval_path: Path, tag: str, conditions, history_len: int, orders: int,
             limit: int | None, bs: int, device: str, model_name: str, log, root: Path = RESULTS / "raw") -> None:
    import torch
    out = proc_dir(dataset)
    rows = read_jsonl(eval_path)[:limit]
    views_df = pd.read_parquet(out / "views_s42.parquet").set_index("item_id")
    views = {col: views_df[col].to_dict() for col in ("category", *CONDITIONS)}
    for cond in conditions:
        path = root / dataset / eval_path.stem / tag / f"{cond}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {(r["user_id"], r["order_idx"]) for r in read_jsonl(path)} if path.exists() else set()
        jobs = [make_job(row, k, views, cond, DATASETS[dataset], history_len)
                for row in rows for k in range(min(orders, len(row["candidates"])))
                if (row["user_id"], k) not in done]
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
                for j, i in enumerate(idx):
                    job = jobs[i]
                    ranked = [labs[k] for k in np.argsort(-sel[j], kind="stable")]
                    rec = {k: v for k, v in job.items() if k not in ("messages", "max_tokens")}
                    rec.update(model=model_name, eval_set=eval_path.name, response=", ".join(ranked),
                               latency_s=round(lat, 4), prompt_tokens=len(enc[i]), completion_tokens=0,
                               label_mass=float(np.exp(sel[j]).sum()))
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if (b // bs) % 200 == 0:
                    log(f"  {b + len(idx):,}/{len(jobs):,}")


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--train-condition", choices=["real", "masked"])
    group.add_argument("--base", action="store_true", help="zero-shot readout of the base model, no training")
    ap.add_argument("--eval-set", type=Path, default=None, help="defaults to the newest main test set")
    ap.add_argument("--condition", default="all", choices=["all", *CONDITIONS], help="evaluation condition(s)")
    ap.add_argument("--history-len", type=int, default=20)
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--examples", type=int, default=TRAIN["examples"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N evaluation users")
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=TRAIN["lr"],
                    help="learning rate (default as in BP13; lower only for a run that collapsed, and report it)")
    args = ap.parse_args()

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    device = "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    eval_path = args.eval_set or default_split(proc_dir(args.dataset), "test")
    tag = f"{args.model}-ro" if args.base else f"{args.model}-lora-{args.train_condition}"
    TRAIN["lr"] = args.lr
    tok, model = load_model(MODELS[args.model], device)
    info = {"tag": tag, "dataset": args.dataset, "model": MODELS[args.model], "eval_set": eval_path.name}
    if not args.base:
        ckpt = ROOT / "checkpoints" / args.dataset / tag
        if (ckpt / "adapter_config.json").exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ckpt).eval()
            log(f"loaded adapter {ckpt}")
        else:
            examples = training_examples(args.dataset, args.train_condition, args.examples, args.history_len,
                                         len(read_jsonl(eval_path)[0]["candidates"][0]), seed=1000 + args.seed)
            log(f"{tag}: {len(examples):,} training examples on {args.dataset}")
            model, tinfo = train(model, tok, examples, device, args.seed, args.max_steps, log)
            info.update(tinfo, lora=LORA, train=TRAIN | {"examples": len(examples)})
            if not args.max_steps:
                model.save_pretrained(ckpt)
                (ckpt / "train_info.json").write_text(json.dumps(info, indent=1))
    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    root = RESULTS / ("checks" if args.max_steps else "raw")
    evaluate(model, tok, args.dataset, eval_path, tag, conditions, args.history_len, args.orders, args.limit,
             args.eval_bs, device, MODELS[args.model], log, root)
    log(f"done {tag} on {args.dataset}")


if __name__ == "__main__":
    main()
