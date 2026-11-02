from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import LAMBDAS, mean_ndcg10, score_rows, to_records, train_matrix
from .common import RESULTS, proc_dir, read_jsonl, write_jsonl
from .sasrec import DEFAULT_MAX_LEN, GRID, MAX_LEN, SASRec, left_pad

OUT = RESULTS / "ext" / "baselines"
MAIN = "test_n2500_c20_s42.jsonl"
VIEWS = ("real", "masked", "described", "content", "both")
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HIST = 20


def load(dataset: str, eval_set: Path | None):
    out = proc_dir(dataset)
    inter = pd.read_parquet(out / "interactions.parquet").sort_values(["user_id", "ts"], kind="stable")
    items = pd.read_parquet(out / "items.parquet")
    iidx = {it: i for i, it in enumerate(items["item_id"])}
    full = inter.groupby("user_id")["item_id"].apply(list).to_dict()
    test_path = eval_set or out / MAIN
    valid_path = test_path.with_name(test_path.name.replace("test_", "valid_", 1))
    return out, inter, items, iidx, full, test_path, read_jsonl(valid_path), read_jsonl(test_path)


def save(dataset: str, test_path: Path, name: str, condition: str, test, scorer, iidx, info: dict) -> float:
    recs = to_records(test, scorer, iidx, name, test_path.name)
    for r in recs:
        r["condition"] = condition
    base = OUT / dataset / test_path.stem / name
    write_jsonl(base / f"{condition}.jsonl", recs)
    test_ndcg = mean_ndcg10(score_rows(test, scorer, iidx))
    info = {**info, "test_ndcg@10": test_ndcg}
    (base / f"{condition}.tuning.json").write_text(json.dumps(info, indent=2))
    return test_ndcg


class TextSASRec(SASRec):

    def __init__(self, text: np.ndarray, max_len: int, use_id: bool, dropout: float, hidden: int = 64):
        super().__init__(len(text), max_len, hidden=hidden, dropout=dropout)
        self.register_buffer("text", torch.from_numpy(np.vstack([np.zeros((1, text.shape[1])), text]).astype(np.float32)))
        self.proj = nn.Linear(text.shape[1], hidden)
        self.proj_norm = nn.LayerNorm(hidden)
        self.use_id = use_id
        if not use_id:
            self.item.weight.requires_grad_(False)
            nn.init.zeros_(self.item.weight)

    def item_matrix(self) -> torch.Tensor:
        m = self.proj_norm(self.proj(self.text))
        m = m * (torch.arange(len(m), device=m.device) > 0).unsqueeze(-1)
        return m + self.item.weight if self.use_id else m

    def forward(self, seq: torch.Tensor, mat: torch.Tensor | None = None) -> torch.Tensor:
        mat = self.item_matrix() if mat is None else mat
        L = seq.shape[1]
        x = mat[seq] * math.sqrt(mat.shape[1]) + self.pos(torch.arange(L, device=seq.device))
        x = self.drop(x) * (seq > 0).unsqueeze(-1)
        x = self.encoder(x, mask=self.causal[:L, :L], is_causal=True)
        return self.norm(x)

    def scores(self, seq: torch.Tensor) -> torch.Tensor:
        mat = self.item_matrix()
        return self.forward(seq, mat)[:, -1] @ mat[1:].T


def make_scorer(model, seqs: dict, iidx: dict, max_len: int, device):
    cache: dict = {}
    model.eval()

    def scorer(u, cols):
        if u not in cache:
            with torch.no_grad():
                x = torch.from_numpy(left_pad([[iidx[i] + 1 for i in seqs[u]]], max_len)).to(device)
                cache[u] = model.scores(x)[0].float().cpu().numpy()
        return cache[u][cols]

    return scorer


def train(build, inputs, targets, valid, user_seq, iidx, max_len, device, epochs, seed, batch_size=128, patience=5,
          label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, betas=(0.9, 0.98))
    best, best_state, best_epoch, curve = -1.0, None, 0, []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch_size):
            idx = perm[start:start + batch_size]
            x = torch.from_numpy(inputs[idx]).to(device)
            y = torch.from_numpy(targets[idx]).to(device)
            if isinstance(model, TextSASRec):
                mat = model.item_matrix()
                logits = model.forward(x, mat) @ mat[1:].T
            else:
                logits = model.forward(x) @ model.item.weight[1:].T
            mask = y > 0
            loss = nn.functional.cross_entropy(logits[mask], y[mask] - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ndcg = mean_ndcg10(score_rows(valid, make_scorer(model, user_seq(2), iidx, max_len, device), iidx))
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_ndcg@10": ndcg,
                      "seconds": round(time.time() - t0, 1)})
        print(f"  {label} epoch {epoch}: loss {np.mean(losses):.4f}, valid NDCG@10 {ndcg:.4f}", flush=True)
        if ndcg > best:
            best, best_epoch = ndcg, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "valid_ndcg@10": best, "curve": curve}


def windows(seqs: list[list[int]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    ins, outs = [], []
    for s in seqs:
        end = len(s)
        while end >= 2:
            w = s[max(0, end - max_len - 1):end]
            ins.append(w[:-1])
            outs.append(w[1:])
            end -= max_len
    return left_pad(ins, max_len), left_pad(outs, max_len)


def tune_sasrec(build_for, inputs, targets, valid, user_seq, iidx, max_len, device, epochs, seed, label):
    best_model, best_info, runs = None, None, []
    for cfg in GRID:
        model, info = train(lambda: build_for(cfg["dropout"]), inputs, targets, valid, user_seq, iidx, max_len,
                            device, epochs, seed, label=f"{label} {cfg}")
        info["config"] = cfg
        runs.append(info)
        if best_info is None or info["valid_ndcg@10"] > best_info["valid_ndcg@10"]:
            best_model, best_info = model, info
    return best_model, {"best": best_info["config"], "best_epoch": best_info["best_epoch"],
                        "valid_ndcg@10": best_info["valid_ndcg@10"], "runs": runs}


def done(dataset: str, eval_set: Path | None, name: str, condition: str) -> bool:
    path = OUT / dataset / (eval_set or Path(MAIN)).stem / name / f"{condition}.jsonl"
    if path.exists():
        print(f"{path} exists, skipped")
    return path.exists()


def cmd_sasrec20(args) -> None:
    if done(args.dataset, args.eval_set, "sasrec20", "idonly"):
        return
    out, inter, items, iidx, full, test_path, valid, test = load(args.dataset, args.eval_set)
    user_seq = lambda drop: {u: s[:-drop] for u, s in full.items()}
    train_seqs = [[iidx[i] + 1 for i in s[:-2]] for s in full.values() if len(s) >= 4]
    inputs, targets = windows(train_seqs, HIST)
    print(f"SASRec-20 {args.dataset}: {len(train_seqs):,} sequences -> {len(inputs):,} windows", flush=True)
    model, info = tune_sasrec(lambda d: SASRec(len(items), HIST, dropout=d), inputs, targets, valid, user_seq, iidx,
                              HIST, args.device, args.epochs, args.seed, "sasrec20")
    nd = save(args.dataset, test_path, "sasrec20", "idonly", test,
              make_scorer(model, user_seq(1), iidx, HIST, args.device), iidx, info)
    print(f"SASRec-20 {args.dataset}: best {info['best']} epoch {info['best_epoch']}, test NDCG@10 {nd:.4f}")


def cmd_ease20(args) -> None:
    if done(args.dataset, args.eval_set, "ease20", "idonly"):
        return
    out, inter, items, iidx, full, test_path, valid, test = load(args.dataset, args.eval_set)
    uidx = {u: i for i, u in enumerate(inter["user_id"].unique())}
    dev = args.device

    def gram(X):
        return torch.from_numpy((X.T @ X).toarray()).double().to(dev)

    def weights(G, lam):
        A = G.clone()
        A.diagonal().add_(lam)
        P = torch.linalg.inv(A)
        B = P / -torch.diag(P)
        B.fill_diagonal_(0.0)
        return B

    def scorer_for(B, rows):
        hist = {r["user_id"]: [iidx[i] for i in r["history"][-HIST:]] for r in rows}
        cache = {}

        def scorer(u, cols):
            if u not in cache:
                cache[u] = B[torch.tensor(hist[u], device=dev)].sum(0).cpu().numpy()
            return cache[u][cols]
        return scorer

    Gv = gram(train_matrix(inter, 2, uidx, iidx))
    tuning = {}
    for lam in LAMBDAS:
        tuning[lam] = mean_ndcg10(score_rows(valid, scorer_for(weights(Gv, lam), valid), iidx))
        print(f"EASE-20 lambda={lam:>5}: valid NDCG@10={tuning[lam]:.4f}", flush=True)
    del Gv
    best = max(tuning, key=tuning.get)
    B = weights(gram(train_matrix(inter, 1, uidx, iidx)), best)
    nd = save(args.dataset, test_path, "ease20", "idonly", test, scorer_for(B, test), iidx,
              {"best_lambda": best, "valid_ndcg@10": tuning})
    print(f"EASE-20 {args.dataset}: lambda {best}, test NDCG@10 {nd:.4f}")


def view_texts(dataset: str, view: str) -> tuple[list, list[str]]:
    from .run_ext import ext_views
    views = ext_views(dataset)
    items = pd.read_parquet(proc_dir(dataset) / "items.parquet")["item_id"].tolist()
    if view == "catonly":
        return items, [f"[{views['category'][i]}]" for i in items]
    return items, [f"{views[view][i]} [{views['category'][i]}]" for i in items]


def emb_path(dataset: str, view: str) -> Path:
    return proc_dir(dataset) / f"emb_{view}_qwen3emb.npy"


def cmd_embed(args) -> None:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL, padding_side="left")
    model = AutoModel.from_pretrained(EMB_MODEL, dtype=torch.float16).to(args.device).eval()
    for view in (*VIEWS, "catonly"):
        path = emb_path(args.dataset, view)
        if path.exists():
            print(f"{path.name} exists")
            continue
        items, texts = view_texts(args.dataset, view)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for b in range(0, len(order), 256):
                idx = order[b:b + 256]
                enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(args.device)
                h = model(**enc).last_hidden_state[:, -1]
                embs[idx] = nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
        np.save(path, embs)
        print(f"{args.dataset} {view}: {len(texts):,} items, e.g. {texts[0]!r} -> {path.name}", flush=True)


def cmd_textsasrec(args) -> None:
    prefix = "catsasrec" if args.view == "catonly" else "textsasrec"
    if done(args.dataset, args.eval_set, f"{prefix}-{args.variant}", args.view):
        return
    out, inter, items, iidx, full, test_path, valid, test = load(args.dataset, args.eval_set)
    text = np.load(emb_path(args.dataset, args.view))
    assert len(text) == len(items)
    max_len = MAX_LEN.get(args.dataset, DEFAULT_MAX_LEN)
    user_seq = lambda drop: {u: s[:-drop] for u, s in full.items()}
    train_seqs = [[iidx[i] + 1 for i in s[:-2]] for s in full.values() if len(s) >= 4]
    inputs = left_pad([s[:-1] for s in train_seqs], max_len)
    targets = left_pad([s[1:] for s in train_seqs], max_len)
    name = f"{prefix}-{args.variant}"
    print(f"{name} {args.dataset}/{args.view}: {len(train_seqs):,} sequences, max_len {max_len}", flush=True)
    model, info = tune_sasrec(lambda d: TextSASRec(text, max_len, args.variant == "textid", d), inputs, targets,
                              valid, user_seq, iidx, max_len, args.device, args.epochs, args.seed,
                              f"{name}/{args.view}")
    info["embedding_model"] = EMB_MODEL
    nd = save(args.dataset, test_path, name, args.view, test, make_scorer(model, user_seq(1), iidx, max_len, args.device),
              iidx, info)
    print(f"{name} {args.dataset}/{args.view}: best {info['best']} epoch {info['best_epoch']}, test NDCG@10 {nd:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["embed", "sasrec20", "ease20", "textsasrec"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--view", default="real", choices=[*VIEWS, "catonly"])
    ap.add_argument("--variant", default="textid", choices=["text", "textid"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"embed": cmd_embed, "sasrec20": cmd_sasrec20, "ease20": cmd_ease20, "textsasrec": cmd_textsasrec}[args.command](args)


if __name__ == "__main__":
    main()
