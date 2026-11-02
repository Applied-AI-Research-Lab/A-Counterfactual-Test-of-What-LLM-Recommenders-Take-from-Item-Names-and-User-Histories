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

from .baselines import mean_ndcg10, score_rows, to_records
from .common import RESULTS, proc_dir, read_jsonl, write_jsonl
from .run import default_split

MAX_LEN = {"ml-1m": 200}
DEFAULT_MAX_LEN = 50
GRID = [{"dropout": 0.2}, {"dropout": 0.5}]


class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int, hidden: int = 64, blocks: int = 2, heads: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden, dropout, batch_first=True, norm_first=True,
                                           activation="relu")
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        x = self.item(seq) * math.sqrt(self.item.embedding_dim) + self.pos(torch.arange(L, device=seq.device))
        x = self.drop(x) * (seq > 0).unsqueeze(-1)
        x = self.encoder(x, mask=self.causal[:L, :L], is_causal=True)
        return self.norm(x)

    def scores(self, seq: torch.Tensor) -> torch.Tensor:
        return self.forward(seq)[:, -1] @ self.item.weight[1:].T


def left_pad(seqs: list[list[int]], max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        if s:
            out[i, -len(s):] = s
    return out


def train_config(cfg: dict, train_seqs, valid_rows, user_seq, iidx, n_items, max_len, device, epochs, seed,
                 batch_size: int = 128, patience: int = 5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SASRec(n_items, max_len, dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    inputs = left_pad([s[:-1] for s in train_seqs], max_len)
    targets = left_pad([s[1:] for s in train_seqs], max_len)
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
            logits = model.forward(x) @ model.item.weight[1:].T
            mask = y > 0
            loss = nn.functional.cross_entropy(logits[mask], y[mask] - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ndcg = mean_ndcg10(score_rows(valid_rows, make_scorer(model, user_seq(2), iidx, max_len, device), iidx))
        curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_ndcg@10": ndcg,
                      "seconds": round(time.time() - t0, 1)})
        print(f"  {cfg} epoch {epoch}: loss {np.mean(losses):.4f}, valid NDCG@10 {ndcg:.4f}", flush=True)
        if ndcg > best:
            best, best_epoch = ndcg, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"config": cfg, "best_epoch": best_epoch, "valid_ndcg@10": best, "curve": curve}


def make_scorer(model, seqs: dict, iidx: dict, max_len: int, device):
    cache: dict = {}
    model.eval()

    def scorer(u, cols):
        if u not in cache:
            with torch.no_grad():
                x = torch.from_numpy(left_pad([[iidx_plus(i) for i in seqs[u]]], max_len)).to(device)
                cache[u] = model.scores(x)[0].float().cpu().numpy()
        return cache[u][cols]

    def iidx_plus(item):
        return iidx[item] + 1

    return scorer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = proc_dir(args.dataset)
    inter = pd.read_parquet(out / "interactions.parquet").sort_values(["user_id", "ts"], kind="stable")
    items = pd.read_parquet(out / "items.parquet")
    iidx = {it: i for i, it in enumerate(items["item_id"])}
    full = inter.groupby("user_id")["item_id"].apply(list).to_dict()
    test_path = args.eval_set or default_split(out, "test")
    valid_path = test_path.with_name(test_path.name.replace("test_", "valid_", 1))
    valid, test = read_jsonl(valid_path), read_jsonl(test_path)
    max_len = MAX_LEN.get(args.dataset, DEFAULT_MAX_LEN)

    def user_seq(drop_last: int) -> dict:
        return {u: s[:-drop_last] for u, s in full.items()}

    train_seqs = [[iidx[i] + 1 for i in s[:-2]] for s in full.values() if len(s) >= 4]
    print(f"SASRec {args.dataset}: {len(train_seqs):,} training sequences, {len(items):,} items, "
          f"max_len {max_len}, device {args.device}", flush=True)
    results = []
    best_model, best_info = None, None
    for cfg in GRID:
        model, info = train_config(cfg, train_seqs, valid, user_seq, iidx, len(items), max_len, args.device,
                                   args.epochs, args.seed)
        results.append(info)
        if best_info is None or info["valid_ndcg@10"] > best_info["valid_ndcg@10"]:
            best_model, best_info = model, info

    base = RESULTS / "raw" / args.dataset / test_path.stem
    scorer = make_scorer(best_model, user_seq(1), iidx, max_len, args.device)
    write_jsonl(base / "sasrec" / "idonly.jsonl", to_records(test, scorer, iidx, "sasrec", test_path.name))
    (base / "sasrec" / "tuning.json").write_text(json.dumps({"best": best_info["config"], "runs": results}, indent=2))

    hits, ndcgs = [], []
    for row in test:
        s = scorer(row["user_id"], np.arange(len(items))).astype(np.float64)
        seen = [iidx[i] for i in full[row["user_id"]][:-1]]
        s[seen] = -np.inf
        rank = 1 + int((s > s[iidx[row["target"]]]).sum())
        hits.append(rank <= 10)
        ndcgs.append(1 / math.log2(rank + 1) if rank <= 10 else 0.0)
    fr_path = base / "fullrank.json"
    fr = json.loads(fr_path.read_text()) if fr_path.exists() else {}
    fr["sasrec"] = {"hr@10": float(np.mean(hits)), "ndcg@10": float(np.mean(ndcgs)), "n_users": len(test)}
    fr_path.write_text(json.dumps(fr, indent=2))
    test_ndcg = mean_ndcg10(score_rows(test, scorer, iidx))
    print(f"SASRec {args.dataset}: best {best_info['config']} (epoch {best_info['best_epoch']}, valid NDCG@10 "
          f"{best_info['valid_ndcg@10']:.4f}); test NDCG@10 {test_ndcg:.4f}; full-catalogue {fr['sasrec']}")


if __name__ == "__main__":
    main()
