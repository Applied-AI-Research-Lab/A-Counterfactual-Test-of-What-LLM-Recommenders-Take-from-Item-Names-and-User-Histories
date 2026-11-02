from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATA_RAW, download, proc_dir

MIN_USER_LEN = 7
MIN_ITEM_COUNT = 5
DEFAULT_MAX_USERS = {"amazon-beauty": 60_000, "amazon-grocery": 60_000}
MAX_DENSE_ITEMS = 35_000

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
AMAZON_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
AMAZON_CATEGORIES = {
    "amazon-videogames": "Video_Games",
    "amazon-beauty": "Beauty_and_Personal_Care",
    "amazon-grocery": "Grocery_and_Gourmet_Food",
}
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

AMAZON_ATTRIBUTE_KEYS = [
    "Type of item", "Rated", "Hardware Platform", "Compatible Devices", "Connectivity Technology",
    "Form Factor", "Special Feature", "Item Form", "Hair Type", "Skin Type", "Scent",
    "Product Benefits", "Finish Type", "Material", "Color", "Age Range (Description)",
    "Flavor", "Specialty", "Diet Type", "Caffeine Content", "Roast Level", "Tea Variety",
    "Package Information",
]
MAX_ATTRIBUTE_VALUE = 40


def clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_ml1m() -> tuple[pd.DataFrame, pd.DataFrame]:
    zpath = download(ML1M_URL, DATA_RAW / "ml-1m.zip")
    with zipfile.ZipFile(zpath) as zf:
        ratings = pd.read_csv(
            io.BytesIO(zf.read("ml-1m/ratings.dat")), sep="::", engine="python",
            names=["user_id", "item_id", "rating", "ts"], encoding="latin-1",
        )
        movies = pd.read_csv(
            io.BytesIO(zf.read("ml-1m/movies.dat")), sep="::", engine="python",
            names=["item_id", "raw_title", "genres"], encoding="latin-1",
        )

    def split_title(raw: str) -> tuple[str, int | None]:
        match = re.match(r"^(.*)\s+\((\d{4})\)\s*$", raw)
        title, year = (match.group(1), int(match.group(2))) if match else (raw, None)
        art = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|El|Das|Der|Die)(\s*\(.*\))?$", title)
        if art:
            title = f"{art.group(2)} {art.group(1)}{art.group(3) or ''}"
        return clean_title(title), year

    parsed = movies["raw_title"].map(split_title)
    years = [y for _, y in parsed]
    items = pd.DataFrame({
        "item_id": movies["item_id"].astype(str),
        "title": [t for t, _ in parsed],
        "category": movies["genres"].str.split("|").str[0],
        "year": pd.array(years, dtype="Int64"),
        "attributes": [
            (f"{y // 10 * 10}s film; " if y else "film; ") + "genres: " + g.replace("|", ", ")
            for y, g in zip(years, movies["genres"])
        ],
    })
    inter = ratings.assign(user_id=ratings["user_id"].astype(str), item_id=ratings["item_id"].astype(str))
    return inter[["user_id", "item_id", "ts"]], items


def load_amazon(dataset: str, max_users: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = AMAZON_CATEGORIES[dataset]
    parts = []
    for split in ("train", "valid", "test"):
        path = download(f"{AMAZON_BASE}/benchmark/5core/last_out/{cat}.{split}.csv",
                        DATA_RAW / "amazon" / f"{cat}.{split}.csv")
        parts.append(pd.read_csv(path, usecols=["user_id", "parent_asin", "timestamp"]))
    inter = pd.concat(parts).rename(columns={"parent_asin": "item_id", "timestamp": "ts"})
    if max_users:
        counts = inter.groupby("user_id").size()
        eligible = np.sort(counts[counts >= MIN_USER_LEN].index.to_numpy())
        if len(eligible) > max_users:
            keep = np.random.default_rng(seed).choice(eligible, size=max_users, replace=False)
            inter = inter[inter["user_id"].isin(set(keep))]
        print(f"{dataset}: sampled {inter['user_id'].nunique():,} of {len(eligible):,} eligible users", flush=True)

    suffix = f"_u{max_users}_s{seed}" if max_users else ""
    meta_path = DATA_RAW / "amazon" / f"meta_{cat}.filtered{suffix}.jsonl"
    if not meta_path.exists():
        stream_amazon_meta(f"{AMAZON_BASE}/raw/meta_categories/meta_{cat}.jsonl",
                           set(inter["item_id"].unique()), meta_path)
    rows = []
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            cats = m.get("categories") or []
            details = m.get("details") or {}
            date_text = " ".join(str(details.get(k, "")) for k in ("Release date", "Date First Available"))
            year = YEAR_RE.search(date_text)
            year = int(year.group(1)) if year else None
            rows.append({
                "item_id": m["parent_asin"],
                "title": clean_title(m.get("title") or ""),
                "category": cats[1] if len(cats) >= 2 else (m.get("main_category") or "Other"),
                "year": year,
                "attributes": amazon_attributes(cats, details, year, m.get("main_category")),
            })
    items = pd.DataFrame(rows)
    items["year"] = items["year"].astype("Int64")
    return inter, items


def amazon_attributes(cats: list, details: dict, year: int | None, main_category: str | None) -> str:
    identity = [str(details.get(k, "")).lower() for k in ("Brand", "Manufacturer")]
    identity = [s for s in identity if len(s) >= 3]
    path = " > ".join(cats[2:]) if len(cats) > 2 else (cats[-1] if cats else (main_category or "product"))
    parts = [path]
    for key in AMAZON_ATTRIBUTE_KEYS:
        value = clean_title(details.get(key) or "")
        if not value or any(s in value.lower() for s in identity):
            continue
        parts.append(f"{key.lower()}: {value[:MAX_ATTRIBUTE_VALUE]}")
    if year:
        parts.append(f"released {year}")
    return "; ".join(parts)


def stream_amazon_meta(url: str, keep: set[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    asin_re = re.compile(r'"parent_asin":\s*"([^"]+)"')
    kept = 0
    print(f"Streaming metadata {url} (keeping {len(keep):,} items)", flush=True)
    with urllib.request.urlopen(url, timeout=300) as resp, open(out.with_suffix(".part"), "w", encoding="utf-8") as fh:
        for raw in resp:
            line = raw.decode("utf-8")
            match = asin_re.search(line)
            if match and match.group(1) in keep:
                fh.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    out.with_suffix(".part").rename(out)
    print(f"  kept {kept:,} metadata rows", flush=True)


def finalize(inter: pd.DataFrame, items: pd.DataFrame, item_core: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = items[items["title"].str.len() > 0].drop_duplicates("item_id")
    inter = inter[inter["item_id"].isin(items["item_id"])]
    inter = (inter.sort_values(["user_id", "ts"], kind="stable")
                  .drop_duplicates(["user_id", "item_id"], keep="first"))
    while True:
        n = len(inter)
        if item_core:
            inter = inter[inter.groupby("item_id")["user_id"].transform("size") >= item_core]
        inter = inter[inter.groupby("user_id")["item_id"].transform("size") >= MIN_USER_LEN]
        if len(inter) == n:
            break
    inter = inter.reset_index(drop=True)
    pop = inter["item_id"].value_counts()
    items = items[items["item_id"].isin(pop.index)].copy()
    items["popularity"] = items["item_id"].map(pop).astype(int)
    return inter, items.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-users", type=int, default=None,
                    help=f"user subsample size (defaults: {DEFAULT_MAX_USERS}; 0 = no subsampling)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = proc_dir(args.dataset)
    max_users = DEFAULT_MAX_USERS.get(args.dataset) if args.max_users is None else (args.max_users or None)
    if args.dataset == "ml-1m":
        inter, items = load_ml1m()
    else:
        inter, items = load_amazon(args.dataset, max_users, args.seed)

    inter, items = finalize(inter, items, item_core=MIN_ITEM_COUNT if max_users else 0)
    inter.to_parquet(out / "interactions.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    meta = {"dataset": args.dataset, "max_users": max_users, "seed": args.seed,
            "item_core": MIN_ITEM_COUNT if max_users else None, "min_user_len": MIN_USER_LEN,
            "users": int(inter["user_id"].nunique()), "items": len(items), "interactions": len(inter)}
    (out / "prepare_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{args.dataset}: {meta['users']:,} users, {meta['items']:,} items, "
          f"{meta['interactions']:,} interactions, {items['category'].nunique()} categories -> {out}")
    if len(items) > MAX_DENSE_ITEMS:
        print(f"WARNING: {len(items):,} items exceed {MAX_DENSE_ITEMS:,}; dense EASE will be slow and memory-hungry. "
              f"Use a smaller --max-users.")


if __name__ == "__main__":
    main()
