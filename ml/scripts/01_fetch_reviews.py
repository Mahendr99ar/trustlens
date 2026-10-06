"""Stream real reviews from Amazon Reviews 2023 (McAuley Lab, UCSD) without downloading whole categories.

    python scripts/01_fetch_reviews.py --out data/real_reviews.parquet --per-category 8000

Reads the gzipped JSONL over HTTP and stops once it has enough reviews per category, keeping a
balanced mix of star ratings and only reviews with 15-400 words.
"""
import argparse
import gzip
import io
import json
import random

import pandas as pd
import requests

URLS = [
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{cat}.jsonl.gz",
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{cat}.jsonl",
]
DEFAULT_CATS = ["Electronics", "Cell_Phones_and_Accessories", "Home_and_Kitchen", "Beauty_and_Personal_Care",
                "Clothing_Shoes_and_Jewelry", "Toys_and_Games", "Sports_and_Outdoors", "Health_and_Household"]


def stream_lines(cat):
    for tmpl in URLS:
        url = tmpl.format(cat=cat)
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"  {url} failed: {e}")
            continue
        print(f"  streaming {url}")
        raw = r.raw
        raw.decode_content = True
        fh = gzip.GzipFile(fileobj=raw) if url.endswith(".gz") else raw
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="ignore"):
            yield line
        return
    raise SystemExit(f"could not download category {cat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real_reviews.parquet")
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATS)
    ap.add_argument("--per-category", type=int, default=8000)
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--max-scan", type=int, default=400000, help="stop scanning a category after this many lines")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    rows = []
    per_star = args.per_category // 5
    for cat in args.categories:
        print(cat)
        buckets = {s: [] for s in range(1, 6)}
        for i, line in enumerate(stream_lines(cat)):
            if i >= args.max_scan or all(len(b) >= per_star for b in buckets.values()):
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            star = int(round(float(r.get("rating", 0))))
            text = (r.get("text") or "").replace("<br />", "\n").strip()
            n = len(text.split())
            if star not in buckets or len(buckets[star]) >= per_star or not (args.min_words <= n <= args.max_words):
                continue
            buckets[star].append({"text": text, "title": r.get("title", ""), "rating": star, "category": cat,
                                  "parent_asin": r.get("parent_asin") or r.get("asin"),
                                  "verified": bool(r.get("verified_purchase")), "helpful": int(r.get("helpful_vote", 0)),
                                  "source": "amazon2023", "label": 0, "generator": "human"})
        got = sum(len(b) for b in buckets.values())
        print(f"  kept {got} reviews " + str({s: len(b) for s, b in buckets.items()}))
        for b in buckets.values():
            rows.extend(b)

    df = pd.DataFrame(rows).drop_duplicates("text")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"saved {len(df):,} real reviews from {df.parent_asin.nunique():,} products -> {args.out}")


if __name__ == "__main__":
    main()
