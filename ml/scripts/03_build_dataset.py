"""Combine real + AI reviews into leak-free splits and report shortcut baselines.

    python scripts/03_build_dataset.py --real data/real_reviews.parquet \
        --ai data/ai_qwen.parquet data/ai_smol.parquet data/ai_phi.parquet --heldout-generator Phi-3.5-mini-instruct \
        --external fake_reviews_dataset.csv --out data/splits

Outputs (parquet): train, val, test (same generators, unseen products), test_unseen_generator (an LLM never
seen in training), test_external (2022 public CG-vs-original dataset), rewrites (real reviews paraphrased by an LLM).
Also writes data/splits/report.json with class balance and a length-only baseline.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def words(s):
    return s.str.split().str.len()


def length_match(df, seed, bin_width=10):
    """Downsample so real and AI reviews have the same length histogram (removes the length shortcut)."""
    df = df.assign(_bin=(words(df.text) // bin_width).clip(upper=40))
    keep = []
    for _, g in df.groupby("_bin"):
        n = g.label.value_counts()
        if len(n) < 2:
            continue
        k = n.min()
        keep.append(g[g.label == 0].sample(k, random_state=seed))
        keep.append(g[g.label == 1].sample(k, random_state=seed))
    return pd.concat(keep).drop(columns="_bin").sample(frac=1.0, random_state=seed)


def length_baseline_auc(train, test):
    f = lambda d: np.c_[np.log1p(words(d.text)), d.text.str.count(r"[!?]") / (words(d.text) + 1)]
    clf = LogisticRegression().fit(f(train), train.label)
    return round(float(roc_auc_score(test.label, clf.predict_proba(f(test))[:, 1])), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--ai", nargs="+", required=True)
    ap.add_argument("--heldout-generator", default=None, help="generator name kept out of training entirely")
    ap.add_argument("--external", default=None, help="CSV with columns label (CG/OR) and text_ (Salminen et al. 2022)")
    ap.add_argument("--external-n", type=int, default=4000)
    ap.add_argument("--no-length-match", action="store_true")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cols = ["text", "rating", "category", "parent_asin", "label", "generator", "source"]

    real = pd.read_parquet(args.real)
    ai = pd.concat([pd.read_parquet(p) for p in args.ai], ignore_index=True)
    ai["mode"] = ai.get("mode", "category")
    rewrites = ai[ai["mode"] == "rewrite"]
    ai = ai[ai["mode"] != "rewrite"]
    held = ai[ai.generator == args.heldout_generator] if args.heldout_generator else ai.iloc[:0]
    ai = ai[ai.generator != args.heldout_generator] if args.heldout_generator else ai

    data = pd.concat([real[cols], ai[cols]], ignore_index=True).drop_duplicates("text")
    data["parent_asin"] = data.parent_asin.fillna("na").astype(str)

    # Split by product, so the same product never appears in both train and test.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, rest_idx = next(gss.split(data, groups=data.parent_asin))
    train, rest = data.iloc[tr_idx], data.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    va_idx, te_idx = next(gss2.split(rest, groups=rest.parent_asin))
    val, test = rest.iloc[va_idx], rest.iloc[te_idx]

    report = {"length_baseline_auc_before_matching": length_baseline_auc(train, val)}
    if not args.no_length_match:
        train, val, test = (length_match(d, args.seed) for d in (train, val, test))
    report["length_baseline_auc_after_matching"] = length_baseline_auc(train, val)

    splits = {"train": train, "val": val, "test": test}
    if len(held):
        n = min(len(held), (test.label == 0).sum())
        splits["test_unseen_generator"] = pd.concat([held[cols].sample(n, random_state=args.seed),
                                                     test[test.label == 0].sample(n, random_state=args.seed)])
    if args.external and os.path.exists(args.external):
        ext = pd.read_csv(args.external)
        ext = pd.DataFrame({"text": ext["text_"].astype(str), "label": (ext["label"] == "CG").astype(int),
                            "rating": ext["rating"], "category": ext["category"], "parent_asin": "external",
                            "generator": np.where(ext["label"] == "CG", "gpt2-2022", "human"), "source": "salminen2022"})
        k = min(args.external_n // 2, ext.label.value_counts().min())
        splits["test_external"] = pd.concat([ext[ext.label == c].sample(k, random_state=args.seed) for c in (0, 1)])
    if len(rewrites):
        splits["rewrites"] = rewrites[cols]

    for name, d in splits.items():
        d = d.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        d.to_parquet(os.path.join(args.out, f"{name}.parquet"), index=False)
        report[name] = {"n": len(d), "ai": int(d.label.sum()), "real": int((d.label == 0).sum()),
                        "median_words_real": float(words(d[d.label == 0].text).median()) if (d.label == 0).any() else None,
                        "median_words_ai": float(words(d[d.label == 1].text).median()) if (d.label == 1).any() else None}
        print(f"{name:>22}: {report[name]}")
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("length-only baseline AUC (want ~0.5 after matching):",
          report["length_baseline_auc_before_matching"], "->", report["length_baseline_auc_after_matching"])


if __name__ == "__main__":
    main()
