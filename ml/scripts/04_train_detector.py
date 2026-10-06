"""Fine-tune a transformer to tell AI-written reviews from real ones.

    python scripts/04_train_detector.py --data data/splits --base microsoft/MiniLM-L12-H384-uncased \
        --epochs 3 --out models/detector

Picks the decision threshold on the validation set so that at most --max-fpr of real reviews are flagged
(wrongly accusing real customers is the costly error), then reports every test split at that threshold.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def batches(df, tok, bs, max_len, shuffle, seed=0):
    idx = np.random.RandomState(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), bs):
        sub = df.iloc[idx[i:i + bs]]
        enc = tok(list(sub.text), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield enc, torch.tensor(sub.label.values)


@torch.no_grad()
def predict(model, tok, df, device, bs=64, max_len=256):
    model.eval()
    probs = []
    for enc, _ in batches(df, tok, bs, max_len, False):
        with torch.autocast("cuda", enabled=device.startswith("cuda")):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def threshold_for_fpr(y, p, max_fpr):
    real = np.sort(p[y == 0])
    if len(real) == 0:
        return 0.5
    k = int(math.floor(len(real) * (1 - max_fpr)))
    return float(real[min(k, len(real) - 1)]) + 1e-6


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    out = {"n": int(len(y)), "threshold": round(thr, 4),
           "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
           "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
           "recall_ai": round(float(recall_score(y, pred, zero_division=0)), 4),
           "false_alarm_rate_real": round(float(pred[y == 0].mean()), 4) if (y == 0).any() else None}
    if len(set(y)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/splits")
    ap.add_argument("--base", default="microsoft/MiniLM-L12-H384-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training set (smoke tests)")
    ap.add_argument("--out", default="models/detector")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train = pd.read_parquet(f"{args.data}/train.parquet")
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=0)
    val = pd.read_parquet(f"{args.data}/val.parquet")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "real", 1: "ai"}, label2id={"real": 0, "ai": 1},
        ignore_mismatched_sizes=True).to(device)

    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history, best_auc = [], -1
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for i, (enc, y) in enumerate(batches(train, tok, args.batch_size, args.max_len, True, seed=epoch)):
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = model(**{k: v.to(device) for k, v in enc.items()}, labels=y.to(device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if i % 100 == 0:
                print(f"epoch {epoch} step {i} loss {loss.item():.4f}", flush=True)
        p = predict(model, tok, val, device, max_len=args.max_len)
        m = metrics(val.label.values, p, threshold_for_fpr(val.label.values, p, args.max_fpr))
        m.update(epoch=epoch, train_loss=round(total / (i + 1), 4), epoch_sec=round(time.time() - t0, 1))
        history.append(m)
        print("val:", m, flush=True)
        if m.get("roc_auc", 0) > best_auc:
            best_auc = m.get("roc_auc", 0)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)

    # Final evaluation with the best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    pv = predict(model, tok, val, device, max_len=args.max_len)
    thr = threshold_for_fpr(val.label.values, pv, args.max_fpr)
    results = {"base_model": args.base, "max_fpr_target": args.max_fpr, "threshold": round(thr, 4),
               "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "history": history}
    for path in sorted(glob.glob(f"{args.data}/*.parquet")):
        name = os.path.basename(path)[:-8]
        if name == "train":
            continue
        d = pd.read_parquet(path)
        p = predict(model, tok, d, device, max_len=args.max_len)
        results[name] = metrics(d.label.values, p, thr)
        if name == "rewrites":
            results[name] = {"n": len(d), "flagged_as_ai": round(float((p >= thr).mean()), 4),
                             "note": "real reviews paraphrased by an LLM; no single right answer"}
        print(f"{name:>22}: {results[name]}")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "trustlens.json"), "w") as f:
        json.dump({"task": "ai_review_detector", "threshold": round(thr, 4), "max_len": args.max_len,
                   "positive_label": "ai"}, f, indent=2)
    print(f"saved best model + metrics -> {args.out}")


if __name__ == "__main__":
    main()
