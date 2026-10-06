"""Aspect sentiment model by knowledge distillation: a large ABSA teacher labels review sentences, a small
student learns to copy it so it can run inside a browser.

    python scripts/05_train_aspect_model.py --reviews data/real_reviews.parquet \
        --teacher yangheng/deberta-v3-base-absa-v1.1 --student microsoft/MiniLM-L12-H384-uncased --out models/aspects

Student input format: "<aspect>: <sentence>"  ->  negative / neutral / positive.
Also writes human_eval_sample.csv: label its `human` column yourself (neg/neu/pos) and re-run with
--human-labels to measure teacher and student against people, not just against each other.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustlens_ml.aspects import find_aspect_terms, split_sentences  # noqa: E402

LABELS = ["negative", "neutral", "positive"]


def build_pairs(reviews, max_pairs, per_aspect_cap, seed):
    rng = random.Random(seed)
    rows, counts = [], {}
    for r in reviews.sample(frac=1.0, random_state=seed).itertuples():
        for s in split_sentences(r.text):
            for aspect, term in find_aspect_terms(s):
                if counts.get(aspect, 0) >= per_aspect_cap:
                    continue
                counts[aspect] = counts.get(aspect, 0) + 1
                rows.append({"review_id": r.Index, "aspect": aspect, "term": term, "sentence": s,
                             "student_input": f"{aspect.replace('_', ' ')}: {s}", "rating": r.rating})
        if len(rows) >= max_pairs:
            break
    rng.shuffle(rows)
    return pd.DataFrame(rows)


@torch.no_grad()
def teacher_label(pairs, name, device, bs=64):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    names = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    order = [names.index(l) if l in names else None for l in LABELS]
    if None in order:
        raise SystemExit(f"teacher labels {names} do not include {LABELS}")
    probs = []
    for i in range(0, len(pairs), bs):
        sub = pairs.iloc[i:i + bs]
        enc = tok(list(sub.sentence), list(sub.term), truncation=True, max_length=160, padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            p = torch.softmax(model(**enc).logits.float(), -1)[:, order]
        probs.append(p.cpu().numpy())
        if i % (bs * 50) == 0:
            print(f"teacher {i}/{len(pairs)}", flush=True)
    return np.concatenate(probs)


def train_student(train, val, base, out, epochs, bs, lr, device):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
        ignore_mismatched_sizes=True).to(device)
    steps = epochs * math.ceil(len(train) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, hist = -1, []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = np.random.RandomState(epoch).permutation(len(train))
        for i in range(0, len(train), bs):
            sub = train.iloc[perm[i:i + bs]]
            enc = tok(list(sub.student_input), truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
            soft = torch.tensor(np.stack(sub.teacher_probs.values), dtype=torch.float32, device=device)
            with torch.autocast("cuda", enabled=device == "cuda"):
                logits = model(**enc).logits.float()
            loss = -(soft * torch.log_softmax(logits, -1)).sum(-1).mean()  # distil the teacher's soft labels
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if (i // bs) % 100 == 0:
                print(f"epoch {epoch} step {i // bs} loss {loss.item():.4f}", flush=True)
        pred = predict(model, tok, val.student_input.tolist(), device)
        m = {"epoch": epoch, "agreement_with_teacher": round(accuracy_score(val.teacher_label, pred), 4),
             "macro_f1_vs_teacher": round(f1_score(val.teacher_label, pred, average="macro"), 4),
             "epoch_sec": round(time.time() - t0, 1)}
        hist.append(m)
        print("val:", m, flush=True)
        if m["macro_f1_vs_teacher"] > best:
            best = m["macro_f1_vs_teacher"]
            model.save_pretrained(out)
            tok.save_pretrained(out)
    return hist


@torch.no_grad()
def predict(model, tok, texts, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        out.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--teacher", default="yangheng/deberta-v3-base-absa-v1.1")
    ap.add_argument("--student", default="microsoft/MiniLM-L12-H384-uncased")
    ap.add_argument("--max-pairs", type=int, default=60000)
    ap.add_argument("--per-aspect-cap", type=int, default=6000)
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--human-labels", default=None, help="human_eval_sample.csv with a filled 'human' column")
    ap.add_argument("--out", default="models/aspects")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    reviews = pd.read_parquet(args.reviews)
    pairs = build_pairs(reviews, args.max_pairs, args.per_aspect_cap, seed=0)
    print(f"{len(pairs):,} (aspect, sentence) pairs; per aspect: {pairs.aspect.value_counts().to_dict()}")
    probs = teacher_label(pairs, args.teacher, device)
    pairs["teacher_probs"] = list(probs)
    pairs["teacher_label"] = probs.argmax(1)
    pairs["teacher_conf"] = probs.max(1)
    confident = pairs[pairs.teacher_conf >= args.min_confidence]
    print(f"kept {len(confident):,} confident teacher labels; distribution "
          f"{confident.teacher_label.map(dict(enumerate(LABELS))).value_counts().to_dict()}")

    # Split by review so sentences of one review never sit on both sides
    ids = confident.review_id.unique()
    rng = np.random.RandomState(0)
    val_ids = set(rng.choice(ids, size=max(1, len(ids) // 10), replace=False))
    val = confident[confident.review_id.isin(val_ids)]
    train = confident[~confident.review_id.isin(val_ids)]
    hist = train_student(train, val, args.student, args.out, args.epochs, args.batch_size, args.lr, device)

    results = {"teacher": args.teacher, "student": args.student, "pairs": len(pairs), "confident_pairs": len(confident),
               "history": hist}
    pool = pairs[~pairs.review_id.isin(train.review_id)]
    sample = pool.sample(min(300, len(pool)), random_state=1)
    sample[["aspect", "sentence"]].assign(human="").to_csv(os.path.join(args.out, "human_eval_sample.csv"), index=False)

    if args.human_labels and os.path.exists(args.human_labels):
        h = pd.read_csv(args.human_labels).dropna(subset=["human"])
        h = h[h.human.astype(str).str.strip().str[:3].isin(["neg", "neu", "pos"])]
        y = h.human.str.strip().str[:3].map({"neg": 0, "neu": 1, "pos": 2}).values
        stok = AutoTokenizer.from_pretrained(args.out)
        smodel = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
        sp = predict(smodel, stok, [f"{a.replace('_', ' ')}: {s}" for a, s in zip(h.aspect, h.sentence)], device)
        tp = teacher_label(h.assign(term=[(find_aspect_terms(s) or [(a, a)])[0][1] for a, s in zip(h.aspect, h.sentence)]),
                           args.teacher, device).argmax(1)
        results["human_eval"] = {"n": int(len(h)), "student_acc": round(accuracy_score(y, sp), 4),
                                 "student_macro_f1": round(f1_score(y, sp, average="macro"), 4),
                                 "teacher_acc": round(accuracy_score(y, tp), 4),
                                 "teacher_macro_f1": round(f1_score(y, tp, average="macro"), 4)}
        print("human eval:", results["human_eval"])

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "trustlens.json"), "w") as f:
        json.dump({"task": "aspect_sentiment", "input_format": "<aspect>: <sentence>", "labels": LABELS}, f, indent=2)
    print(f"saved student + metrics -> {args.out}")


if __name__ == "__main__":
    main()
