"""Create AI-written reviews with open LLMs, matched to the real reviews' categories, star ratings and lengths.

    python scripts/02_generate_ai_reviews.py --real data/real_reviews.parquet --model Qwen/Qwen2.5-1.5B-Instruct \
        --n 6000 --device cuda:0 --out data/ai_qwen.parquet

Matching rating and length distributions matters: otherwise a classifier can "detect AI" just by
looking at length or star rating, which would not work on real fake reviews.

Prompt modes (chosen at random per review):
  product  - the model sees a real customer's review of a product and writes a *different* customer's review
  category - the model only knows the category and invents a plausible product
  rewrite  - the model rewrites a real review in its own words (kept in a separate analysis set)
"""
import argparse
import random
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PERSONAS = ["a busy parent", "a college student", "a retired teacher", "a tech enthusiast", "a small business owner",
            "someone buying a gift", "a first-time buyer", "a frequent online shopper", "a nurse working night shifts",
            "a fitness enthusiast", "someone on a tight budget", "a picky customer"]
STYLES = ["casual and short sentences", "detailed and specific", "a bit informal with minor typos",
          "enthusiastic", "matter-of-fact", "slightly sarcastic", "comparing it with a previous product"]
STAR_WORDS = {1: "very negative (1 star)", 2: "negative (2 stars)", 3: "mixed (3 stars)", 4: "positive (4 stars)",
              5: "very positive (5 stars)"}


def make_prompt(row, mode, rng):
    words = max(15, int(len(row.text.split()) * rng.uniform(0.8, 1.2)))
    cat = row.category.replace("_", " ").lower()
    common = (f"Write it as {rng.choice(PERSONAS)}, {rng.choice(STYLES)}. About {words} words. "
              "Output only the review text: no title, no star symbols, no preamble.")
    if mode == "product":
        return (f"Here is one customer's review of a product in {cat}:\n\"\"\"{row.text[:1200]}\"\"\"\n\n"
                f"Write a different customer's {STAR_WORDS[row.rating]} review of the same product. {common}")
    if mode == "rewrite":
        return (f"Rewrite this product review in your own words, keeping its meaning and rating:\n"
                f"\"\"\"{row.text[:1200]}\"\"\"\n\nOutput only the rewritten review.")
    return f"Write a {STAR_WORDS[row.rating]} Amazon customer review for a product in the {cat} category. {common}"


_PREFIX = re.compile(r"^\s*(sure[,!.]?|here('s| is)[^\n:]*:|review:|\*\*review\*\*:?|certainly[,!.]?)\s*", re.I)


def clean(text):
    t = text.strip().strip('"').strip()
    for _ in range(2):
        t = _PREFIX.sub("", t).strip()
    t = re.sub(r"^(title|rating)\s*:[^\n]*\n", "", t, flags=re.I).strip()
    t = re.sub(r"[★☆]+", "", t).strip().strip('"').strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--rewrite-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    real = pd.read_parquet(args.real).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    seeds = real.iloc[: args.n]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    jobs = []
    for row in seeds.itertuples():
        r = rng.random()
        mode = "rewrite" if r < args.rewrite_frac else ("product" if r < 0.6 else "category")
        jobs.append((row, mode, make_prompt(row, mode, rng)))

    out = []
    for i in range(0, len(jobs), args.batch_size):
        batch = jobs[i:i + args.batch_size]
        texts = []
        for _, _, p in batch:
            msgs = [{"role": "user", "content": p}]
            try:
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            except Exception:  # base model without a chat template
                texts.append(p + "\n\n")
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(args.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=rng.uniform(0.7, 1.0), top_p=0.95,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
        decoded = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (row, mode, _), d in zip(batch, decoded):
            t = clean(d)
            if len(t.split()) < 8:
                continue
            out.append({"text": t, "title": "", "rating": row.rating, "category": row.category,
                        "parent_asin": row.parent_asin if mode != "category" else f"gen-{row.category}-{len(out)}",
                        "verified": False, "helpful": 0, "source": f"llm:{mode}", "label": 1,
                        "generator": args.model.split("/")[-1], "mode": mode})
        print(f"{min(i + args.batch_size, len(jobs))}/{len(jobs)} prompts, {len(out)} kept", flush=True)

    pd.DataFrame(out).to_parquet(args.out, index=False)
    print(f"saved {len(out):,} AI-written reviews -> {args.out}")


if __name__ == "__main__":
    main()
