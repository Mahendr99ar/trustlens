"""Builds tiny random models + a stand-in review file so the whole ML pipeline can be tested offline in minutes.
Nothing here is used for real training."""
import os
import sys

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizerFast, GPT2Config, GPT2LMHeadModel)

out, csv = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.read_csv(csv)
texts = df.text_.astype(str).tolist()

tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tk.normalizer = normalizers.BertNormalizer(lowercase=True)
tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tk.train_from_iterator(texts[:20000], trainers.WordPieceTrainer(vocab_size=3000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                  special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
fast = BertTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                         sep_token="[SEP]", mask_token="[MASK]", model_max_length=512)

cfg = dict(vocab_size=3000, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
           max_position_embeddings=512)
for name, labels in [("tiny-bert", ["LABEL_0", "LABEL_1"]), ("tiny-absa-teacher", ["Negative", "Neutral", "Positive"])]:
    m = BertForSequenceClassification(BertConfig(**cfg, num_labels=len(labels), id2label=dict(enumerate(labels)),
                                                 label2id={l: i for i, l in enumerate(labels)}))
    m.save_pretrained(f"{out}/{name}")
    fast.save_pretrained(f"{out}/{name}")

gen_tok = BertTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[SEP]",
                            bos_token="[CLS]", model_max_length=512)
GPT2LMHeadModel(GPT2Config(vocab_size=3000, n_embd=64, n_layer=2, n_head=2, n_positions=1024,
                           eos_token_id=3, bos_token_id=2, pad_token_id=0)).save_pretrained(f"{out}/tiny-gen")
gen_tok.save_pretrained(f"{out}/tiny-gen")

orig = df[df.label == "OR"].sample(3000, random_state=0)
rng = np.random.RandomState(0)
real = pd.DataFrame({"text": orig.text_.values, "title": "", "rating": orig.rating.round().astype(int).values,
                     "category": orig.category.str.replace("_5", "").values,
                     "parent_asin": [f"P{rng.randint(0, 400)}" for _ in range(len(orig))], "verified": True,
                     "helpful": 0, "source": "standin", "label": 0, "generator": "human"})
real.to_parquet(f"{out}/real_reviews.parquet", index=False)
cg = df[df.label == "CG"].sample(1500, random_state=0)
for i, gen in enumerate(["gen-a", "gen-b", "gen-held"]):
    part = cg.iloc[i * 500:(i + 1) * 500]
    pd.DataFrame({"text": part.text_.values, "title": "", "rating": part.rating.round().astype(int).values,
                  "category": part.category.str.replace("_5", "").values,
                  "parent_asin": [f"P{rng.randint(0, 400)}" for _ in range(len(part))], "verified": False,
                  "helpful": 0, "source": "llm:product", "label": 1, "generator": gen,
                  "mode": ["rewrite" if j % 10 == 0 else "product" for j in range(len(part))]}
                 ).to_parquet(f"{out}/ai_{gen}.parquet", index=False)
print("tiny models + stand-in data written to", out)
