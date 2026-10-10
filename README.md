# TrustLens

*Read the reviews that were actually written by customers.*

A Chrome extension that spots AI-written Amazon reviews, recomputes the star rating without them, and tells you what the remaining reviewers say about battery, build, value and ten other aspects.

[![CI](https://github.com/Mahendr99ar/Trustlens/actions/workflows/ci.yml/badge.svg)](https://github.com/Mahendr99ar/Trustlens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Chrome MV3](https://img.shields.io/badge/Chrome-Manifest%20V3-4285F4.svg)](extension/src/manifest.json)
[![Models on Hugging Face](https://img.shields.io/badge/models-Hugging%20Face-FFD21E.svg)](https://huggingface.co/Mahendra99ar/trustlens-detector)
[![Runs in the browser](https://img.shields.io/badge/inference-in%20browser-2E7D32.svg)](extension/src/offscreen.js)

> **TrustLens does not tell you a review is fake.** It tells you a review reads like it came from a language model. A paid human reviewer passes straight through, and a real customer who polished their text with ChatGPT may get flagged. Treat the badge as a signal, not a verdict.

**Website:** [mahendr99ar.github.io/trustlens](https://mahendr99ar.github.io/trustlens/)
**Extension:** [latest release](https://github.com/Mahendr99ar/Trustlens/releases/latest)
**Models:** [trustlens-detector](https://huggingface.co/Mahendra99ar/trustlens-detector) | [trustlens-aspects](https://huggingface.co/Mahendra99ar/trustlens-aspects)

---

## What this is

Mozilla shut Fakespot down in July 2025. Around the same time, small open models became good enough to write a believable three-paragraph review in two seconds. TrustLens asks a narrow question: how well can a model small enough to run inside a browser tab spot them?

On an Amazon product page it adds a card above the reviews with three things:

1. A badge on every review: likely genuine, or reads AI-written.
2. A **trust-adjusted rating**, recomputed with the suspicious reviews counted less.
3. **What genuine reviewers say**, aspect by aspect: positive, negative, or mixed on quality, battery, comfort and so on.

Both models run on your machine. Review text never leaves the browser.

---

## Why not just send the reviews to a big model?

Because that means shipping every review you read to a server, paying for inference on every page view, and trusting whoever runs it. The alternative is a model that fits in a browser tab:

- **Privacy.** The content script reads the page, the offscreen document runs the models, and only numbers (a probability per review, a sentiment per aspect) are shared.
- **Cost.** No GPU server. The shared backend is a Cloudflare Worker on the free tier.
- **Speed.** About 44 ms per review on one CPU thread after the model loads.

The price is model size. Everything below is about making a 33 MB model good enough.

---

## How it works

### Stage 1: Build a dataset where the answer is known

`ml/scripts/01_fetch_reviews.py` pulls 47,896 real reviews from Amazon Reviews 2023 across 8 categories, balanced over 1 to 5 stars. `02_generate_ai_reviews.py` has Qwen2.5-1.5B, SmolLM2-1.7B and Phi-3.5-mini write 17,453 reviews for the same products, matched to the real ones on star rating and length (median 58 words in both classes). Length alone still gives 0.65 AUC, which is the floor the detector has to clear.

### Stage 2: Split so the test is honest

`03_build_dataset.py` splits **by product**, so no product appears in both train and test. Phi-3.5-mini is held out entirely, so the test also asks whether the detector learned "LLM writing" or just one model's habits.

### Stage 3: Train the detector

`04_train_detector.py` fine-tunes MiniLM-L12 on 22,670 balanced reviews. The decision threshold is picked on validation data so that at most 5% of real reviews get flagged. Accusing a real customer is the worse mistake, so the threshold leans that way.

### Stage 4: Distil the aspect model

`05_train_aspect_model.py` uses a DeBERTa-v3 aspect-sentiment model as a teacher. It labelled 60,001 (aspect, sentence) pairs over 13 aspects; the 56,264 it was confident about train a second MiniLM. The student agrees with the teacher on 88.0% of held-out sentences (macro F1 0.86).

### Stage 5: Shrink both for the browser

`06_export_onnx.py` exports to ONNX and quantizes to INT8: 128 MB down to 33 MB per model. A parity check compares the INT8 and full-precision detector on 500 test reviews; they agree on 97.8%. `07_push_to_hub.py` uploads both to Hugging Face, where Transformers.js loads them.

### Stage 6: Score the page

The content script finds elements with `data-hook="review"`, the offscreen document runs both models, and the card renders inside a shadow DOM so Amazon's CSS can't touch it. Scores go to the shared API so the next visitor sees a product report without re-running anything.

---

## Architecture

```
 Training (Colab / Kaggle GPU)                  In the user's browser                    Shared backend (free tier)
 -----------------------------                  ---------------------                    --------------------------
 Amazon Reviews 2023: 47,896 reviews,           content script finds reviews             Cloudflare Worker (TypeScript)
 8 categories, balanced 1-5 stars               on the product page                        POST /v1/install   token
 + 17,453 AI reviews from 3 open LLMs                 |                                     POST /v1/reviews   scores only
       | split by product, held-out LLM               v                                     GET  /v1/products/:asin/report
       v                                        offscreen document runs both              D1 (SQLite) + KV cache
 detector (MiniLM)    aspects (MiniLM,          INT8 models with Transformers.js          HMAC-signed install tokens,
       |              distilled from DeBERTa)         |                                   per-install rate limits
       +---> ONNX INT8 ---> Hugging Face Hub -------->| card above the reviews: trust-adjusted
                                                      | rating, badge on each review,
                                                      | what genuine reviewers say per aspect
```

**Trust-adjusted rating.** Each review counts with weight `1 - P(AI)`. The result is then pulled toward the product's plain average by 2 pseudo-reviews, so a product with four reviews doesn't swing two stars because one of them got flagged. The same formula lives in `ml/trustlens_ml/scoring.py` and `api/src/lib.ts`, and both are tested.

---

## Results

| Test set | ROC-AUC | F1 | AI reviews caught | Real reviews flagged |
|---|---|---|---|---|
| New products, same two LLMs (2,748) | 0.988 | 0.95 | 96% | 5.9% |
| New LLM, Phi-3.5-mini, never seen in training (2,700) | 0.999 | 0.97 | 99.8% | 5.8% |
| Salminen et al. 2022 fake reviews (4,000) | 0.584 | 0.31 | 21% | 13.2% |

Read the third row before the first two. Salminen et al. generated their fakes in 2022 with an older, smaller model, and on that set the detector is barely better than chance. TrustLens detects current instruction-tuned LLM writing. It is not a general fake-review detector.

When an LLM paraphrases a genuine review, the detector flags 86% of them. There is no single right label for those, so they are reported separately and kept out of the table.

---

## How it compares

| | TrustLens | Server-side LLM check | Star average as shown |
|---|---|---|---|
| Review text stays on your device | ✓ | ✗ | ✓ |
| Works without a paid server | ✓ | ✗ | ✓ |
| Down-weights suspicious reviews | ✓ | depends | ✗ |
| Per-aspect summary of genuine reviews | ✓ | depends | ✗ |
| Catches paid human reviews | ✗ | ✗ | ✗ |
| Model size | 33 MB per model | large | none |

---

## Install the extension

1. Download `trustlens-extension.zip` from the [latest release](https://github.com/Mahendr99ar/Trustlens/releases/latest) and unzip it.
2. Open `chrome://extensions` and turn on **Developer mode**.
3. Click **Load unpacked** and pick the unzipped folder.
4. Open any product on amazon.in, amazon.com, amazon.co.uk, amazon.ca, amazon.de, amazon.com.au, amazon.ae or amazon.sg and scroll to the reviews.

The first page downloads both models once (about 66 MB); after that they are cached.

## Build and run it yourself

```bash
# Train (Colab with one T4, about 4.5 hours, resumes after a disconnect)
# open notebooks/colab_trustlens.ipynb, add an HF_TOKEN secret, run the cells in order
# or notebooks/kaggle_trustlens.ipynb on two T4s

# API
cd api && npm install && npm test && npx wrangler dev

# Extension
cd extension && npm install && npm run build

# ML pipeline on CPU with tiny random models (no GPU, no downloads beyond the CSV)
cd ml && bash tests/test_offline_pipeline.sh fake_reviews.csv

# Load test against the local Worker
cd api && node scripts/loadtest.mjs http://127.0.0.1:8787
```

Deploying the API, the website and a release of the extension is written out step by step in [docs/GO_LIVE.md](docs/GO_LIVE.md). Every piece fits in a free tier, and none of the accounts need a card.

---

## API

| Method | Path | What it does | Limit per minute |
|---|---|---|---|
| GET | `/health` | `{"status":"ok"}` | |
| POST | `/v1/install` | Issues an HMAC-signed token for one extension install | 10 |
| POST | `/v1/reviews` | Stores scores for up to 50 reviews of one product. Text is rejected, only ids, stars, probabilities and aspect signs are accepted | 60 |
| GET | `/v1/products/:asin/report` | Trust-adjusted rating and aspect summary, cached in KV for 10 minutes | 120 |

---

## Repository structure

```
Trustlens/
├── ml/
│   ├── scripts/              01_fetch → 02_generate → 03_build → 04_train → 05_aspects → 06_onnx → 07_push
│   ├── trustlens_ml/         aspect matching and the rating maths
│   └── tests/                offline pipeline test with tiny random models
├── api/
│   ├── src/                  Cloudflare Worker: validation, tokens, rate limits, D1, KV
│   ├── migrations/           D1 schema
│   ├── test/                 Vitest
│   └── scripts/loadtest.mjs
├── extension/
│   └── src/                  content script, service worker, offscreen inference, aspect rules
├── web/                      website: paste a review, look up a product, live model metrics
├── notebooks/                Colab and Kaggle training notebooks
├── docs/GO_LIVE.md           deployment, free tiers only
└── .github/workflows/        CI, website deploy, extension release on a version tag
```

---

## What TrustLens will not do

- Call a review fake. "Reads AI-written" is the strongest claim it makes.
- Send review text anywhere. The API rejects payloads that try.
- Catch paid human reviews, review swaps, or incentivised five-stars.
- Work when Amazon changes its markup. If `data-hook="review"` disappears, the card stays hidden instead of guessing.

## Limitations

- Training data comes from three small open models. Reviews written by much larger models were not tested.
- 5.9% of real reviews still get flagged. On a product with 200 reviews, that is about a dozen honest customers wearing a badge they didn't earn.
- Aspect sentiment is distilled from a teacher, so it inherits the teacher's mistakes; 88% agreement means 12% disagreement.
- The shared report is only as good as the pages people have opened. A product nobody has visited has no report.

---

## Data and credit

- Amazon Reviews 2023: Hou et al., 2024, McAuley Lab, UC San Diego
- Fake Reviews Dataset: Salminen et al., 2022
- Models run with [Transformers.js](https://github.com/huggingface/transformers.js)

## License

MIT. See [LICENSE](LICENSE).

---

*Built by [Mahendra Meena](https://www.linkedin.com/in/mahendra-meena-72047b201/). A small model, a narrow question, and an honest table.*
