# TrustLens

A Chrome extension that marks Amazon reviews which look AI-written, recomputes the star rating with those reviews down-weighted, and summarises what the remaining reviewers say about battery, build quality, value and other aspects.

Both models run in the browser. The shared backend only stores scores, never review text, so everything runs on free tiers.

**Website:** https://mahendr99ar.github.io/trustlens/
**Extension:** [latest release](../../releases/latest)
**Models:** [trustlens-detector](https://huggingface.co/Mahendra99ar/trustlens-detector), [trustlens-aspects](https://huggingface.co/Mahendra99ar/trustlens-aspects)

Mozilla shut down Fakespot in July 2025, and small open LLMs now write believable reviews. TrustLens checks how well a model small enough to run in the browser can spot them.

## Results

The detector is MiniLM-L12 fine-tuned on 22,670 reviews: half real (Amazon Reviews 2023), half written by Qwen2.5-1.5B and SmolLM2-1.7B. The decision threshold is picked on validation data so that at most 5% of real reviews get flagged, because wrongly accusing a real customer is the worse mistake.

| Test set | ROC-AUC | F1 | AI reviews caught | Real reviews flagged |
|---|---|---|---|---|
| New products, same two LLMs (2,748) | 0.988 | 0.95 | 96% | 5.9% |
| New LLM, Phi-3.5-mini, never seen in training (2,700) | 0.999 | 0.97 | 99.8% | 5.8% |
| Salminen et al. 2022 fake reviews (4,000) | 0.584 | 0.31 | 21% | 13.2% |

What these numbers do and don't say:

- **The held-out LLM.** Phi-3.5 reviews were never used for training, and the detector still catches almost all of them. That suggests it learned something about current instruction-tuned LLM writing rather than one model's quirks.
- **The external set.** Salminen et al. generated their fake reviews in 2022 with an older, smaller model, and their reviews are shorter. On that set the detector is barely better than chance, so it should not be read as a general "fake review" detector.
- **Paraphrases.** When an LLM rewrites a genuine review, the detector flags 86% of them. There is no single right label for those, so they are reported separately and not counted in any score above.
- **Length.** AI reviews were generated to match the star rating and word count of real ones (median 58 words in both classes). A classifier that only sees length still reaches 0.65 AUC, so length carries some signal. The detector's 0.988 is far above that.

The aspect model is a second MiniLM, distilled from a DeBERTa-v3 aspect-sentiment teacher. The teacher labelled 60,001 (aspect, sentence) pairs from real reviews across 13 aspects, and I kept the 56,264 it was confident about. The student agrees with the teacher on 88.0% of held-out sentences (macro F1 0.86).

**In the browser.** Both models are exported to ONNX and quantized to INT8: 128 MB down to 33 MB each. The INT8 detector gives the same verdict as the full-precision one on 97.8% of 500 test reviews and takes about 44 ms per review on one CPU thread.

## How it works

```
 Training (Colab / Kaggle GPU)                  In the user's browser                    Shared backend (free tier)
 ─────────────────────────────                  ─────────────────────                    ──────────────────────────
 Amazon Reviews 2023: 47,896 reviews,           content script finds reviews             Cloudflare Worker (TypeScript)
 8 categories, balanced 1-5 stars               on the product page                        POST /v1/reviews  scores only
 + 17,453 AI reviews from 3 open LLMs                 │                                    GET  /v1/products/:asin/report
       │ split by product, held-out LLM               ▼                                    D1 (SQLite) + KV cache
       ▼                                        offscreen document runs both               HMAC-signed install tokens,
 detector (MiniLM)    aspects (MiniLM,          INT8 models with Transformers.js           per-install rate limits
       │              distilled from DeBERTa)         │
       └──► ONNX INT8 ──► Hugging Face Hub ──────────►│ card above the reviews: trust-adjusted
                                                      │ rating, badge on each review,
                                                      │ what genuine reviewers say per aspect
```

| Folder | What is in it |
|---|---|
| `ml/` | Data collection, AI review generation, splits, training, distillation, ONNX export with parity checks, upload to the Hub. `trustlens_ml` holds the aspect matching and the rating maths |
| `api/` | Cloudflare Worker: input validation, HMAC install tokens, KV rate limiting and caching, D1 upserts, Vitest tests, a load-test script |
| `extension/` | Chrome MV3: content script with a shadow-DOM card, service worker, offscreen document for inference, esbuild bundle |
| `web/` | The website: paste a review to test it, look up a product's shared report, live metrics from the model repo |
| `notebooks/` | `colab_trustlens.ipynb` (one T4, about 4.5 hours, resumes after a disconnect) and `kaggle_trustlens.ipynb` (two T4s) |
| `.github/workflows` | CI for all three parts, website deploy, extension build on a version tag |

**Trust-adjusted rating.** Each review counts with weight 1 minus its AI probability. The result is then pulled toward the product's plain average by 2 pseudo-reviews, so a product with only a few reviews does not swing wildly when one of them is flagged.

## Run it

- **Train:** open `notebooks/colab_trustlens.ipynb` in Colab with a T4, add an `HF_TOKEN` secret, and run the cells in order. The models land in your Hugging Face account.
- **Deploy:** [docs/GO_LIVE.md](docs/GO_LIVE.md) covers the API, website and extension.
- **Tests:** `cd api && npm test`, `cd ml && bash tests/test_offline_pipeline.sh fake_reviews.csv` (Salminen CSV, runs on CPU with tiny random models), `cd extension && npm run build`.
- **Load test:** `cd api && npx wrangler dev`, then `node scripts/loadtest.mjs http://127.0.0.1:8787`.

## Limitations

- "AI-written" is not the same as "fake". Paid human reviews are not detected, and a genuine review polished with an LLM may be flagged.
- Training data comes from three small open models. Reviews from much larger models were not tested.
- Amazon changes its page markup from time to time. The extension looks for `data-hook="review"` and shows nothing if it is missing.

Data: Amazon Reviews 2023 (Hou et al., 2024, McAuley Lab, UCSD); Fake Reviews Dataset (Salminen et al., 2022).
