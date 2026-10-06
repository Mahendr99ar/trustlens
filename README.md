# TrustLens — is this review written by a person?

TrustLens flags Amazon reviews that look **AI-written**, recomputes a **trust-adjusted star rating**, and summarises what
**genuine** reviewers say about each aspect (battery, build quality, value…). The models run **on the user's device**;
only scores are shared, so the whole system runs on free tiers.

**[Live demo](https://REPLACE.github.io/trustlens/)** · **[Chrome extension](../../releases/latest)** · **Models:** [detector](https://huggingface.co/REPLACE/trustlens-detector), [aspects](https://huggingface.co/REPLACE/trustlens-aspects)

> Why: review-checking tools such as Fakespot (shut down July 2025) and ReviewMeta are gone, while LLMs make
> fake reviews cheap to write.

## Architecture

```
 OFFLINE (Kaggle GPU)                                   ONLINE (all free tier)
 ─────────────────────                                  ─────────────────────────────────────────────────────────
 Amazon Reviews 2023 ─┐                                 Chrome extension (MV3)            Cloudflare Worker (TypeScript)
 (real reviews)       ├─► leak-free splits ─► detector   content script reads reviews ─►  POST /v1/reviews  (scores only)
 3 open LLMs ─────────┘   (by product,       (MiniLM)    offscreen doc runs ONNX models    GET  /v1/products/:asin/report
 (AI reviews, rating      length-matched,                Transformers.js + WASM, INT8      D1 (SQLite) + KV cache
  & length matched)       held-out LLM)     ─► aspects ─► ONNX INT8 ─► Hugging Face Hub ──► HMAC install tokens, rate limits
 DeBERTa ABSA teacher ──► distillation ───► (MiniLM)                                       Website (GitHub Pages): live demo,
                                                                                            product lookup, live metrics
```

| Part | Tech | What it shows |
|---|---|---|
| `ml/` | PyTorch, Transformers, ONNX Runtime | Dataset construction with shortcut controls, fine-tuning, knowledge distillation, calibrated threshold, INT8 quantization, parity checks; pip-installable `trustlens_ml` package |
| `api/` | Cloudflare Workers, D1, KV, TypeScript, Vitest | REST API, input validation, HMAC-signed tokens, KV rate limiting, cache invalidation, upserts, load test |
| `extension/` | Chrome MV3, Transformers.js, esbuild | On-device inference in an offscreen document, shadow-DOM UI, incremental analysis as reviews load |
| `web/` | Static HTML + Transformers.js | Paste-a-review demo, product reports, evaluation numbers read live from the model repo |
| `.github/workflows` | GitHub Actions | CI for all three parts, website deploy, extension release on tag |

## Results

Filled from `metrics.json` after training (`notebooks/kaggle_trustlens.ipynb` prints them).

| Test set | ROC-AUC | F1 | AI reviews caught | Real reviews wrongly flagged |
|---|---|---|---|---|
| Unseen products, same LLMs | – | – | – | – |
| Unseen LLM (Phi-3.5) | – | – | – | – |
| External: Salminen et al. 2022 | – | – | – | – |

| Model size / speed | fp32 | INT8 |
|---|---|---|
| Detector size | – MB | – MB |
| CPU latency per review (1 thread) | – ms | – ms |
| Top-1 agreement with fp32 | | –% |

**Evaluation design**
- Splits are **by product**, so no product appears in both training and test data.
- AI reviews are generated to **match the star-rating and length distribution** of real ones, then the training set is length-matched. A length-only baseline's AUC is reported before/after matching to show the shortcut is removed.
- One LLM is **held out entirely** to test generalisation to a generator the model never saw.
- The decision threshold is chosen on validation data to **flag at most 5% of genuine reviews**, because wrongly accusing a real customer is the costly error.
- LLM-**paraphrased** real reviews are reported separately: there is no single right label for them.

**Limitations.** "AI-written" is not the same as "fake": paid human reviews are not detected, and an AI-polished genuine
review may be flagged. Scores are estimates and the UI says so.

## Run it

- Train: open `notebooks/kaggle_trustlens.ipynb` on Kaggle (GPU T4 x2) → models land on your Hugging Face account.
- Deploy: [docs/GO_LIVE.md](docs/GO_LIVE.md) walks through the API, website and extension.
- Tests: `cd api && npm test` · `cd ml && bash tests/test_offline_pipeline.sh fake_reviews.csv` · `cd extension && npm run build`
- Load test: `cd api && npx wrangler dev` then `node scripts/loadtest.mjs http://127.0.0.1:8787`

Data: Amazon Reviews 2023 (Hou et al., McAuley Lab, UCSD); Fake Reviews Dataset (Salminen et al., 2022).
