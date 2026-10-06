# Go live — step by step (all free)

| What | Where it runs | Cost |
|---|---|---|
| Training | Kaggle notebook (GPU T4 x2) | free |
| Models | Hugging Face Hub (model repos) | free |
| API + database + cache | Cloudflare Workers + D1 + KV | free tier |
| Website | GitHub Pages | free |
| Extension | user's Chrome (Load unpacked / GitHub release zip) | free |

Accounts you need: GitHub, Kaggle (phone-verified), Hugging Face, Cloudflare. No card is needed for any of them.

---

## 1. Put the code on GitHub
1. Create a **public** repo named `trustlens`.
2. Upload everything in this folder (including `.github/`). Do not upload `node_modules/`.
3. Check that the **Actions** tab shows the CI workflow running.

## 2. Train the models on Kaggle (~3 hours, mostly waiting)
1. Hugging Face → *Settings → Access Tokens* → **Create new token**, type **Write**. Copy it.
2. Kaggle → *Create → New Notebook → File → Import Notebook* → upload `notebooks/kaggle_trustlens.ipynb`.
3. Right panel: **Accelerator: GPU T4 x2**, **Internet: On**.
4. *Add-ons → Secrets → Add*: name `HF_TOKEN`, value = your token. Tick it for this notebook.
5. In the first code cell set `HF_USER` to your Hugging Face username. If your GitHub username is not `Mahendr99ar`, also fix `REPO`.
6. *Save Version → Save & Run All*. You can close the browser.
7. When it finishes, open `huggingface.co/<HF_USER>/trustlens-detector`. You should see `onnx/model_quantized.onnx`, `metrics.json` and `trustlens.json`.
8. Copy the **Resume bullet** printed by the last-but-one cell.

## 3. Deploy the API on Cloudflare (~15 minutes)
You need Node.js 20+ on your laptop (nodejs.org → LTS). No laptop setup? Open the GitHub repo → *Code → Codespaces → Create* and run the same commands there.

```bash
cd api
npm install
npx wrangler login                                  # opens the browser; allow access
npx wrangler d1 create trustlens                    # copy the database_id it prints
npx wrangler kv namespace create CACHE              # copy the id it prints
```
Paste both ids into `api/wrangler.toml` (replace the two `REPLACE_WITH_...` values), then:
```bash
npx wrangler d1 migrations apply trustlens --remote
node -e "console.log(crypto.randomUUID()+crypto.randomUUID())"   # copy this random string
npx wrangler secret put TOKEN_SECRET                # paste the random string
npx wrangler deploy                                 # prints https://trustlens-api.<you>.workers.dev
```
Open `https://trustlens-api.<you>.workers.dev/health` in the browser. It should show `{"status":"ok"}`.

## 4. Point the website and extension at your API and models
Edit two files (GitHub's web editor is fine):

**`web/index.html`**, in the `CONFIG` block near the bottom:
```js
API_BASE: "https://trustlens-api.<you>.workers.dev",
DETECTOR_MODEL: "<HF_USER>/trustlens-detector",
ASPECT_MODEL: "<HF_USER>/trustlens-aspects",
GITHUB: "https://github.com/<github-user>/trustlens",
```
**`extension/src/config.js`**: set the same `API_BASE`, `DETECTOR_MODEL` and `ASPECT_MODEL`.

Commit both.

## 5. Publish the website (GitHub Pages)
1. Repo → *Settings → Pages* → **Source: GitHub Actions**.
2. Repo → *Actions* → **Website** → *Run workflow* (later commits to `web/` deploy automatically).
3. Open `https://<github-user>.github.io/trustlens/`. Click an example. The first run downloads about 60 MB, then shows the verdict.

## 6. Release the extension
1. Repo → *Releases → Draft a new release* → tag `v0.1.0` → **Publish**.
2. The **Release extension** workflow builds `trustlens-extension.zip` and attaches it to the release (2–3 minutes).
3. Install it yourself: unzip → `chrome://extensions` → **Developer mode** on → **Load unpacked** → choose the unzipped folder.
4. Open any Amazon product page (amazon.in or amazon.com) and scroll to the reviews. A TrustLens card appears above them, with a badge on each review.

## 7. Finish
- Fill the Results tables in `README.md` from `metrics.json` / `export_report.json`.
- Run the load test (`api/scripts/loadtest.mjs`) against `wrangler dev` and add the numbers.
- Record a 20-second screen capture of the extension on a real product page and put the GIF at the top of the README.

## Troubleshooting
| Problem | Fix |
|---|---|
| Website says "Could not run the model" | Check the model names in `CONFIG`. Open `https://huggingface.co/<HF_USER>/trustlens-detector` — the repo must be public. |
| Extension card never appears | Amazon changes its page layout sometimes. Open DevTools → Console on the product page and look for `TrustLens` messages. Reviews must be in elements with `data-hook="review"`. |
| Extension shows results but no shared report | `API_BASE` is empty or wrong in `config.js`; check `/health` in the browser. |
| `wrangler deploy` fails with an id error | The two ids in `wrangler.toml` were not replaced. |
| 429 Too many requests | Rate limit (60 review posts / 120 reports per minute). Wait a minute. |

## Free-tier limits (as of 2026; check Cloudflare's pricing page)
Cloudflare Workers free plan allows a daily request quota and D1/KV have free storage and read/write quotas that are far
above what a portfolio project needs. Hugging Face model repos are free. GitHub Pages is free for public repos.
