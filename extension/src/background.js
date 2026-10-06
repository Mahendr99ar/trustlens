// Service worker: routes work from the Amazon page to the offscreen model, then shares scores with the TrustLens API.
import { CONFIG } from "./config.js";

const OFFSCREEN = "offscreen.html";
let creating = null;

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument?.()) return;
  creating ||= chrome.offscreen.createDocument({
    url: OFFSCREEN,
    reasons: ["WORKERS"],
    justification: "Run the review-analysis model (WebAssembly) outside the web page.",
  }).finally(() => { creating = null; });
  await creating;
}

async function apiToken() {
  const { token } = await chrome.storage.local.get("token");
  if (token) return token;
  const r = await fetch(`${CONFIG.API_BASE}/v1/install`, { method: "POST" });
  if (!r.ok) throw new Error(`install failed (${r.status})`);
  const t = (await r.json()).token;
  await chrome.storage.local.set({ token: t });
  return t;
}

async function shareAndFetchReport(asin, threshold, results) {
  if (!CONFIG.API_BASE) return null;
  try {
    const post = (token) => fetch(`${CONFIG.API_BASE}/v1/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ asin, model_version: CONFIG.MODEL_VERSION, threshold, reviews: results.slice(0, 50) }),
    });
    let r = await post(await apiToken());
    if (r.status === 401) { // token from an older deployment: get a new one once
      await chrome.storage.local.remove("token");
      r = await post(await apiToken());
    }
    const rep = await fetch(`${CONFIG.API_BASE}/v1/products/${asin}/report`);
    return rep.ok ? await rep.json() : null;
  } catch (e) {
    console.warn("TrustLens API unavailable:", e);
    return null;
  }
}

/** Same formula as the API and ml/trustlens_ml/scoring.py. */
function trustRating(ratings, probs, prior = 2) {
  if (!ratings.length) return null;
  const plain = ratings.reduce((a, b) => a + b, 0) / ratings.length;
  let num = prior * plain, den = prior;
  ratings.forEach((r, i) => { const w = Math.max(0, 1 - probs[i]); num += w * r; den += w; });
  return Math.round((num / den) * 100) / 100;
}

function pageSummary(results, threshold) {
  const rated = results.filter((r) => r.rating);
  const flagged = results.filter((r) => r.ai_prob >= threshold);
  const aspects = {};
  for (const r of results) {
    if (r.ai_prob >= threshold) continue;
    for (const [a, s] of Object.entries(r.aspects)) {
      const d = (aspects[a] ||= { aspect: a, pos: 0, neg: 0, neu: 0 });
      d[s > 0 ? "pos" : s < 0 ? "neg" : "neu"]++;
    }
  }
  return {
    reviews: results.length,
    flagged: flagged.length,
    shown_rating: rated.length ? Math.round((rated.reduce((a, r) => a + r.rating, 0) / rated.length) * 100) / 100 : null,
    trust_rating: trustRating(rated.map((r) => r.rating), rated.map((r) => r.ai_prob)),
    aspects: Object.values(aspects).map((d) => ({ ...d, mentions: d.pos + d.neg + d.neu })).sort((a, b) => b.mentions - a.mentions),
  };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "analyze-page" || !sender.tab) return false;
  (async () => {
    await ensureOffscreen();
    const out = await chrome.runtime.sendMessage({ target: "offscreen", type: "analyze", reviews: msg.reviews });
    if (!out?.ok) throw new Error(out?.error || "model failed");
    const report = await shareAndFetchReport(msg.asin, out.threshold, out.results);
    return { ok: true, threshold: out.threshold, loadMs: out.loadMs, inferMs: out.inferMs, results: out.results,
             page: pageSummary(out.results, out.threshold), report };
  })().then(sendResponse, (e) => sendResponse({ ok: false, error: String(e?.message || e) }));
  return true;
});
