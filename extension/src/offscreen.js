// Runs the two ML models with Transformers.js (ONNX Runtime WebAssembly) inside an offscreen document.
// MV3 service workers cannot load the WebAssembly runtime, so the background script forwards work here.
import { env, pipeline } from "@huggingface/transformers";
import { CONFIG } from "./config.js";
import { aspectInputs } from "./aspects.js";

env.allowLocalModels = false;
env.remoteHost = CONFIG.MODEL_HOST;
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
env.backends.onnx.wasm.numThreads = 1; // extension pages are not cross-origin isolated

let models = null;
async function load() {
  if (models) return models;
  models = (async () => {
    const t0 = performance.now();
    const [detector, aspects] = await Promise.all([
      pipeline("text-classification", CONFIG.DETECTOR_MODEL, { dtype: "q8" }),
      pipeline("text-classification", CONFIG.ASPECT_MODEL, { dtype: "q8" }),
    ]);
    let threshold = CONFIG.DEFAULT_THRESHOLD;
    try {
      const r = await fetch(`${CONFIG.MODEL_HOST}${CONFIG.DETECTOR_MODEL}/resolve/main/trustlens.json`);
      if (r.ok) threshold = (await r.json()).threshold ?? threshold;
    } catch { /* keep default */ }
    return { detector, aspects, threshold, loadMs: Math.round(performance.now() - t0) };
  })();
  try {
    return await models;
  } catch (e) {
    models = null; // allow a retry
    throw e;
  }
}

const SIGN = { positive: 1, neutral: 0, negative: -1 };

async function analyze(reviews) {
  const m = await load();
  const t0 = performance.now();
  const texts = reviews.map((r) => `${r.title ? r.title + ". " : ""}${r.text}`.slice(0, 2000));
  const det = await m.detector(texts, { top_k: null });
  const aiProbs = det.map((scores) => (scores.find((s) => s.label === "ai") || { score: 0 }).score);

  const pairs = reviews.flatMap((r, i) => aspectInputs(r.text).map((p) => ({ ...p, review: i })));
  const perReview = reviews.map(() => ({}));
  if (pairs.length) {
    const out = await m.aspects(pairs.map((p) => p.input), { top_k: 1 });
    const sums = reviews.map(() => ({}));
    pairs.forEach((p, k) => {
      const top = Array.isArray(out[k]) ? out[k][0] : out[k];
      const s = SIGN[top.label] ?? 0;
      (sums[p.review][p.aspect] ||= []).push(s);
    });
    sums.forEach((byAspect, i) => {
      for (const [a, vals] of Object.entries(byAspect)) {
        const mean = vals.reduce((x, y) => x + y, 0) / vals.length;
        perReview[i][a] = mean > 0.2 ? 1 : mean < -0.2 ? -1 : 0;
      }
    });
  }
  return {
    threshold: m.threshold,
    loadMs: m.loadMs,
    inferMs: Math.round(performance.now() - t0),
    results: reviews.map((r, i) => ({ id: r.id, rating: r.rating, ai_prob: aiProbs[i], aspects: perReview[i] })),
  };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target !== "offscreen") return false;
  if (msg.type === "analyze") {
    analyze(msg.reviews).then((r) => sendResponse({ ok: true, ...r }), (e) => sendResponse({ ok: false, error: String(e?.message || e) }));
    return true; // async response
  }
  return false;
});
