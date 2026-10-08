// Reads the reviews visible on an Amazon product / reviews page, asks the extension to score them,
// then shows a trust panel and a small badge on each review. Review text never leaves the device;
// only scores (and Amazon's public review IDs) are shared with the TrustLens API.
(() => {
  const ASIN = (location.pathname.match(/\/(?:dp|gp\/product|product-reviews)\/([A-Z0-9]{10})/) || [])[1];
  if (!ASIN) return;

  const seen = new Map(); // review id -> result
  let pending = null, busy = false, threshold = null, lastPage = null, lastReport = null;

  const ASPECT_LABEL = {
    quality: "Build quality", value: "Value for money", durability: "Durability", battery: "Battery", sound: "Sound",
    display: "Display", performance: "Performance", size_fit: "Size & fit", comfort: "Comfort",
    ease_of_use: "Ease of use", design: "Look & design", delivery: "Delivery & packaging", service: "Seller & support",
  };

  function extract() {
    return [...document.querySelectorAll('[data-hook="review"]')].map((el) => {
      const starText = el.querySelector('[data-hook="review-star-rating"], [data-hook="cmps-review-star-rating"], .review-rating')?.textContent || "";
      const rating = parseFloat((starText.match(/(\d[.,]?\d?)/) || [])[1]?.replace(",", ".") || "") || null;
      const body = el.querySelector('[data-hook="review-body"]');
      const titleEl = el.querySelector('[data-hook="review-title"]');
      const title = titleEl ? [...titleEl.querySelectorAll("span")].map((s) => s.textContent.trim()).filter(Boolean).pop() || "" : "";
      return { el, id: el.id, rating, title, text: (body?.innerText || "").replace(/Read more$/i, "").trim() };
    }).filter((r) => r.id && r.text.length > 20);
  }

  // UI lives in a shadow root so Amazon's CSS cannot restyle it.
  const host = document.createElement("div");
  host.id = "trustlens-panel";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>
    :host{all:initial}
    .card{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1F2A37;background:#fff;border:1px solid #D5DCE3;
      border-left:4px solid #0F766E;border-radius:10px;padding:14px 16px;margin:12px 0;max-width:640px;box-sizing:border-box}
    .floating{position:fixed;right:16px;bottom:16px;z-index:2147483646;width:min(360px,calc(100vw - 32px));box-shadow:0 8px 24px rgba(15,30,40,.18)}
    .head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px}
    .brand{font-weight:700;letter-spacing:-.01em}
    .x{border:0;background:none;font-size:18px;cursor:pointer;color:#5B6774;padding:0 4px}
    .ratings{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;margin:6px 0 4px}
    .big{font-size:28px;font-weight:700;letter-spacing:-.02em}
    .muted{color:#5B6774;font-size:13px}
    .flag{color:#B45309;font-weight:600}
    .ok{color:#0F766E;font-weight:600}
    table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
    td{padding:4px 0;vertical-align:middle}
    td.n{width:56px;text-align:right;color:#5B6774;font-variant-numeric:tabular-nums}
    .bar{display:flex;height:8px;border-radius:4px;overflow:hidden;background:#EEF1F4;min-width:120px}
    .bar i{display:block;height:100%}
    .pos{background:#0F766E}.neg{background:#B45309}.neu{background:#A9B4BF}
    .foot{margin-top:10px;font-size:12px;color:#5B6774}
    .err{color:#B42318}
  </style><div class="card" part="card"><div class="head"><span class="brand">TrustLens</span><button class="x" title="Hide">×</button></div><div class="body">Checking reviews on this page…</div></div>`;
  root.querySelector(".x").onclick = () => host.remove();
  const body = root.querySelector(".body");

  function mount() {
    if (host.isConnected) return;
    const anchor = document.querySelector('#cm-cr-dp-review-list, [data-hook="top-customer-reviews-widget"], #cm_cr-review_list');
    if (anchor?.parentElement) anchor.parentElement.insertBefore(host, anchor);
    else { root.querySelector(".card").classList.add("floating"); document.body.appendChild(host); }
  }

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function render() {
    const p = lastPage, rep = lastReport;
    if (!p) return;
    const s = rep && rep.reviews_analyzed > p.reviews ? rep : null; // prefer the bigger, shared sample
    const n = s ? s.reviews_analyzed : p.reviews, flagged = s ? s.suspicious : p.flagged;
    const shown = s ? s.shown_rating : p.shown_rating, trust = s ? s.trust_rating : p.trust_rating;
    const aspects = (s ? s.aspects : p.aspects).slice(0, 6);
    const flagLine = flagged
      ? `<span class="flag">${flagged} of ${n} reviews look AI-written</span>`
      : `<span class="ok">None of ${n} reviews look AI-written</span>`;
    body.innerHTML = `
      ${flagLine}
      <div class="ratings">
        <div><div class="big">${trust ?? "–"}★</div><div class="muted">trust-adjusted</div></div>
        <div><div class="big" style="color:#5B6774">${shown ?? "–"}★</div><div class="muted">average of these reviews</div></div>
      </div>
      ${aspects.length ? `<div class="muted" style="margin-top:6px">What genuine reviewers mention</div><table>${aspects.map((a) => `
        <tr><td style="width:40%">${esc(ASPECT_LABEL[a.aspect] || a.aspect)}</td>
        <td><div class="bar"><i class="pos" style="width:${(100 * a.pos) / a.mentions}%"></i><i class="neu" style="width:${(100 * a.neu) / a.mentions}%"></i><i class="neg" style="width:${(100 * a.neg) / a.mentions}%"></i></div></td>
        <td class="n">${a.pos}↑ ${a.neg}↓</td></tr>`).join("")}</table>` : ""}
      <div class="foot">${s ? `Based on ${n} reviews scored by TrustLens users for this product.` : `Based on the ${n} reviews loaded on this page.`}
        Scores are model estimates, not proof of fraud.</div>`;
  }

  function badge(r, res) {
    r.el.querySelector(".trustlens-badge")?.remove();
    const b = document.createElement("span");
    b.className = "trustlens-badge";
    const flagged = res.ai_prob >= threshold;
    b.textContent = flagged ? `TrustLens: looks AI-written (${Math.round(res.ai_prob * 100)}%)` : "TrustLens: looks genuine";
    b.title = "Probability this review was written by an AI model, estimated on your device.";
    b.style.cssText = `display:inline-block;margin:4px 8px;padding:2px 8px;border-radius:999px;font:600 12px system-ui,sans-serif;` +
      (flagged ? "background:#FEF3E2;color:#92400E;border:1px solid #F5C99B" : "background:#E7F5F3;color:#0F5F58;border:1px solid #B5DED8");
    (r.el.querySelector('[data-hook="review-title"]') || r.el.firstElementChild || r.el).after(b);
  }

  async function run() {
    const fresh = extract().filter((r) => !seen.has(r.id));
    if (!fresh.length) return;
    mount();
    if (!lastPage) body.textContent = "Loading the on-device model (first time only, ~60 MB)…";
    const res = await chrome.runtime.sendMessage({
      type: "analyze-page", asin: ASIN,
      reviews: [...fresh.slice(0, 50)].map(({ id, rating, title, text }) => ({ id, rating, title, text })),
    });
    if (!res?.ok) { body.innerHTML = `<span class="err">Could not analyse reviews: ${esc(res?.error || "unknown error")}</span>`; return; }
    threshold = res.threshold;
    const byId = new Map(res.results.map((x) => [x.id, x]));
    fresh.forEach((r) => { const x = byId.get(r.id); if (x) { seen.set(r.id, x); badge(r, x); } });
    // Recompute the page summary over everything analysed so far
    const all = [...seen.values()];
    lastPage = summarize(all, threshold);
    if (res.report) lastReport = res.report;
    render();
  }

  function summarize(results, thr) {
    const rated = results.filter((r) => r.rating);
    const trust = (() => {
      if (!rated.length) return null;
      const plain = rated.reduce((a, r) => a + r.rating, 0) / rated.length;
      let num = 2 * plain, den = 2;
      rated.forEach((r) => { const w = Math.max(0, 1 - r.ai_prob); num += w * r.rating; den += w; });
      return Math.round((num / den) * 100) / 100;
    })();
    const agg = {};
    results.filter((r) => r.ai_prob < thr).forEach((r) => Object.entries(r.aspects).forEach(([a, s]) => {
      const d = (agg[a] ||= { aspect: a, pos: 0, neg: 0, neu: 0 });
      d[s > 0 ? "pos" : s < 0 ? "neg" : "neu"]++;
    }));
    return {
      reviews: results.length, flagged: results.filter((r) => r.ai_prob >= thr).length,
      shown_rating: rated.length ? Math.round((rated.reduce((a, r) => a + r.rating, 0) / rated.length) * 100) / 100 : null,
      trust_rating: trust,
      aspects: Object.values(agg).map((d) => ({ ...d, mentions: d.pos + d.neg + d.neu })).sort((a, b) => b.mentions - a.mentions),
    };
  }

  const schedule = () => {
    clearTimeout(pending);
    pending = setTimeout(async () => {
      if (busy) return schedule();
      busy = true;
      try { await run(); } catch (e) { console.warn("TrustLens:", e); } finally { busy = false; }
    }, 700);
  };
  new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  schedule();
})();
