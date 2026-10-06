// Concurrency load test for the report endpoint: compares cache-miss vs cache-hit latency.
//   node scripts/loadtest.mjs http://127.0.0.1:8787 [requests=400] [concurrency=20]
// Run against `wrangler dev` with DISABLE_RATE_LIMIT=1 in .dev.vars (rate limits would otherwise kick in).
const [base = "http://127.0.0.1:8787", total = "400", conc = "20"] = process.argv.slice(2);
const asins = Array.from({ length: 50 }, (_, i) => `B0LOAD${String(i).padStart(4, "0")}`);

const tok = (await (await fetch(`${base}/v1/install`, { method: "POST" })).json()).token;
for (const asin of asins) {
  const reviews = Array.from({ length: 20 }, (_, j) => ({
    id: `R${asin}${j}`, rating: 1 + (j % 5), ai_prob: (j * 37 % 100) / 100, aspects: { quality: j % 2 ? 1 : -1 },
  }));
  await fetch(`${base}/v1/reviews`, { method: "POST", headers: { Authorization: `Bearer ${tok}`, "Content-Type": "application/json" },
    body: JSON.stringify({ asin, model_version: "load", threshold: 0.7, reviews }) });
}

async function run(label) {
  const lat = []; let next = 0, errors = 0;
  const t0 = performance.now();
  await Promise.all(Array.from({ length: +conc }, async () => {
    while (next < +total) {
      const asin = asins[next++ % asins.length];
      const s = performance.now();
      const r = await fetch(`${base}/v1/products/${asin}/report`);
      if (!r.ok) errors++; else await r.json();
      lat.push(performance.now() - s);
    }
  }));
  lat.sort((a, b) => a - b);
  const q = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))].toFixed(1);
  const secs = (performance.now() - t0) / 1000;
  console.log(`${label}: ${lat.length} requests, ${conc} concurrent, ${(lat.length / secs).toFixed(0)} req/s, p50 ${q(0.5)} ms, p95 ${q(0.95)} ms, errors ${errors}`);
}
await run("first pass (50 cold products, rest cached)");
await run("second pass (KV cache hits)");
