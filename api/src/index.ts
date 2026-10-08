/**
 * TrustLens API: a Cloudflare Worker (free tier) with D1 (SQLite) and KV.
 *
 * POST /v1/install                 -> { token }                    one per extension install, rate-limited per IP
 * POST /v1/reviews   (Bearer)      -> { stored }                   scores computed in the browser, no review text stored
 * GET  /v1/products/:asin/report   -> trust report (KV-cached 10 min, invalidated on new scores)
 * GET  /health
 */
import { buildReport, HttpError, signToken, validatePayload, verifyToken, ASIN_RE, type StoredRow } from "./lib";

export interface Env {
  DB: D1Database;
  CACHE: KVNamespace;
  TOKEN_SECRET: string;
  ALLOWED_ORIGINS?: string; // comma-separated; "*" allows any
  DISABLE_RATE_LIMIT?: string; // "1" only for local load tests
}

const REPORT_TTL = 600;
const LIMITS = { install: 10, reviews: 60, report: 120 }; // requests per minute

function cors(req: Request, env: Env): Record<string, string> {
  const origin = req.headers.get("Origin") || "";
  const allowed = (env.ALLOWED_ORIGINS || "*").split(",").map((s) => s.trim());
  const ok = allowed.includes("*") || allowed.includes(origin) || origin.startsWith("chrome-extension://");
  return {
    "Access-Control-Allow-Origin": ok ? origin || "*" : "null",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function json(data: unknown, status: number, headers: Record<string, string>) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...headers } });
}

/** Fixed-window rate limit in KV. Good enough for a free-tier side project; not exact under heavy concurrency. */
async function rateLimit(env: Env, bucket: keyof typeof LIMITS, who: string) {
  if (env.DISABLE_RATE_LIMIT === "1") return;
  const window = Math.floor(Date.now() / 60000);
  const key = `rl:${bucket}:${who}:${window}`;
  const n = Number((await env.CACHE.get(key)) || 0);
  if (n >= LIMITS[bucket]) throw new HttpError(429, "Too many requests. Wait a minute and try again.");
  await env.CACHE.put(key, String(n + 1), { expirationTtl: 120 });
}

async function report(env: Env, asin: string) {
  const cached = await env.CACHE.get(`report:${asin}`);
  if (cached) return { ...JSON.parse(cached), cached: true };
  const { results } = await env.DB.prepare(
    "SELECT rating, ai_prob, aspects, threshold FROM reviews WHERE asin = ?1",
  ).bind(asin).all<StoredRow>();
  const r = { ...buildReport(asin, results || []), generated_at: new Date().toISOString() };
  await env.CACHE.put(`report:${asin}`, JSON.stringify(r), { expirationTtl: REPORT_TTL });
  return { ...r, cached: false };
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const h = cors(req, env);
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: h });
    const url = new URL(req.url);
    const ip = req.headers.get("CF-Connecting-IP") || "local";
    const t0 = Date.now();
    try {
      if (url.pathname === "/health") return json({ status: "ok" }, 200, h);

      if (url.pathname === "/v1/install" && req.method === "POST") {
        await rateLimit(env, "install", ip);
        return json({ token: await signToken(env.TOKEN_SECRET, crypto.randomUUID()) }, 201, h);
      }

      if (url.pathname === "/v1/reviews" && req.method === "POST") {
        const iid = await verifyToken(env.TOKEN_SECRET, (req.headers.get("Authorization") || "").replace(/^Bearer\s+/i, ""));
        if (!iid) throw new HttpError(401, "Missing or invalid token. Call POST /v1/install first.");
        await rateLimit(env, "reviews", iid);
        let body: unknown;
        try { body = await req.json(); } catch { throw new HttpError(400, "Body is not valid JSON."); }
        const p = validatePayload(body);
        const now = Math.floor(Date.now() / 1000);
        const stmt = env.DB.prepare(
          `INSERT INTO reviews (asin, review_id, rating, ai_prob, aspects, threshold, model_version, install_id, updated_at)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
           ON CONFLICT (asin, review_id) DO UPDATE SET rating = excluded.rating, ai_prob = excluded.ai_prob,
             aspects = excluded.aspects, threshold = excluded.threshold, model_version = excluded.model_version,
             install_id = excluded.install_id, updated_at = excluded.updated_at`,
        );
        await env.DB.batch(p.reviews.map((r) =>
          stmt.bind(p.asin, r.id, r.rating, r.ai_prob, JSON.stringify(r.aspects), p.threshold, p.model_version, iid, now)));
        await env.CACHE.delete(`report:${p.asin}`);
        return json({ stored: p.reviews.length, asin: p.asin, ms: Date.now() - t0 }, 200, h);
      }

      const m = url.pathname.match(/^\/v1\/products\/([A-Za-z0-9]{10})\/report$/);
      if (m && req.method === "GET") {
        const asin = m[1].toUpperCase();
        if (!ASIN_RE.test(asin)) throw new HttpError(400, "Invalid ASIN.");
        await rateLimit(env, "report", ip);
        const r = await report(env, asin);
        return json({ ...r, ms: Date.now() - t0 }, 200, { ...h, "Cache-Control": "public, max-age=60" });
      }

      throw new HttpError(404, "Not found. See GET /health, POST /v1/install, POST /v1/reviews, GET /v1/products/:asin/report.");
    } catch (e) {
      if (e instanceof HttpError) return json({ error: e.message }, e.status, h);
      console.error(e);
      return json({ error: "Internal error. Try again." }, 500, h);
    }
  },
};
