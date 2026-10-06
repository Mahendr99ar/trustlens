// Pure logic (no Cloudflare APIs) so it can be unit-tested in Node.

export const ASIN_RE = /^[A-Z0-9]{10}$/;
export const MAX_REVIEWS_PER_POST = 50;
export const ASPECTS = new Set([
  "quality", "value", "durability", "battery", "sound", "display", "performance",
  "size_fit", "comfort", "ease_of_use", "design", "delivery", "service",
]);

export interface ScoredReview {
  id: string;
  rating: number;      // 1-5 stars shown on the review
  ai_prob: number;     // model's P(AI-written)
  aspects: Record<string, -1 | 0 | 1>;
}

export interface ReviewsPayload {
  asin: string;
  model_version: string;
  threshold: number;
  reviews: ScoredReview[];
}

export class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const isNum = (x: unknown, lo: number, hi: number): x is number =>
  typeof x === "number" && Number.isFinite(x) && x >= lo && x <= hi;

/** Validate untrusted JSON from the extension. Throws HttpError(400) with a specific message. */
export function validatePayload(body: unknown): ReviewsPayload {
  const b = body as Record<string, unknown>;
  if (!b || typeof b !== "object") throw new HttpError(400, "Body must be a JSON object.");
  const asin = String(b.asin ?? "").toUpperCase();
  if (!ASIN_RE.test(asin)) throw new HttpError(400, "asin must be 10 letters/digits, e.g. B0C1234567.");
  const model_version = String(b.model_version ?? "").slice(0, 40);
  if (!model_version) throw new HttpError(400, "model_version is required.");
  if (!isNum(b.threshold, 0, 1)) throw new HttpError(400, "threshold must be a number between 0 and 1.");
  if (!Array.isArray(b.reviews) || b.reviews.length === 0) throw new HttpError(400, "reviews must be a non-empty array.");
  if (b.reviews.length > MAX_REVIEWS_PER_POST) throw new HttpError(400, `At most ${MAX_REVIEWS_PER_POST} reviews per request.`);
  const reviews: ScoredReview[] = b.reviews.map((r: any, i: number) => {
    const id = String(r?.id ?? "");
    if (!/^[A-Za-z0-9_-]{4,40}$/.test(id)) throw new HttpError(400, `reviews[${i}].id is not a valid review id.`);
    if (!isNum(r.rating, 1, 5)) throw new HttpError(400, `reviews[${i}].rating must be 1-5.`);
    if (!isNum(r.ai_prob, 0, 1)) throw new HttpError(400, `reviews[${i}].ai_prob must be 0-1.`);
    const aspects: Record<string, -1 | 0 | 1> = {};
    for (const [k, v] of Object.entries(r.aspects ?? {})) {
      if (ASPECTS.has(k) && (v === -1 || v === 0 || v === 1)) aspects[k] = v;
    }
    return { id, rating: r.rating, ai_prob: r.ai_prob, aspects };
  });
  return { asin, model_version, threshold: b.threshold as number, reviews };
}

/** Same formula as ml/trustlens_ml/scoring.py: weight each review by (1 - P(AI)), shrink toward the plain mean. */
export function trustAdjustedRating(ratings: number[], aiProbs: number[], priorWeight = 2): number | null {
  if (!ratings.length) return null;
  const plain = ratings.reduce((a, b) => a + b, 0) / ratings.length;
  let num = priorWeight * plain, den = priorWeight;
  ratings.forEach((r, i) => {
    const w = Math.max(0, 1 - aiProbs[i]);
    num += w * r;
    den += w;
  });
  return Math.round((num / den) * 100) / 100;
}

export interface StoredRow { rating: number; ai_prob: number; aspects: string; threshold: number }

export interface Report {
  asin: string;
  reviews_analyzed: number;
  suspicious: number;
  suspicious_pct: number;
  shown_rating: number | null;
  trust_rating: number | null;
  aspects: { aspect: string; pos: number; neg: number; neu: number; mentions: number }[];
}

export function buildReport(asin: string, rows: StoredRow[]): Report {
  const ratings = rows.map((r) => r.rating);
  const probs = rows.map((r) => r.ai_prob);
  const suspicious = rows.filter((r) => r.ai_prob >= r.threshold).length;
  const agg = new Map<string, { pos: number; neg: number; neu: number }>();
  for (const r of rows) {
    if (r.ai_prob >= r.threshold) continue; // summarise only reviews that look genuine
    let aspects: Record<string, number> = {};
    try { aspects = JSON.parse(r.aspects || "{}"); } catch { /* ignore corrupt row */ }
    for (const [k, v] of Object.entries(aspects)) {
      const d = agg.get(k) ?? { pos: 0, neg: 0, neu: 0 };
      if (v > 0) d.pos++; else if (v < 0) d.neg++; else d.neu++;
      agg.set(k, d);
    }
  }
  const shown = ratings.length ? Math.round((ratings.reduce((a, b) => a + b, 0) / ratings.length) * 100) / 100 : null;
  return {
    asin,
    reviews_analyzed: rows.length,
    suspicious,
    suspicious_pct: rows.length ? Math.round((1000 * suspicious) / rows.length) / 10 : 0,
    shown_rating: shown,
    trust_rating: trustAdjustedRating(ratings, probs),
    aspects: [...agg.entries()]
      .map(([aspect, d]) => ({ aspect, ...d, mentions: d.pos + d.neg + d.neu }))
      .sort((a, b) => b.mentions - a.mentions),
  };
}

// ---- Install tokens: stateless HMAC-signed tokens, "<base64url(payload)>.<base64url(sig)>" ----

const enc = new TextEncoder();
const b64url = (buf: ArrayBuffer | Uint8Array) =>
  btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const fromB64url = (s: string) =>
  Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4)), (c) => c.charCodeAt(0));

async function hmacKey(secret: string) {
  return crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

export async function signToken(secret: string, installId: string, now = Date.now()): Promise<string> {
  const payload = b64url(enc.encode(JSON.stringify({ iid: installId, iat: Math.floor(now / 1000) })));
  const sig = await crypto.subtle.sign("HMAC", await hmacKey(secret), enc.encode(payload));
  return `${payload}.${b64url(sig)}`;
}

export async function verifyToken(secret: string, token: string): Promise<string | null> {
  const [payload, sig] = (token || "").split(".");
  if (!payload || !sig) return null;
  try {
    const ok = await crypto.subtle.verify("HMAC", await hmacKey(secret), fromB64url(sig), enc.encode(payload));
    if (!ok) return null;
    const { iid } = JSON.parse(new TextDecoder().decode(fromB64url(payload)));
    return typeof iid === "string" ? iid : null;
  } catch {
    return null;
  }
}
