import { describe, expect, it } from "vitest";
import { buildReport, HttpError, signToken, trustAdjustedRating, validatePayload, verifyToken } from "../src/lib";

const good = {
  asin: "b0c1234567",
  model_version: "detector-v1",
  threshold: 0.7,
  reviews: [
    { id: "R1ABCDEF", rating: 5, ai_prob: 0.95, aspects: { battery: 1 } },
    { id: "R2ABCDEF", rating: 2, ai_prob: 0.05, aspects: { battery: -1, value: 1, bogus: 1 } },
  ],
};

describe("validatePayload", () => {
  it("accepts a good payload, upper-cases ASIN and drops unknown aspects", () => {
    const p = validatePayload(good);
    expect(p.asin).toBe("B0C1234567");
    expect(p.reviews[1].aspects).toEqual({ battery: -1, value: 1 });
  });
  it.each([
    [{ ...good, asin: "short" }, "asin"],
    [{ ...good, threshold: 2 }, "threshold"],
    [{ ...good, reviews: [] }, "non-empty"],
    [{ ...good, reviews: [{ ...good.reviews[0], rating: 9 }] }, "rating"],
    [{ ...good, reviews: [{ ...good.reviews[0], id: "<script>" }] }, "id"],
    [{ ...good, reviews: Array(51).fill(good.reviews[0]) }, "At most"],
  ])("rejects bad input %#", (body, msg) => {
    expect(() => validatePayload(body)).toThrowError(HttpError);
    expect(() => validatePayload(body)).toThrowError(new RegExp(msg));
  });
});

describe("scoring", () => {
  it("matches the Python implementation", () => {
    // python: trust_adjusted_rating([5,5,1,4],[0.9,0.95,0.1,0.2]) == 3.21
    expect(trustAdjustedRating([5, 5, 1, 4], [0.9, 0.95, 0.1, 0.2])).toBe(3.21);
  });
  it("builds a report that ignores suspicious reviews for aspects", () => {
    const r = buildReport("B0C1234567", [
      { rating: 5, ai_prob: 0.95, threshold: 0.7, aspects: '{"battery":1}' },
      { rating: 2, ai_prob: 0.05, threshold: 0.7, aspects: '{"battery":-1,"value":1}' },
    ]);
    expect(r.suspicious).toBe(1);
    expect(r.suspicious_pct).toBe(50);
    expect(r.shown_rating).toBe(3.5);
    expect(r.trust_rating).toBeLessThan(3.5);
    expect(r.aspects.find((a) => a.aspect === "battery")).toMatchObject({ pos: 0, neg: 1 });
  });
});

describe("tokens", () => {
  it("round-trips and rejects tampering", async () => {
    const t = await signToken("s3cret", "install-1");
    expect(await verifyToken("s3cret", t)).toBe("install-1");
    expect(await verifyToken("other", t)).toBeNull();
    expect(await verifyToken("s3cret", t.slice(0, -2) + "xx")).toBeNull();
    expect(await verifyToken("s3cret", "garbage")).toBeNull();
  });
});
