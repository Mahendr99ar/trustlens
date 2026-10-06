"""Turning per-review model outputs into a product-level report. Mirrored in api/src/lib.ts."""


def trust_adjusted_rating(ratings, ai_probs, prior_weight=2.0):
    """Weighted mean of star ratings where each review counts (1 - P(AI-written)).

    Shrunk toward the plain average with `prior_weight` pseudo-reviews so a product with
    only a few reviews does not swing wildly.
    """
    if not ratings:
        return None
    plain = sum(ratings) / len(ratings)
    weights = [max(0.0, 1.0 - p) for p in ai_probs]
    num = sum(w * r for w, r in zip(weights, ratings)) + prior_weight * plain
    den = sum(weights) + prior_weight
    return round(num / den, 2)


def summarise_aspects(per_review_aspects, ai_probs, threshold):
    """per_review_aspects: list of {aspect: -1|0|1}. Only reviews below the AI threshold are counted."""
    agg = {}
    for aspects, p in zip(per_review_aspects, ai_probs):
        if p >= threshold:
            continue
        for a, s in aspects.items():
            d = agg.setdefault(a, {"pos": 0, "neg": 0, "neu": 0})
            d["pos" if s > 0 else "neg" if s < 0 else "neu"] += 1
    rows = [{"aspect": a, **d, "mentions": d["pos"] + d["neg"] + d["neu"]} for a, d in agg.items()]
    return sorted(rows, key=lambda r: -r["mentions"])
