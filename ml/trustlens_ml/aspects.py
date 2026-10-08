"""Aspect detection shared by training (Python) and inference (extension/web read the same aspects.json).

The aspect model is a classifier over strings of the form  "<aspect>: <sentence>"  ->  negative / neutral / positive.
This module finds which aspects a sentence mentions; extension/src/aspects.js implements the identical rules.
"""
import json
import os
import re

with open(os.path.join(os.path.dirname(__file__), "aspects.json")) as f:
    ASPECTS = json.load(f)["aspects"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_PATTERNS = {a: re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)) + r")\b", re.I)
             for a, kws in ASPECTS.items()}
MAX_SENT_CHARS = 300


def split_sentences(text):
    sents = [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]
    return [s[:MAX_SENT_CHARS] for s in sents if len(s) >= 3]


def find_aspects(sentence):
    return [a for a, p in _PATTERNS.items() if p.search(sentence)]


def find_aspect_terms(sentence):
    """[(aspect, matched keyword)]. The keyword is what the ABSA teacher model is asked about."""
    out = []
    for a, p in _PATTERNS.items():
        m = p.search(sentence)
        if m:
            out.append((a, m.group(1)))
    return out


def aspect_inputs(text, max_pairs=12):
    """[(aspect, sentence, model_input_string)] for every aspect mention in a review."""
    out = []
    for s in split_sentences(text):
        for a in find_aspects(s):
            out.append((a, s, f"{a.replace('_', ' ')}: {s}"))
            if len(out) >= max_pairs:
                return out
    return out
