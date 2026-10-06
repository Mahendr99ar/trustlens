// Same rules as ml/trustlens_ml/aspects.py (both read aspects.json).
import data from "../../ml/trustlens_ml/aspects.json";

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const PATTERNS = Object.entries(data.aspects).map(([aspect, kws]) => [
  aspect,
  new RegExp("\\b(" + [...kws].sort((a, b) => b.length - a.length).map(esc).join("|") + ")\\b", "i"),
]);

export function splitSentences(text) {
  return (text || "").split(/(?<=[.!?])\s+|\n+/).map((s) => s.trim()).filter((s) => s.length >= 3).map((s) => s.slice(0, 300));
}

export function findAspects(sentence) {
  return PATTERNS.filter(([, p]) => p.test(sentence)).map(([a]) => a);
}

/** [{aspect, sentence, input}] — `input` is exactly what the aspect model was trained on: "<aspect>: <sentence>". */
export function aspectInputs(text, maxPairs = 12) {
  const out = [];
  for (const s of splitSentences(text)) {
    for (const a of findAspects(s)) {
      out.push({ aspect: a, sentence: s, input: `${a.replace("_", " ")}: ${s}` });
      if (out.length >= maxPairs) return out;
    }
  }
  return out;
}

export const ASPECT_NAMES = Object.keys(data.aspects);
