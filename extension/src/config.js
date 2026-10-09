// Set these after deploying (docs/GO_LIVE.md).
export const CONFIG = {
  // Your Cloudflare Worker URL, e.g. "https://trustlens-api.<your-subdomain>.workers.dev". Leave "" to run offline.
  API_BASE: "https://trustlens-api.mahendra99ar.workers.dev",
  // Hugging Face model repos created by ml/scripts/07_push_to_hub.py
  DETECTOR_MODEL: "Mahendra99ar/trustlens-detector",
  ASPECT_MODEL: "Mahendra99ar/trustlens-aspects",
  // Where models are downloaded from. Keep the default unless you self-host the model files.
  MODEL_HOST: "https://huggingface.co/",
  // Used only if the model repo has no trustlens.json with a tuned threshold.
  DEFAULT_THRESHOLD: 0.7,
  MODEL_VERSION: "v1",
};
