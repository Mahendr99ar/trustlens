"""Upload exported models to your Hugging Face account (free model hosting; the extension and website load from there).

    HF_TOKEN=hf_xxx python scripts/07_push_to_hub.py --user <hf-username> --detector export/detector --aspects export/aspects
"""
import argparse
import os

from huggingface_hub import HfApi

CARD = """---
library_name: transformers.js
pipeline_tag: text-classification
tags: [trustlens, onnx, reviews]
---
# {name}

Part of **TrustLens** (detect AI-written product reviews, summarise genuine opinion by aspect).
Runs in the browser with Transformers.js (`onnx/model_quantized.onnx`, INT8).

See `metrics.json` and `export_report.json` for evaluation results.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--detector", default="export/detector")
    ap.add_argument("--aspects", default="export/aspects")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for folder, name in [(args.detector, "trustlens-detector"), (args.aspects, "trustlens-aspects")]:
        if not os.path.isdir(folder):
            print(f"skip {folder} (not found)")
            continue
        repo = f"{args.user}/{name}"
        with open(os.path.join(folder, "README.md"), "w") as f:
            f.write(CARD.format(name=name))
        api.create_repo(repo, exist_ok=True, private=args.private)
        api.upload_folder(repo_id=repo, folder_path=folder, commit_message="Upload TrustLens model")
        print(f"uploaded https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
