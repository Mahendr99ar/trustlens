"""Generates notebooks/kaggle_trustlens.ipynb (the whole ML side: data -> AI reviews -> models -> ONNX -> Hugging Face)."""
import json
import os

MD, CODE = "markdown", "code"
cells = [
(MD, """# TrustLens: train the models on Kaggle (free GPU)

**Before running**
1. Right panel → *Session options* → **Accelerator: GPU T4 x2**, **Internet: On**.
2. *Add-ons → Secrets* → add a secret named **`HF_TOKEN`** with a Hugging Face **write** token (huggingface.co → Settings → Access Tokens), and switch it on for this notebook.
3. Put your Hugging Face username in the settings cell below.

Then **Save Version → Save & Run All (Commit)**: it runs in the background, so you can close the browser or sleep. (Running cells by hand also works, but Kaggle stops an idle interactive session.) Takes about 2.5–3.5 hours. At the end your two models are on Hugging Face and the last cell prints your resume numbers."""),

(CODE, """import os
os.environ.update(
    HF_USER="Mahendra99ar",                          # your Hugging Face username
    REPO="https://github.com/Mahendr99ar/trustlens.git",
    WORK="/kaggle/working",
    PER_CATEGORY="6000",                              # real reviews per category (8 categories)
    N_AI_PER_GEN="8000",                              # AI reviews from each training generator
    N_AI_HELDOUT="3000",                              # AI reviews from the held-out generator (test only)
    GEN_A="Qwen/Qwen2.5-1.5B-Instruct",
    GEN_B="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    GEN_HELD="microsoft/Phi-3.5-mini-instruct",       # never used for training: tests generalisation
    BASE="microsoft/MiniLM-L12-H384-uncased",         # small enough for the browser
    TEACHER="yangheng/deberta-v3-base-absa-v1.1",
)
# Save the settings to a file so every later cell can reload them (even after a kernel restart)
os.makedirs("/kaggle/working", exist_ok=True)
with open("/kaggle/working/settings.env", "w") as f:
    for k in ["HF_USER", "REPO", "WORK", "PER_CATEGORY", "N_AI_PER_GEN", "N_AI_HELDOUT", "GEN_A", "GEN_B", "GEN_HELD", "BASE", "TEACHER"]:
        f.write(f'export {k}="{os.environ[k]}"\\n')
print("settings ok, HF user:", os.environ["HF_USER"])
"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 1. Code + libraries
set -e
cd $WORK && rm -rf trustlens && git clone -q $REPO
pip install -q onnxscript onnx onnxruntime 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: turn on GPU T4 x2 in Session options" """),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 2. Real reviews from Amazon Reviews 2023, streamed (10-20 min)
cd $WORK/trustlens/ml
if [ -f $WORK/data/real_reviews.parquet ]; then echo "real_reviews.parquet already exists, skipping"; else
python scripts/01_fetch_reviews.py --out $WORK/data/real_reviews.parquet --per-category $PER_CATEGORY
fi"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 3a. AI-written reviews from the two training generators, in parallel (one per GPU, ~45 min)
cd $WORK/trustlens/ml
if [ -f $WORK/data/ai_a.parquet ] && [ -f $WORK/data/ai_b.parquet ]; then echo "ai_a / ai_b already exist, skipping"; else
python scripts/02_generate_ai_reviews.py --real $WORK/data/real_reviews.parquet --model $GEN_A --n $N_AI_PER_GEN \\
    --device cuda:0 --seed 1 --out $WORK/data/ai_a.parquet > $WORK/gen_a.log 2>&1 &
python scripts/02_generate_ai_reviews.py --real $WORK/data/real_reviews.parquet --model $GEN_B --n $N_AI_PER_GEN \\
    --device cuda:1 --seed 2 --out $WORK/data/ai_b.parquet > $WORK/gen_b.log 2>&1 &
# print progress every minute (the work itself writes to log files)
while [ -n "$(jobs -r)" ]; do sleep 60; echo "[$(date +%H:%M)] A: $(tr '\\r' '\\n' < $WORK/gen_a.log 2>/dev/null | tail -n 1 | cut -c1-70) | B: $(tr '\\r' '\\n' < $WORK/gen_b.log 2>/dev/null | tail -n 1 | cut -c1-70)"; done
wait
tail -n 2 $WORK/gen_a.log $WORK/gen_b.log
fi
ls -lh $WORK/data/"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 3b. AI reviews from the HELD-OUT generator (test only, ~30 min).
# Phi-3.5 (3.8B) is bigger than the others, so it writes 8 reviews at a time to fit in the T4's 15 GB.
# If a GPU still runs out of memory, that half is retried automatically with 4 at a time.
cd $WORK/trustlens/ml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HALF=$((N_AI_HELDOUT / 2))
gen_held () {   # $1 = gpu, $2 = seed, $3 = part
  OUT=$WORK/data/ai_held_$3.parquet
  if [ -f $OUT ]; then echo "$OUT already exists, skipping"; return; fi
  for BS in 8 4; do
    python scripts/02_generate_ai_reviews.py --real $WORK/data/real_reviews.parquet --model $GEN_HELD --n $HALF \\
        --batch-size $BS --max-new-tokens 256 --device cuda:$1 --seed $2 --out $OUT > $WORK/gen_h$3.log 2>&1
    if [ -f $OUT ]; then return; fi
    echo "part $3 failed with batch size $BS (see gen_h$3.log), retrying smaller"
  done
}
gen_held 0 3 1 &
gen_held 1 4 2 &
# print progress every minute (the work itself writes to log files)
while [ -n "$(jobs -r)" ]; do sleep 60; echo "[$(date +%H:%M)] H1: $(tr '\\r' '\\n' < $WORK/gen_h1.log 2>/dev/null | tail -n 1 | cut -c1-70) | H2: $(tr '\\r' '\\n' < $WORK/gen_h2.log 2>/dev/null | tail -n 1 | cut -c1-70)"; done
wait
tail -n 2 $WORK/gen_h1.log $WORK/gen_h2.log
ls -lh $WORK/data/
test -f $WORK/data/ai_held_1.parquet && test -f $WORK/data/ai_held_2.parquet && echo "OK: held-out reviews ready" """),

(CODE, """import pandas as pd, os
# 4. Look at a few generated reviews before training on them (always read your data)
import os
for _l in open("/kaggle/working/settings.env"):  # reload settings
    _k, _v = _l[len("export "):].strip().split("=", 1); os.environ[_k] = _v.strip('"')
W = os.environ["WORK"]
for f in ["ai_a", "ai_b", "ai_held_1", "ai_held_2"]:
    if not os.path.exists(f"{W}/data/{f}.parquet"):
        print(f"\\n== {f}: MISSING - run step 3b again"); continue
    d = pd.read_parquet(f"{W}/data/{f}.parquet")
    print(f"\\n== {f}: {len(d)} reviews, generator {d.generator.iloc[0]}")
    for t in d.sample(3, random_state=0).text: print("-", t[:300].replace("\\n", " "))"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 5. Leak-free splits + external test set (Salminen et al. 2022 fake reviews dataset)
cd $WORK/trustlens/ml
EXT=""
if wget -q -O $WORK/data/salminen_fake_reviews.csv "https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv" \\
   && [ -s $WORK/data/salminen_fake_reviews.csv ]; then EXT="--external $WORK/data/salminen_fake_reviews.csv"
else echo "external dataset download failed: continuing without it"; fi
HELD=$(basename $GEN_HELD)
AI=$(ls $WORK/data/ai_a.parquet $WORK/data/ai_b.parquet $WORK/data/ai_held_*.parquet 2>/dev/null)
python scripts/03_build_dataset.py --real $WORK/data/real_reviews.parquet --ai $AI \\
    --heldout-generator $HELD $EXT --out $WORK/data/splits"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 6. Train the AI-review detector (20-30 min)
cd $WORK/trustlens/ml
python scripts/04_train_detector.py --data $WORK/data/splits --base $BASE --epochs 3 --out $WORK/models/detector 2>&1 | grep -vE "Warning|warn"
"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 7. Aspect sentiment model: DeBERTa teacher labels sentences, MiniLM student learns from it (20-30 min)
cd $WORK/trustlens/ml
python scripts/05_train_aspect_model.py --reviews $WORK/data/real_reviews.parquet --teacher $TEACHER --student $BASE \\
    --out $WORK/models/aspects 2>&1 | grep -vE "Warning|warn"
"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 8. Export to ONNX + INT8 for the browser, and check accuracy is unchanged
cd $WORK/trustlens/ml
python scripts/06_export_onnx.py --model $WORK/models/detector --eval $WORK/data/splits/test.parquet --out $WORK/export/detector 2>&1 | grep -vE "Warning|warn|torch"
python scripts/06_export_onnx.py --model $WORK/models/aspects --out $WORK/export/aspects 2>&1 | grep -vE "Warning|warn|torch"
"""),

(CODE, """# 9. Publish both models to your Hugging Face account (free hosting; the extension and website download from here)
import os, subprocess
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
import os
for _l in open("/kaggle/working/settings.env"):  # reload settings
    _k, _v = _l[len("export "):].strip().split("=", 1); os.environ[_k] = _v.strip('"')
W = os.environ["WORK"]
r = subprocess.run(["python", "scripts/07_push_to_hub.py", "--user", os.environ["HF_USER"],
                      "--detector", f"{W}/export/detector", "--aspects", f"{W}/export/aspects"],
                   cwd=f"{W}/trustlens/ml", capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
assert r.returncode == 0, 'push failed: check the HF_TOKEN secret is a WRITE token and is switched on for this notebook'"""),

(CODE, """# 10. Results + resume bullet (numbers from this run)
import json, os
import os
for _l in open("/kaggle/working/settings.env"):  # reload settings
    _k, _v = _l[len("export "):].strip().split("=", 1); os.environ[_k] = _v.strip('"')
W = os.environ["WORK"]
m = json.load(open(f"{W}/models/detector/metrics.json"))
x = json.load(open(f"{W}/export/detector/export_report.json"))
a = json.load(open(f"{W}/models/aspects/metrics.json"))
rep = json.load(open(f"{W}/data/splits/report.json"))
for k in ["test", "test_unseen_generator", "test_external"]:
    if k in m: print(f"{k:>22}: AUC {m[k].get('roc_auc')}  F1 {m[k]['f1']}  caught {m[k]['recall_ai']:.0%}  false alarms {m[k]['false_alarm_rate_real']:.0%}")
if "rewrites" in m: print(f"{'LLM-paraphrased real':>22}: flagged {m['rewrites']['flagged_as_ai']:.0%}")
print("length-only baseline AUC:", rep["length_baseline_auc_before_matching"], "->", rep["length_baseline_auc_after_matching"])
print(f"browser model: {x['fp32_mb']} MB -> {x['int8_mb']} MB ({x['size_reduction_x']}x), {x['speedup_x']}x faster, agreement {x['int8_top1_agreement_with_fp32']:.1%}")
print("aspect student vs teacher:", a["history"][-1])
n_train = rep["train"]["n"]
auc_test = m["test"].get("roc_auc")
auc_unseen = m.get("test_unseen_generator", {}).get("roc_auc", "-")
fa = m["test"]["false_alarm_rate_real"]
fp32_mb, int8_mb = x["fp32_mb"], x["int8_mb"]
print()
print("Resume bullet:")
print(f"Built TrustLens, a Chrome extension + Cloudflare Workers API that flags AI-written Amazon reviews and computes a "
      f"trust-adjusted rating; fine-tuned MiniLM on {n_train:,} length-matched real/LLM reviews (ROC-AUC {auc_test} on unseen "
      f"products, {auc_unseen} on an unseen LLM, {fa:.0%} false-alarm rate) and distilled a DeBERTa ABSA teacher into a "
      f"browser-sized aspect model; INT8 ONNX ({fp32_mb} -> {int8_mb} MB) running on-device via Transformers.js.")"""),

(CODE, """%%bash
source /kaggle/working/settings.env 2>/dev/null || { echo "Run the settings cell (the first code cell) first"; exit 1; }
# 11. Download this zip from the Output panel (metrics + reports; the models are on Hugging Face)
cd $WORK && rm -f results.zip
zip -q -r results.zip models/detector/metrics.json models/aspects/metrics.json models/aspects/human_eval_sample.csv \\
    export/detector/export_report.json export/aspects/export_report.json data/splits/report.json
ls -lh results.zip"""),

(MD, """**Troubleshooting**
- *CUDA out of memory in step 3*: lower `--batch-size` (3b already retries with 4). Every step skips work whose output file already exists, so just re-run the cell.
- *A generator model fails to download*: replace it with another open instruct model (e.g. `Qwen/Qwen2.5-3B-Instruct`).
- *Step 2 cannot reach the data server*: it automatically falls back to the Hugging Face copy of the dataset.
- *Session stopped*: steps write to `/kaggle/working`, so re-run from the first step that has no output file.
- *Better aspect evaluation*: open `models/aspects/human_eval_sample.csv`, fill the `human` column (neg/neu/pos) for 100-300 rows, upload it, and re-run step 7 with `--human-labels <file>`."""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return base if kind == MD else {**base, "execution_count": None, "outputs": []}


nb = {"cells": [cell(k, s) for k, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_trustlens.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
