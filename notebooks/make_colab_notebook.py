"""Generates notebooks/colab_trustlens.ipynb: the same pipeline as the Kaggle notebook, for Google Colab (1 GPU).

Differences from Kaggle: one T4 GPU (generators run one after another), work saved to Google Drive so a
disconnect does not lose progress, HF token read from Colab Secrets. Every step skips work that is already done.
"""
import json
import os

MD, CODE = "markdown", "code"
LOAD_SH = 'source /content/settings.env 2>/dev/null || { echo "Run the settings cell (first code cell) first"; exit 1; }\n'
LOAD_PY = ('import os\nfor _l in open("/content/settings.env"):  # reload settings\n'
           '    _k, _v = _l[len("export "):].strip().split("=", 1); os.environ[_k] = _v.strip(\'"\')\n')


def progress(logs):
    parts = " | ".join(f"{n}: $(tr '\\r' '\\n' < $WORK/{f} 2>/dev/null | tail -n 1 | cut -c1-80)" for n, f in logs)
    return ('while [ -n "$(jobs -r)" ]; do sleep 60; echo "[$(date +%H:%M)] ' + parts + '"; done\nwait\n')


def gen(name, model, n, seed, extra=""):
    out = f"$WORK/data/{name}.parquet"
    return (f'if [ -f {out} ]; then echo "{name} already done, skipping"; else\n'
            f'python scripts/02_generate_ai_reviews.py --real $WORK/data/real_reviews.parquet --model {model} --n {n} \\\n'
            f'    {extra}--device cuda:0 --seed {seed} --out {out} > $WORK/gen_{name}.log 2>&1 &\n'
            + progress([(name, f"gen_{name}.log")]) +
            f'tail -n 2 $WORK/gen_{name}.log\nfi\n')


cells = [
(MD, """# TrustLens: train the models on Google Colab (free T4)

**Before running**
1. *Runtime → Change runtime type → **T4 GPU** → Save*.
2. Left bar → **🔑 Secrets** → *Add new secret*: Name **`HF_TOKEN`**, Value = your Hugging Face **write** token. Turn **Notebook access** ON.
3. Run the cells **one by one, top to bottom**. Keep this tab open (Colab disconnects idle tabs).

All work is saved in Google Drive (`MyDrive/trustlens_work`). **If Colab disconnects:** reconnect, run the first two code cells again, then re-run from the step that stopped. Finished steps are skipped automatically.

Total time on one T4: about 4–5 hours."""),

(CODE, """# Settings + Google Drive (saves progress)
import os
from google.colab import drive
drive.mount("/content/drive")
os.environ.update(
    HF_USER="Mahendra99ar",                          # your Hugging Face username
    REPO="https://github.com/Mahendr99ar/trustlens.git",
    CODE="/content/trustlens",                       # code (re-downloaded each session)
    WORK="/content/drive/MyDrive/trustlens_work",    # data + models (kept in Drive)
    PER_CATEGORY="6000",
    N_AI_PER_GEN="8000",
    N_AI_HELDOUT="1500",                             # enough for a test set; Phi-3.5 is slow on one GPU
    GEN_A="Qwen/Qwen2.5-1.5B-Instruct",
    GEN_B="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    GEN_HELD="microsoft/Phi-3.5-mini-instruct",      # never used for training: tests generalisation
    BASE="microsoft/MiniLM-L12-H384-uncased",
    TEACHER="yangheng/deberta-v3-base-absa-v1.1",
)
os.makedirs(os.environ["WORK"] + "/data", exist_ok=True)
with open("/content/settings.env", "w") as f:
    for k in ["HF_USER", "REPO", "CODE", "WORK", "PER_CATEGORY", "N_AI_PER_GEN", "N_AI_HELDOUT",
              "GEN_A", "GEN_B", "GEN_HELD", "BASE", "TEACHER"]:
        f.write(f'export {k}="{os.environ[k]}"\\n')
print("settings ok, HF user:", os.environ["HF_USER"], "| work folder:", os.environ["WORK"])"""),

(CODE, "%%bash\n" + LOAD_SH + """# 1. Code + libraries (run again after every reconnect)
set -e
if [ ! -d $CODE ]; then git clone -q $REPO $CODE; fi
pip install -q onnxscript onnx onnxruntime 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime -> Change runtime type -> T4 GPU"
ls -lh $WORK/data/"""),

(CODE, "%%bash\n" + LOAD_SH + """# 2. Real reviews from Amazon Reviews 2023 (10-20 min)
cd $CODE/ml
if [ -f $WORK/data/real_reviews.parquet ]; then echo "real_reviews already done, skipping"; else
python scripts/01_fetch_reviews.py --out /content/real_reviews.parquet --per-category $PER_CATEGORY \\
  && cp /content/real_reviews.parquet $WORK/data/real_reviews.parquet
fi"""),

(CODE, "%%bash\n" + LOAD_SH + "# 3a. AI reviews, generator A (Qwen, ~40 min). Prints progress every minute.\ncd $CODE/ml\n"
 + gen("ai_a", "$GEN_A", "$N_AI_PER_GEN", 1)),

(CODE, "%%bash\n" + LOAD_SH + "# 3b. AI reviews, generator B (SmolLM2, ~40 min)\ncd $CODE/ml\n"
 + gen("ai_b", "$GEN_B", "$N_AI_PER_GEN", 2)),

(CODE, "%%bash\n" + LOAD_SH + """# 3c. AI reviews from the HELD-OUT generator (Phi-3.5, test only, ~50 min)
# Phi-3.5 is bigger, so it writes 8 reviews at a time (4 if memory runs out).
cd $CODE/ml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$WORK/data/ai_held.parquet
if [ -f $OUT ]; then echo "ai_held already done, skipping"; else
for BS in 8 4; do
  python scripts/02_generate_ai_reviews.py --real $WORK/data/real_reviews.parquet --model $GEN_HELD --n $N_AI_HELDOUT \\
      --batch-size $BS --max-new-tokens 256 --device cuda:0 --seed 3 --out $OUT > $WORK/gen_ai_held.log 2>&1 &
""" + progress([("held", "gen_ai_held.log")]) + """  if [ -f $OUT ]; then break; fi
  echo "failed with batch size $BS (see gen_ai_held.log), retrying smaller"
done
tail -n 2 $WORK/gen_ai_held.log
fi
test -f $OUT && echo "OK: held-out reviews ready" """),

(CODE, LOAD_PY + """import pandas as pd
# 4. Look at a few generated reviews before training on them
W = os.environ["WORK"]
for f in ["ai_a", "ai_b", "ai_held"]:
    p = f"{W}/data/{f}.parquet"
    if not os.path.exists(p):
        print(f"\\n== {f}: MISSING - run its step 3 cell again"); continue
    d = pd.read_parquet(p)
    print(f"\\n== {f}: {len(d)} reviews, generator {d.generator.iloc[0]}")
    for t in d.sample(3, random_state=0).text: print("-", t[:300].replace("\\n", " "))"""),

(CODE, "%%bash\n" + LOAD_SH + """# 5. Leak-free splits + external test set (Salminen et al. 2022)
cd $CODE/ml
if [ -f $WORK/data/splits/report.json ]; then echo "splits already done, skipping"; exit 0; fi
EXT=""
if wget -q -O $WORK/data/salminen_fake_reviews.csv "https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv" \\
   && [ -s $WORK/data/salminen_fake_reviews.csv ]; then EXT="--external $WORK/data/salminen_fake_reviews.csv"
else echo "external dataset download failed: continuing without it"; fi
python scripts/03_build_dataset.py --real $WORK/data/real_reviews.parquet \\
    --ai $WORK/data/ai_a.parquet $WORK/data/ai_b.parquet $WORK/data/ai_held.parquet \\
    --heldout-generator $(basename $GEN_HELD) $EXT --out $WORK/data/splits"""),

(CODE, "%%bash\n" + LOAD_SH + """# 6. Train the AI-review detector (30-40 min)
cd $CODE/ml
if [ -f $WORK/models/detector/metrics.json ]; then echo "detector already trained, skipping"; exit 0; fi
python scripts/04_train_detector.py --data $WORK/data/splits --base $BASE --epochs 3 --out $WORK/models/detector 2>&1 | grep -vE "Warning|warn"
"""),

(CODE, "%%bash\n" + LOAD_SH + """# 7. Aspect sentiment model: DeBERTa teacher -> MiniLM student (30-40 min)
cd $CODE/ml
if [ -f $WORK/models/aspects/metrics.json ]; then echo "aspect model already trained, skipping"; exit 0; fi
python scripts/05_train_aspect_model.py --reviews $WORK/data/real_reviews.parquet --teacher $TEACHER --student $BASE \\
    --out $WORK/models/aspects 2>&1 | grep -vE "Warning|warn"
"""),

(CODE, "%%bash\n" + LOAD_SH + """# 8. Export to ONNX + INT8 for the browser, and check accuracy is unchanged
cd $CODE/ml
[ -f $WORK/export/detector/export_report.json ] || python scripts/06_export_onnx.py --model $WORK/models/detector \\
    --eval $WORK/data/splits/test.parquet --out $WORK/export/detector 2>&1 | grep -vE "Warning|warn|torch"
[ -f $WORK/export/aspects/export_report.json ] || python scripts/06_export_onnx.py --model $WORK/models/aspects \\
    --out $WORK/export/aspects 2>&1 | grep -vE "Warning|warn|torch"
ls $WORK/export/*/"""),

(CODE, LOAD_PY + """# 9. Publish both models to your Hugging Face account
import subprocess
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN").strip()
W = os.environ["WORK"]
r = subprocess.run(["python", "scripts/07_push_to_hub.py", "--user", os.environ["HF_USER"],
                    "--detector", f"{W}/export/detector", "--aspects", f"{W}/export/aspects"],
                   cwd=f"{os.environ['CODE']}/ml", capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:])
assert r.returncode == 0, 'push failed: check the HF_TOKEN secret is a WRITE token with Notebook access ON'"""),
]

# results + zip cells are the same as Kaggle, with paths from settings
cells += [
(CODE, LOAD_PY + """import json
# 10. Results + resume bullet (numbers from this run)
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

(CODE, "%%bash\n" + LOAD_SH + """# 11. results.zip is saved in Drive: MyDrive/trustlens_work/results.zip
cd $WORK && rm -f results.zip
zip -q -r results.zip models/detector/metrics.json models/aspects/metrics.json models/aspects/human_eval_sample.csv \\
    export/detector/export_report.json export/aspects/export_report.json data/splits/report.json
ls -lh results.zip"""),

(MD, """**Troubleshooting**
- *Disconnected / runtime reset*: reconnect, run the first two code cells, then run from the step that stopped. Finished steps are skipped.
- *"You cannot currently connect to a GPU"*: Colab's free GPU limit for today is used up. Try again after some hours (your progress in Drive is safe).
- *CUDA out of memory*: re-run the cell; step 3c already retries with a smaller batch.
- *A step was interrupted half-way*: delete that step's half-finished output folder in Drive (e.g. `trustlens_work/models/detector`) and re-run it."""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return base if kind == MD else {**base, "execution_count": None, "outputs": []}


nb = {"cells": [cell(k, s) for k, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_trustlens.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
