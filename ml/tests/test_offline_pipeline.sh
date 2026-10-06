#!/usr/bin/env bash
# Runs scripts 02-06 end to end on CPU with tiny random models (no internet). Proves the code paths work;
# the numbers it prints mean nothing.
#   bash tests/test_offline_pipeline.sh /path/to/fake_reviews_dataset.csv
set -euo pipefail
cd "$(dirname "$0")/.."
CSV=${1:?path to Salminen fake reviews CSV}
T=$(mktemp -d)
python tests/make_tiny_models.py "$T/m" "$CSV"
python scripts/02_generate_ai_reviews.py --real "$T/m/real_reviews.parquet" --model "$T/m/tiny-gen" --n 16 \
    --batch-size 8 --max-new-tokens 24 --device cpu --out "$T/ai_gen_test.parquet"
python scripts/03_build_dataset.py --real "$T/m/real_reviews.parquet" \
    --ai "$T/m/ai_gen-a.parquet" "$T/m/ai_gen-b.parquet" "$T/m/ai_gen-held.parquet" \
    --heldout-generator gen-held --external "$CSV" --external-n 200 --out "$T/splits"
python scripts/04_train_detector.py --data "$T/splits" --base "$T/m/tiny-bert" --epochs 1 --batch-size 16 \
    --max-len 64 --max-train 300 --out "$T/models/detector"
python scripts/05_train_aspect_model.py --reviews "$T/m/real_reviews.parquet" --teacher "$T/m/tiny-absa-teacher" \
    --student "$T/m/tiny-bert" --max-pairs 400 --min-confidence 0.0 --epochs 1 --out "$T/models/aspects"
python scripts/06_export_onnx.py --model "$T/models/detector" --eval "$T/splits/test.parquet" --eval-n 40 \
    --max-len 64 --out "$T/export/detector"
python scripts/06_export_onnx.py --model "$T/models/aspects" --max-len 64 --out "$T/export/aspects"
python -c "from trustlens_ml import aspect_inputs, trust_adjusted_rating; \
print(aspect_inputs('Battery lasts two days. The strap broke after a month, but the price was fair.')); \
print(trust_adjusted_rating([5,5,1,4],[0.9,0.95,0.1,0.2]))"
echo "EXPORT_DIR=$T/export"
echo "OFFLINE PIPELINE PASSED"
