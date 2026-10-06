"""Export a fine-tuned classifier to ONNX, quantize it to INT8, check accuracy is preserved, and lay the files out
the way Transformers.js expects (so the browser extension / website can load it straight from the Hugging Face Hub).

    python scripts/06_export_onnx.py --model models/detector --eval data/splits/test.parquet --out export/detector
    python scripts/06_export_onnx.py --model models/aspects --out export/aspects

Output: <out>/{config.json, tokenizer.json, ...}, <out>/onnx/model.onnx (fp32), <out>/onnx/model_quantized.onnx (int8),
<out>/export_report.json (sizes, CPU latency, fp32-vs-int8 agreement).
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def mb(path):
    return round(os.path.getsize(path) / 2**20, 1)


def run_onnx(sess, tok, texts, max_len):
    names = {i.name for i in sess.get_inputs()}
    probs = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="np")
        feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
        logits = sess.run(None, feeds)[0][0]
        e = np.exp(logits - logits.max())
        probs.append(e / e.sum())
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=None, help="parquet with a text column, to measure fp32 vs int8 agreement")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    try:  # eager attention exports cleanly with dynamic sequence length
        model = AutoModelForSequenceClassification.from_pretrained(args.model, attn_implementation="eager").eval()
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    os.makedirs(os.path.join(args.out, "onnx"), exist_ok=True)
    tok.save_pretrained(args.out)
    model.config.save_pretrained(args.out)
    for extra in ("trustlens.json", "metrics.json"):
        if os.path.exists(os.path.join(args.model, extra)):
            shutil.copy(os.path.join(args.model, extra), args.out)
    if not os.path.exists(os.path.join(args.out, "tokenizer.json")):
        raise SystemExit("no tokenizer.json written: use a model with a fast tokenizer")

    input_names = [n for n in tok.model_input_names if n in ("input_ids", "attention_mask", "token_type_ids")]
    dummy = tok(["a short example review", "another, slightly longer example review"], padding=True, return_tensors="pt")
    fp32 = os.path.join(args.out, "onnx", "model.onnx")
    axes = {n: {0: "batch", 1: "sequence"} for n in input_names}
    axes["logits"] = {0: "batch"}

    class WithTypes(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    class NoTypes(WithTypes):
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    input_names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in input_names]
    wrapper = (WithTypes if "token_type_ids" in input_names else NoTypes)(model)
    args_tuple = tuple(dummy[n] for n in input_names)
    try:  # torch.export-based exporter: exact parity with dynamic batch/sequence
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence", max=512)
        torch.onnx.export(wrapper, args_tuple, fp32, input_names=input_names, output_names=["logits"],
                          dynamic_shapes={n: {0: batch, 1: seq} for n in input_names}, opset_version=18, dynamo=True)
    except Exception as e:  # older PyTorch: classic TorchScript exporter
        print(f"dynamo export failed ({type(e).__name__}); using the TorchScript exporter")
        kw = dict(input_names=input_names, output_names=["logits"], dynamic_axes=axes, opset_version=args.opset)
        try:
            torch.onnx.export(wrapper, args_tuple, fp32, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(wrapper, args_tuple, fp32, **kw)
    # Drop stale shape annotations the exporter leaves behind (they confuse the quantizer), keep weights inline
    import onnx
    m = onnx.load(fp32, load_external_data=True)
    del m.graph.value_info[:]
    onnx.save(m, fp32, save_as_external_data=False)
    for leftover in (fp32 + ".data",):
        if os.path.exists(leftover):
            os.remove(leftover)
    int8 = os.path.join(args.out, "onnx", "model_quantized.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # like a single browser thread
    s32 = ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8, so, providers=["CPUExecutionProvider"])

    texts = ["Battery lasts two days and the sound is great, but the strap broke after a month."]
    if args.eval and os.path.exists(args.eval):
        texts = pd.read_parquet(args.eval).text.head(args.eval_n).tolist()
    with torch.no_grad():
        ref = np.stack([torch.softmax(model(**tok(t, truncation=True, max_length=args.max_len, return_tensors="pt")).logits[0], -1).numpy()
                        for t in texts[:32]])
    p32 = run_onnx(s32, tok, texts, args.max_len)
    diff = float(np.abs(p32[:32] - ref).max())
    assert diff < 1e-3, f"fp32 ONNX differs from PyTorch by {diff}"

    t0 = time.perf_counter()
    p8 = run_onnx(s8, tok, texts, args.max_len)
    ms8 = (time.perf_counter() - t0) * 1000 / len(texts)
    t0 = time.perf_counter()
    run_onnx(s32, tok, texts, args.max_len)
    ms32 = (time.perf_counter() - t0) * 1000 / len(texts)

    report = {"n_eval_texts": len(texts), "fp32_mb": mb(fp32), "int8_mb": mb(int8),
              "size_reduction_x": round(mb(fp32) / max(mb(int8), 0.1), 2),
              "cpu_1thread_ms_per_text_fp32": round(ms32, 2), "cpu_1thread_ms_per_text_int8": round(ms8, 2),
              "speedup_x": round(ms32 / max(ms8, 1e-6), 2),
              "int8_top1_agreement_with_fp32": round(float((p8.argmax(1) == p32.argmax(1)).mean()), 4),
              "max_prob_diff": round(float(np.abs(p8 - p32).max()), 4), "labels": model.config.id2label}
    with open(os.path.join(args.out, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
