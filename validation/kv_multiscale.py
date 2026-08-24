#!/usr/bin/env python3
"""
KV-cache eviction validation across MODEL SCALES and CONTEXT LENGTHS.

Same protocol as kv_real_validate.py (real attentions -> A/P/S/R features ->
policy comparison against the future half), repeated for each model in
--models and each context cap in --caps.

Models:
  --models is a comma-separated HF id list. Default "distilgpt2" runs today
  with no downloads (cached). To add Llama-3.2-1B you need an HF token and
  gated-repo access:
      set HF_TOKEN=hf_...   (or $env:HF_TOKEN="hf_..." in PowerShell)
      python kv_multiscale.py --models distilgpt2,Llama-3.2-1B
  (meta-llama/Llama-3.2-1B requires accepting the license on huggingface.co.)

Usage:
  python kv_multiscale.py                          # distilgpt2, all caps
  python kv_multiscale.py --models distilgpt2,Llama-3.2-1B
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_policy_generated import spear_kv_policy  # noqa: E402

CAPS = [64, 128, 256, 320]
KV_SINK = 4


def build_prompts(num_prompts):
    base = [
        "The history of computing is filled with surprising turns, and",
        "In a small town at the edge of the mountains, the winter",
        "Modern machine learning systems rely on matrix multiplications that",
        "The recipe calls for butter, flour, three eggs, and a generous",
        "When the spacecraft finally reached orbit, the crew discovered",
        "Economic policy often changes slowly, but every so often a crisis",
        "She opened the old drawer and found letters nobody had read since",
        "Deep inside the forest, the river splits into two branches that",
        "The committee reviewed every proposal carefully before announcing",
        "A single candle lit the room while the storm outside grew louder,",
        "Quantum computers promise speedups for problems whose structure",
        "Every morning the fishermen checked the tide tables before",
    ]
    prompts = []
    for i in range(num_prompts):
        p = base[i % len(base)]
        reps = 3 + (i % 6)
        prompts.append((p + " ") * reps)
    return prompts


def run_model(model_name, num_prompts, caps):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval()

    model_results = {}
    for cap in caps:
        keep = max(8, cap // 8)
        results = {k: [] for k in ["spear", "h2o", "streaming", "random", "oracle"]}
        samples = 0

        for pi, text in enumerate(build_prompts(num_prompts)):
            ids = tok(text, return_tensors="pt").input_ids[0][:cap]
            S_len = ids.shape[0]
            if S_len < 16:
                continue
            with torch.no_grad():
                out = model(ids.unsqueeze(0), output_attentions=True)

            half = S_len // 2
            for li, att in enumerate(out.attentions):
                a = att[0].mean(dim=0)  # (S_q, S_k) head-averaged
                past = a[:half].mean(dim=0)
                future = a[half:].mean(dim=0)

                total_past = float(past.sum())
                if total_past <= 0:
                    continue
                A_feat = np.array([(float(m) / total_past) * cap for m in past])
                P_feat = np.arange(S_len, dtype=float) / max(1, S_len - 1)
                S_feat = np.array([1.0 if i < KV_SINK else 0.0 for i in range(S_len)])
                R_feat = np.array([1.0 if i >= S_len - 40 else 0.0 for i in range(S_len)])

                future_total = float(future.sum())
                if future_total <= 0:
                    continue

                def retain(scores):
                    order = sorted(range(S_len), key=lambda i: -scores[i])
                    kept = order[:keep]
                    return 100.0 * sum(float(future[i]) for i in kept) / future_total

                rng = torch.Generator().manual_seed(1234 + pi * 100 + li)
                rand_scores = torch.rand(S_len, generator=rng).tolist()
                stream_scores = [
                    1.0 if (i < keep // 2 or i >= S_len - keep // 2) else 0.0
                    for i in range(S_len)
                ]

                results["spear"].append(retain(spear_kv_policy(S_feat, A_feat, R_feat)))
                results["h2o"].append(retain(A_feat))
                results["streaming"].append(retain(stream_scores))
                results["random"].append(retain(rand_scores))
                results["oracle"].append(retain([float(f) for f in future]))
                samples += 1

            print(f"[{model_name} cap={cap}] prompt {pi+1}/{num_prompts} seq={S_len} samples={samples}")

        def summarize(vals):
            vals = sorted(vals)
            mid = vals[len(vals) // 2]
            return {"mean": sum(vals) / len(vals), "median": mid,
                    "min": vals[0], "max": vals[-1], "n": len(vals)}

        model_results[str(cap)] = {k: summarize(v) for k, v in results.items()}
        print(f"[{model_name} cap={cap}] done, {samples} samples, keep={keep}")

    del model
    return model_results


def markdown_table(model_name, model_results, caps):
    lines = [f"\n=== RETAINED FUTURE-ATTENTION MASS (%) — {model_name} ===", ""]
    header = "| policy | " + " | ".join(f"cap={c}" for c in caps) + " |"
    sep = "|" + "---|" * (len(caps) + 1)
    lines.append(header)
    lines.append(sep)
    for k in ["spear", "h2o", "streaming", "random", "oracle"]:
        row = [k.ljust(9)]
        for c in caps:
            s = model_results.get(str(c), {}).get(k)
            row.append(f"{s['mean']:6.2f}" if s else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="distilgpt2",
                    help="comma-separated HF ids; Llama-3.2-1B needs HF_TOKEN + gated access")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--caps", default=",".join(map(str, CAPS)))
    ap.add_argument("--out", default="kv-multiscale-results.json")
    args = ap.parse_args()

    torch.set_num_threads(2)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    caps = [int(c) for c in args.caps.split(",")]

    all_results = {}
    for m in models:
        print(f"\n### model: {m} ###")
        all_results[m] = run_model(m, args.num_prompts, caps)
        print(markdown_table(m, all_results[m], caps))

    with open(args.out, "w") as f:
        json.dump({"models": all_results, "caps": caps}, f, indent=2)
    print(f"\nsauvegardé: {args.out}")


if __name__ == "__main__":
    main()
