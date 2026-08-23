#!/usr/bin/env python3
"""
KV-cache eviction validation on REAL transformer attention traces.

Hooks a HuggingFace causal LM (default distilgpt2), captures true attention
patterns across layers and prompts, converts them into the SPEAR kv-cache
feature format (A/P/S/R per key position), then compares eviction policies:

  - spear      : the discovered champion  4.5*S + A + R   (auto-generated)
  - h2o        : keep top-K by past attention mass (H2O heuristic)
  - streaming  : keep first K/2 sinks + last K/2 recent positions
  - random     : random K positions
  - oracle     : top-K by the future attention itself (upper bound)

Protocol mirrors the SPEAR kv_cache task: features are computed from the
FIRST HALF of the sequence (past queries), retention is scored against the
SECOND HALF (future queries). Kept budget = K positions out of S total.

Usage:
  python kv_real_validate.py --model distilgpt2 --num-prompts 12 --keep 40
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_policy_generated import spear_kv_policy  # noqa: E402

SEQ_CAP = 320
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--keep", type=int, default=40)
    ap.add_argument("--out", default="kv-real-results.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()

    policies = {
        "spear": lambda A, P, S, R: spear_kv_policy(S, A, R),
        "h2o": lambda A, P, S, R: A,
        "streaming": lambda A, P, S, R: [1.0 if (i < args.keep // 2 or i >= SEQ_CAP - args.keep // 2) else 0.0 for i, _ in enumerate(A)],
        "random": None,  # handled separately for reproducibility
        "oracle": None,
    }

    results = {k: [] for k in ["spear", "h2o", "streaming", "random", "oracle"]}
    samples = 0

    prompts = build_prompts(args.num_prompts)
    for pi, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids[0][:SEQ_CAP]
        S_len = ids.shape[0]
        with torch.no_grad():
            out = model(ids.unsqueeze(0), output_attentions=True)

        # attentions: tuple[L] of (1, H, S, S)
        half = S_len // 2
        for li, att in enumerate(out.attentions):
            a = att[0].mean(dim=0)  # (S_q, S_k) head-averaged
            past = a[:half].mean(dim=0)  # (S_k,) past-query view
            future = a[half:].mean(dim=0)  # (S_k,) future-query view

            total_past = float(past.sum())
            if total_past <= 0:
                continue
            A_feat = np.array([(float(m) / total_past) * SEQ_CAP for m in past])
            P_feat = np.arange(S_len, dtype=float) / max(1, S_len - 1)
            S_feat = np.array([1.0 if i < KV_SINK else 0.0 for i in range(S_len)])
            R_feat = np.array([1.0 if i >= S_len - 40 else 0.0 for i in range(S_len)])

            future_total = float(future.sum())
            if future_total <= 0:
                continue

            def retain(scores):
                order = sorted(range(S_len), key=lambda i: -scores[i])
                kept = order[: args.keep]
                return 100.0 * sum(float(future[i]) for i in kept) / future_total

            rng = torch.Generator().manual_seed(1234 + pi * 100 + li)
            rand_scores = torch.rand(S_len, generator=rng).tolist()

            results["spear"].append(retain(policies["spear"](A_feat, P_feat, S_feat, R_feat)))
            results["h2o"].append(retain(A_feat))
            results["streaming"].append(retain(policies["streaming"](A_feat, P_feat, S_feat, R_feat)))
            results["random"].append(retain(rand_scores))
            results["oracle"].append(retain([float(f) for f in future]))
            samples += 1

        print(f"[prompt {pi+1}/{len(prompts)}] seq={S_len} layers={len(out.attentions)} cumulated samples={samples}")

    def summarize(vals):
        vals = sorted(vals)
        mid = vals[len(vals) // 2]
        return {"mean": sum(vals) / len(vals), "median": mid, "min": vals[0], "max": vals[-1], "n": len(vals)}

    summary = {k: summarize(v) for k, v in results.items()}
    print("\n=== RETAINED FUTURE-ATTENTION MASS (%) — REAL ATTENTIONS ===")
    for k in ["spear", "h2o", "streaming", "random", "oracle"]:
        s = summary[k]
        print(f"{k.ljust(10)} mean {s['mean']:6.2f}  median {s['median']:6.2f}  [{s['min']:.2f} .. {s['max']:.2f}]  n={s['n']}")

    spear_mean = summary["spear"]["mean"]
    h2o_mean = summary["h2o"]["mean"]
    print(f"\nspear vs h2o : {'+' if spear_mean >= h2o_mean else ''}{(spear_mean - h2o_mean):.2f} pts")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "keep": args.keep, "samples": samples, "summary": summary}, f, indent=2)
    print(f"sauvegardé: {args.out}")


if __name__ == "__main__":
    main()
