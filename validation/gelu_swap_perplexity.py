#!/usr/bin/env python3
"""
SPEAR-fast edge validation: swap distilgpt2's GELU for the discovered
algebraic kernel, measure perplexity delta and throughput on CPU.

Protocol:
  1. Build a held-out text corpus (~3000 tokens).
  2. Baseline: stock model -> perplexity + tokens/sec over several rounds.
  3. Capture REAL activation inputs at the MLP act points (one probe pass).
  4. Swap every block's mlp.act with the SPEAR algebraic GELU.
  5. Swapped pass: perplexity + tokens/sec again.
  6. Kernel microbench: exact GELU vs SPEAR GELU on the captured REAL inputs.

Usage: python gelu_swap_perplexity.py [--model distilgpt2]
"""
import argparse
import math
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gelu_policy_generated import spear_gelu

CHUNK = 512


def build_corpus(tok, n_tokens):
    base = [
        "The history of computing is filled with surprising turns, and every decade seems to rewrite",
        "In a small town at the edge of the mountains, winter arrives early and stays long, so the",
        "Modern machine learning systems rely on matrix multiplications whose cost dominates every",
        "She opened the old drawer and found letters nobody had read since the summer the mill",
        "Economic policy changes slowly, but a crisis can compress ten years of debate into a single",
        "When the spacecraft reached orbit the crew discovered that half their instruments had",
        "Deep inside the forest the river splits into two branches that never meet again, and the",
        "Every morning the fishermen checked the tide tables before launching their small boats,",
    ]
    text = ""
    i = 0
    while len(tok(text).input_ids) < n_tokens:
        text += " " + base[i % len(base)] + f" Passage number {i} continues the record."
        i += 1
    return tok(text, return_tensors="pt").input_ids[0][:n_tokens]


def perplexity_and_speed(model, ids):
    n = (ids.shape[0] // CHUNK) * CHUNK
    chunks = ids[:n].view(-1, CHUNK)
    losses = []
    t0 = time.perf_counter()
    rounds = 4
    for _ in range(rounds):
        for c in chunks.unsqueeze(0):
            logits = model(c).logits[:, :-1, :]
            tgt = c[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)
            )
            if _ == 0:
                losses.append(loss.item())
    dt = time.perf_counter() - t0
    ppl = math.exp(sum(losses) / len(losses))
    tok_s = (rounds * n) / dt
    return ppl, tok_s


class SpearGELU(torch.nn.Module):
    def forward(self, x):
        return spear_gelu(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--tokens", type=int, default=3072)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    torch.set_num_threads(2)

    ids = build_corpus(tok, args.tokens)

    # ---- baseline ----
    ppl_ref, tok_ref = perplexity_and_speed(model, ids)

    # ---- capture real activation inputs (probe pass with hooks) ----
    acts = []
    def hook(module, inp, out):
        acts.append(inp[0].detach().flatten())
    handles = []
    for h in model.transformer.h:
        handles.append(h.mlp.act.register_forward_hook(hook))
    with torch.no_grad():
        n = (ids.shape[0] // CHUNK) * CHUNK
        model(ids[:n].view(-1, CHUNK))
    for hd in handles:
        hd.remove()
    real_inputs = torch.cat(acts)

    # ---- swap ----
    for h in model.transformer.h:
        h.mlp.act = SpearGELU()
    ppl_sp, tok_sp = perplexity_and_speed(model, ids)

    # ---- kernel microbench on REAL distributions ----
    with torch.no_grad():
        ref_k = torch.nn.functional.gelu(real_inputs)
        t0 = time.perf_counter()
        for _ in range(20):
            ref_k = torch.nn.functional.gelu(real_inputs)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(20):
            sp_k = spear_gelu(real_inputs)
        t_sp = time.perf_counter() - t0
    max_err = float((ref_k - sp_k).abs().max())

    print("\n=== SPEAR-FAST EDGE — distilgpt2, CPU, 2 threads ===")
    print(f"perplexite : stock {ppl_ref:.4f} -> spear {ppl_sp:.4f}   (delta {(ppl_sp - ppl_ref):+.4f})")
    print(f"throughput : stock {tok_ref:.0f} tok/s -> spear {tok_sp:.0f} tok/s   ({((tok_sp/tok_ref)-1)*100:+.1f} %)")
    print(f"kernel GELU sur activations reelles ({real_inputs.numel()} elems):")
    print(f"  torch F.gelu {t_ref*1000:.1f} ms/round -> spear {t_sp*1000:.1f} ms/round  (x{(t_ref/max(t_sp,1e-9)):.2f})")
    print(f"  erreur max abs: {max_err:.4f}")

    with open("gelu-swap-results.json", "w") as f:
        import json
        json.dump({
            "model": args.model,
            "tokens": args.tokens,
            "perplexity_stock": ppl_ref,
            "perplexity_spear": ppl_sp,
            "toks_sec_stock": tok_ref,
            "toks_sec_spear": tok_sp,
            "kernel_ms_round_ref": t_ref * 1000,
            "kernel_ms_round_spear": t_sp * 1000,
            "kernel_max_abs_err": max_err,
        }, f, indent=2)


if __name__ == "__main__":
    main()
