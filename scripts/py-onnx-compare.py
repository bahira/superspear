# Compare baseline vs SPEAR-swapped ONNX models: perplexity + throughput
# over a held-out token corpus. Run after py-onnx-swap-gelu.py.
import math
import time

import numpy as np
import onnxruntime as ort

BASE = "validation/distilgpt2.onnx"
SPEAR = "validation/distilgpt2-spear.onnx"


def build_corpus(n_tokens=3072):
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
    # tokenize via transformers (cached)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    text = ""
    i = 0
    while len(tok(text).input_ids) < n_tokens:
        text += " " + base[i % len(base)] + f" Passage {i} continues."
        i += 1
    return np.array(tok(text).input_ids[:n_tokens], dtype=np.int64)


def perplexity_and_speed(sess, ids, rounds=4):
    n = (ids.shape[0] // 512) * 512
    chunks = ids[:n].reshape(-1, 512)
    losses = []
    t0 = time.perf_counter()
    for r in range(rounds):
        for c in chunks:
            logits = sess.run(["logits"], {"input_ids": c[None, :]})[0][0]
            tgt = c[1:]
            lg = logits[:-1]
            # softmax cross-entropy in numpy
            mx = lg.max(axis=-1, keepdims=True)
            e = np.exp(lg - mx)
            sm = e / e.sum(axis=-1, keepdims=True)
            nll = -np.log(sm[np.arange(len(tgt)), tgt] + 1e-12)
            if r == 0:
                losses.append(nll.mean())
    dt = time.perf_counter() - t0
    ppl = math.exp(sum(losses) / len(losses))
    tok_s = (rounds * n) / dt
    return ppl, tok_s


def main():
    import os

    os.environ["OMP_NUM_THREADS"] = "2"
    ids = build_corpus(3072)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    base = ort.InferenceSession(BASE, so, providers=["CPUExecutionProvider"])
    spear_sess = ort.InferenceSession(SPEAR, so, providers=["CPUExecutionProvider"])

    for name, sess in [("baseline", base), ("spear", spear_sess)]:
        ppl, tok_s = perplexity_and_speed(sess, ids)
        print(f"{name:9} ppl={ppl:.4f} tok/s={tok_s:.0f}")

    p1, s1 = perplexity_and_speed(base, ids)
    p2, s2 = perplexity_and_speed(spear_sess, ids)
    print(f"\ndelta perplexite : {(p2 - p1):+.4f}")
    print(f"delta throughput : {((s2 / s1) - 1) * 100:+.1f} %")


if __name__ == "__main__":
    main()
