# SPEAR Validation Scripts

Empirical validation of SPEAR-discovered kernels/policies on real model internals.

## Scripts

### `kv_real_validate.py`
KV-cache eviction policy validation on a single model at a single context length
(SEQ_CAP=320). Hooks a HuggingFace causal LM with eager attention, captures real
attention patterns per layer per prompt, converts them to the SPEAR kv-cache
feature format (A = normalized past-attention mass, P = position, S = sink flag
for first KV_SINK=4 positions, R = recency flag for last 40), then compares:

| policy | scores keys by |
|---|---|
| `spear` | `4.5*S + A + R` (auto-generated champion from `kv_policy_generated.py`) |
| `h2o` | past attention mass (H2O heuristic) |
| `streaming` | first keep/2 sinks + last keep/2 recent |
| `random` | random (seeded) |
| `oracle` | future attention itself (upper bound) |

Retention is scored against the **second half** of the sequence (future queries),
features built from the **first half** — mirrors the SPEAR kv_cache task protocol.
Output: `kv-real-results.json`.

### `kv_multiscale.py`
Same protocol, swept across **model scales and context lengths**: for each model
in `--models` and each cap in `[64, 128, 256, 320]`, runs the full validation
with kept budget `max(8, cap // 8)`. Output:
- `kv-multiscale-results.json`: `{model: {cap: {policy: {mean, median, min, max, n}}}}`
- printed markdown table per model (rows = policy, columns = cap, cells = mean %).

Default models: `distilgpt2` only (cached locally, runs offline).

### `gelu_swap_perplexity.py`
Swaps distilgpt2's GELU activations for the discovered algebraic kernel
(`gelu_policy_generated.spear_gelu`) and measures perplexity delta + CPU
throughput before/after, plus a kernel microbench on real captured activations.

## Install & run

```powershell
pip install torch transformers sentencepiece   # or: pip install -r requirements.txt

# baseline single-scale validation
python validation\kv_real_validate.py --model distilgpt2 --num-prompts 12 --keep 40

# multiscale sweep (distilgpt2 only, ~1 min on 2 cores)
python validation\kv_multiscale.py

# multiscale with more models
python validation\kv_multiscale.py --models distilgpt2,Llama-3.2-1B

# GELU swap perplexity
python validation\gelu_swap_perplexity.py --model distilgpt2
```

## Hardware notes

Tested fine on an i7-7660U (2 cores / 4 threads, no GPU): scripts pin
`torch.set_num_threads(2)` and run eager attention on CPU. distilgpt2 full
multiscale sweep completes in well under 2 minutes. Larger models scale roughly
linearly in parameters x layers; expect several minutes per extra small model.

## Adding Llama-3.2-1B later

`meta-llama/Llama-3.2-1B` is a **gated repo**: accept the license on its
HuggingFace page, then export a token before running:

```powershell
$env:HF_TOKEN = "hf_..."
python validation\kv_multiscale.py --models distilgpt2,Llama-3.2-1B
```

Transformers-compat notes for Llama-3.2 attention outputs:
- eager attention (`attn_implementation="eager"`) is supported; `output_attentions=True`
  returns the standard tuple-of-`(B, H, S, S)` tensors, so the head-mean / past-future
  split works unchanged.
- Llama-3.2 uses grouped-query attention (32 query heads, 4 KV heads). The captured
  attention tensor has shape `(B, num_query_heads=32, S, S)` — head-averaging over dim 0
  is still valid since the script averages all heads uniformly.
- No API mismatch found between transformers 5.15 and this usage pattern; if an
  older/newer transformers renames `output_attentions`, it affects both models equally.
