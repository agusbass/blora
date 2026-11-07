# BloRA — BCMS ⊕ LoRA Complementary-Subspace Adapter

**Parameter-efficient fine-tuning combining BCMS circulant structure with LoRA low-rank decomposition, separated by PiSSA-style SVD subspace split.**

BloRA assigns each mechanism to a distinct SVD subspace so they fill gaps rather than fight over the same directions. At inference, all weights merge into a **single matrix** — faster than LoRA's two-matmul path.

## Three production serving capabilities for 2026-scale inference

BloRA is built around the three LoRA capabilities large-scale 2026 inference
needs most. Each has a CPU-proven implementation in this repo; GPU throughput
and end-to-end accuracy are reported only as measured (see [docs/ROADMAP.md](docs/ROADMAP.md)).

| # | Capability | What it does | Status |
|---|------------|--------------|--------|
| 1 | **Rank-Agnostic Serving** | thousands of requests, each a different adapter/rank, in **one vectorized pass** (no per-rank queue) | ✅ proven identical to per-adapter; **40k req/s, 6.8–13.9× vs grouped** (decode) |
| 2 | **KV-Cache Memory Fusion** | adapter fused into k/v_proj (zero cache-time cost) + int8/int4 KV | ✅ **50% / 75% VRAM**, int8 attn drift 0.79% |
| 3 | **Per-Expert MoE Adaptation** | a BloRA adapter per MoE expert; BCMS generalized to any expert shape | ✅ inserts at ratio 2.75; **merged inference ~20× faster than train fwd** |

Measured on 1× NVIDIA L4 — full numbers and honest caveats (incl. where the
fused path *loses*, e.g. large prefill) in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.
Correctness is proven on CPU (hardware-independent); performance numbers are
only published as measured on named hardware.

## How It Works

```
TRAIN:  out = W_frozen·x + (B@A)·lora_scale·x + BCMS_delta·bcms_scale·x
EVAL:   out = W_merged·x      (single matmul)
```

**Subspace Separation via PiSSA SVD:**
```
W = U S Vh
LoRA  = top-r  : B@A = U[:,:r] S[:r] Vh[:r]   (high-energy → PPL)
BCMS  = residual: SVD-init from remainder       (structure LoRA misses)
W_frozen = W − top-r   (model preserved at init)
```

## Quick Start

```python
from blora import BloRAAdapter, freeze_for_blora
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
model.enable_input_require_grads()

# Replace attention projections with BloRA adapters
adapters = BloRAAdapter.replace_in_model(
    model,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_r=16,
    n_bases=4,
    multi_dir=True,
    lora_init="zero",   # or "pissa" for PiSSA subspace split
)

# Freeze base model, keep only adapter params trainable
freeze_for_blora(model)

# Train normally — same API as LoRA
# At model.eval(), weights auto-merge for single-matmul inference
```

## Installation

```bash
pip install -e .
pip install -e ".[dev]"       # with pytest, ruff, mypy
pip install -e ".[benchmark]" # with datasets, matplotlib
```

## Run Tests

```bash
cd F:\1.rayalabs.dev\blora
python tests/test_blora.py
```

## Params (N=4096, lora_r=16, k=4, hvd)

| Component | Formula | Count |
|-----------|---------|-------|
| LoRA A + B | 16·4096·2 | 131,072 |
| BCMS bases | 4·512 | 2,048 |
| BCMS rows | 4·512 | 2,048 |
| BCMS cols | 4·512 | 2,048 |
| BCMS alphas + scale | 4 + 1 | 5 |
| **Total** | | **~137K** |

vs LoRA r=32: 262K params → BloRA is ~2x smaller with complementary structure.

## Rank-Agnostic Batched Serving

Serve many requests, each bound to a **different adapter with a different rank**
(and a different number of BCMS bases), in **one vectorized pass** — no
per-rank queue:

```python
from blora import BloRAAdapter, BloRAServingPack
import torch

# adapters share one base layer; ranks/bases may differ
pack = BloRAServingPack([adapter_r4, adapter_r16, adapter_r8])

x = torch.randn(B, T, in_features)          # B requests
adapter_ids = torch.tensor([0, 2, 1, ...])  # which adapter per request
out = pack(x, adapter_ids)                   # single batched forward
```

The LoRA half is a padded batched matmul; the circulant BCMS half is batched
*across adapters* with a single FFT. This is numerically identical to running
each adapter separately (`tests/test_serving.py`, rel < 1e-4). See
[docs/ROADMAP.md](docs/ROADMAP.md) for the full 2026 serving roadmap
(rank-agnostic serving · KV-cache fusion · per-expert MoE).

## Performance notes (honest)

- **Inference:** after merge, BloRA is a single matmul — **same speed as the
  base model / LoRA**. This is the headline.
- **Training:** the BCMS (FFT) half makes the training step ~2–3× slower than
  plain LoRA per step. BloRA trades a heavier train step for a complementary
  subspace and identical-cost inference. Numbers are published only as measured.

## Requirements

- PyTorch ≥ 2.0
- CUDA GPU recommended (falls back to CPU; correctness tests run on CPU)

## Development

```bash
make dev-install   # editable install + dev tools + pre-commit
make check         # ruff + mypy + pytest (what CI runs)
```

## License

MIT License — see [LICENSE](LICENSE). © 2026 PT Raya Laksana Teknologi ([ralafs.com](https://ralafs.com)).

**Open core.** The core library is MIT-licensed and free to use. Production /
enterprise features (rank-agnostic serving, KV-cache fusion, and per-expert MoE
adaptation at scale) plus commercial support are available under a separate
enterprise license — contact demo@ralafs.com.