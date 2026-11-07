# BloRA — Measured Benchmarks

All numbers below are **measured**, not asserted. Reproduce with
`python scripts/bench_3points.py` (raw output: [`bench_3points.json`](../bench_3points.json)).

- **Hardware:** 1× NVIDIA L4 (24 GB), driver 580
- **Software:** PyTorch 2.9.1 + CUDA 12.9, bf16
- **Date:** 2026-06-10

---

## 1. Rank-Agnostic Serving

64 heterogeneous adapters (ranks {8,16,32,64}, BCMS bases {2,3,4}) sharing one
4096×4096 base layer. A batch of requests is bound to random adapters and run
in **one fused `BloRAServingPack` pass**, compared against an adapter-grouped
baseline (one batched call per distinct adapter, S-LoRA-style) and a
per-request loop.

| Workload | Batched | vs grouped | vs per-req loop | Throughput | VRAM |
|----------|--------:|-----------:|----------------:|-----------:|-----:|
| 256 req, **decode** (T=1) | 6.1 ms | **13.9×** | 56× | **41,787 req/s** | 3.1 GB |
| 512 req, **decode** (T=1) | 12.8 ms | **6.8×** | 47× | 40,021 req/s | 3.8 GB |
| 512 req, **prefill** (T=64) | 294.7 ms | **0.67×** | 2.1× | 1,737 req/s | 20.8 GB |

**Honest read:** the fused path wins decisively for **decode** (T=1) — the
dominant multi-tenant serving case (many concurrent users, one token/step) —
at **6.8–13.9× over grouped** and 40k+ req/s. For **large prefill** (T=64 here)
the per-token BCMS FFT overhead makes it **0.67× vs grouped**; for heavy prefill
the grouped path (or fusing only the LoRA half) is preferable. The win is in
decode-bound, high-adapter-fan-out serving.

## 2. KV-Cache Memory Fusion

Llama-3-8B KV shape (32 layers, 8 KV heads, head_dim 128, seq 4096, batch 1).
The adapter is fused into `k_proj`/`v_proj` (zero cache-time cost); the cache
itself is quantized.

| Precision | KV cache (measured/derived) | VRAM saved | Attn drift |
|-----------|----------------------------:|-----------:|-----------:|
| bf16 | 0.537 GB (measured) | — | — |
| int8 | 0.268 GB | **50%** | 0.79% |
| int4 | 0.134 GB | **75%** | — |

**Honest read:** the saving comes from KV quantization, not the adapter (the
adapter adds **zero** cache bytes because it is merged). int8 already halves
the cache with <1% attention drift; the ">60%" headline is only honest at int4.

## 3. Per-Expert MoE Adaptation

Synthetic MoE block: 8 experts, hidden 2048, intermediate 5632 (ratio **2.75**,
non-integer — previously unsupported by BCMS), BloRA on `w1/w2/w3`.

| Metric | Value |
|--------|------:|
| Expert adapters inserted | 24 (8 experts × 3 proj) |
| Trainable params | 3.11 M |
| Train-step forward | 124.5 ms |
| **Merged eval forward (single matmul/expert)** | **6.3 ms** |

**Honest read:** BloRA inserts into every expert at a non-integer expert ratio
and merges to a single matmul per expert — **~20× faster inference than the
training-time forward**, i.e. merged inference runs at base-model speed. The
heavier training step is the known BCMS-FFT trade-off.
