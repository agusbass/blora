# BloRA Roadmap — 2026 Serving Capabilities

BloRA's core is a **BCMS ⊕ LoRA** complementary-subspace adapter that merges to
a single matmul at inference. On top of that core we are building the three
serving capabilities most demanded by large-scale 2026 LLM infrastructure.

Status is reported honestly: **done / in progress / designed / not started**.
Performance numbers are only published once measured on real hardware; until
then they are marked *pending*.

---

## 1. Rank-Agnostic Serving — *in progress*

**Goal:** serve thousands of requests, each bound to a *different* adapter with
a *different* rank, in one GPU pass — no per-rank queue (the S-LoRA / Punica
problem), extended to BloRA's circulant BCMS half.

| Item | Status |
|------|--------|
| Heterogeneous-rank LoRA batched gather (padded BMM) | ✅ done |
| Per-request **batched circulant BCMS** via one FFT pass | ✅ done |
| Numerical identity vs per-adapter execution (square + GQA) | ✅ proven, rel < 1e-4 (`tests/test_serving.py`) |
| GPU throughput vs grouped / per-request loop | ✅ measured: 40k req/s, 6.8–13.9× (decode) — [BENCHMARKS](BENCHMARKS.md) |
| Fused CUDA/SYCL kernel via `uxl-fix` GEMM + scheduler | 🔵 designed |

Implementation: [`src/blora/serving.py`](../src/blora/serving.py) — `BloRAServingPack`.

**Original contribution:** the circulant (FFT) adapter half is batched *across
adapters* — each request carries its own generator, computed in a single
batched `rfft`. This is the BCMS analogue of S-LoRA's BGMV/SGMV and, to our
knowledge, has no prior art.

---

## 2. KV-Cache Memory Fusion — *in progress*

**Goal:** remove all adapter overhead from the attention KV-cache path and cut
server VRAM via fused + quantized KV. Kept as **two separately-verified
claims** (no blended marketing number):

- **(A) Adapter fusion is exact and free at cache time.** `fuse_kv` bakes the
  LoRA+BCMS delta into `k_proj`/`v_proj`, so K/V are a single matmul — zero
  per-request adapter math. Proven equal to the full forward within bf16.
- **(B) VRAM reduction is measured, not asserted.** KV-cache size is
  independent of the adapter; the saving comes from quantizing the cache:
  **int8 = 50%, int4 = 75%** (exact byte accounting in `kv_cache_bytes`). The
  ">60%" headline is honest **only at int4**. Production maps to
  `uxl-fix/real`'s paged + INT8/INT4 KV allocator.

| Item | Status |
|------|--------|
| Merge into k/v projections (no cache-time adapter cost) | ✅ done (`fuse_kv`, `fuse_kv_in_model`) |
| Exact fused-K/V == full forward | ✅ proven (`tests/test_kv.py`) |
| Reference int8/int4 KV codec + byte accounting | ✅ done; attention drift < 5% (int8) |
| Measured saving: int8 50% / int4 75% (>60% ⇒ int4) | ✅ measured (L4: bf16 0.537GB, int8 drift 0.79%) |
| Integrate `uxl-fix` paged + quantized KV (production) | 🔵 designed |

---

## 3. Per-Expert MoE Adaptation — *in progress*

**Goal:** attach a dedicated BloRA adapter to each MoE expert sub-network for
high accuracy without runtime overhead (merged per expert).

The engineering blocker is solved: the **BCMS half now supports arbitrary
rectangular shapes**. Previously it required `in % out == 0` (square or
integer-ratio GQA). The new general-rect path builds the inter-block operator
as the `(nb_out × nb_in)` top-left slice of a single circulant of size
`nb = max(nb_in, nb_out)` (FFT conv on the padded length, then slice), which
reduces *exactly* to the validated square path when `nb_in == nb_out` and
handles **up-projection (out > in)** and **non-integer ratios** (MoE MLP, e.g.
4096 ↔ 14336).

| Item | Status |
|------|--------|
| LoRA half on MoE expert linears | ✅ supported (shape-agnostic) |
| BCMS generalization to arbitrary rectangular (up + non-integer) | ✅ done (`_multi_base_forward_rect`) |
| `replace_in_model` targeting MoE expert names | ✅ done (only `block_size` divisibility required) |
| train(FFT) ≡ eval(merged) for up- and down-projection | ✅ proven (`tests/test_moe.py`, rel < 1e-2) |
| Per-expert gradient + merge on a small MoE | ✅ proven (6 expert adapters) |
| End-to-end accuracy on a real MoE (e.g. Mixtral) | ⏳ pending (GPU) |

---

## Validation principle

No capability is marked "done" without a CPU correctness test; no performance
number is published without a measured run on named hardware with the config,
seed, and environment disclosed. This is what keeps the numbers defensible.
