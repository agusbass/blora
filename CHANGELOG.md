# Changelog

All notable changes to BloRA are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **KV-Cache Memory Fusion** (`blora.kv_fusion`) — `fuse_kv` / `fuse_kv_in_model`
  bake the adapter into `k_proj`/`v_proj` (K/V become a single matmul, zero
  cache-time adapter cost, proven exact). Plus a reference affine int8/int4 KV
  codec and exact byte accounting (`kv_cache_bytes`, `kv_cache_saving`):
  int8 = 50%, int4 = 75% VRAM saved (">60%" only at int4). Verified in
  `tests/test_kv.py` (fused exactness, quant roundtrip, attention drift < 5%).
- **General-rectangular BCMS** (`_multi_base_forward_rect`) — the BCMS half now
  supports arbitrary `(in, out)` shapes: up-projection (out > in) and
  non-integer ratios, i.e. MoE MLP / expert projections (e.g. 4096 ↔ 14336).
  The inter-block operator is the `(nb_out × nb_in)` slice of one circulant of
  size `max(nb_in, nb_out)`; it reduces exactly to the validated square path.
  `replace_in_model` now only requires `block_size` to divide both dims, so
  BloRA inserts into MoE experts. Proven train≡eval + gradient + MoE-insertion
  in `tests/test_moe.py`.
- `blora.serving.BloRAServingPack` — **rank-agnostic batched multi-adapter
  serving**. Serves a batch of requests where each request is bound to a
  different adapter (heterogeneous LoRA rank *and* number of BCMS bases) in a
  single vectorized pass: padded gather for the LoRA half, and a per-request
  batched FFT for the circulant BCMS half (no per-request Python loop).
- `tests/test_serving.py` — proves the packed batched path is numerically
  identical (rel < 1e-4) to running each adapter on its own slice, for both
  square and rectangular (GQA) layers.
- Engineering baseline for open release: CI (ruff + mypy + pytest on
  py3.10–3.12), pre-commit hooks, Makefile, `.gitignore`, `LICENSE`,
  `SECURITY.md`, `CONTRIBUTING.md`, and `docs/ROADMAP.md`.

### Changed
- Test suites are now ASCII-only so they run on Windows consoles (cp1252) as
  well as Linux; primary runner is `pytest`.

## [1.0.0] — 2026

### Added
- `BloRAAdapter` — BCMS ⊕ LoRA complementary-subspace PEFT adapter with
  PiSSA-style SVD subspace split; single-matmul merged inference.
- `freeze_for_blora` — freeze base, train adapter params only.
- Self-contained BCMS v6.3 core (`_bcms_core`): magic-square kernels,
  cached randomized SVD init, FFT circulant multiply, multi-base forward.
- `replace_in_model` with 4-bit / 8-bit (bitsandbytes) dequant support.
- Train≡eval consistency, gradient-flow, and SFT sanity tests.
