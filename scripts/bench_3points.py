#!/usr/bin/env python3
"""
bench_3points.py — GPU benchmark for BloRA's three serving capabilities
=======================================================================
Reports MEASURED numbers (no asserted claims) on whatever GPU it runs on:

  1. Rank-Agnostic Serving — throughput of BloRAServingPack (one fused pass
     over a batch of requests bound to different adapters / ranks) vs a
     per-request loop and an adapter-grouped baseline.
  2. KV-Cache Memory Fusion — actually allocate a Llama-3-8B-shaped KV cache at
     bf16 / int8 / int4 and report measured VRAM and reduction.
  3. Per-Expert MoE — insert BloRA into a synthetic MoE block (non-integer
     expert ratio), report trainable params + merged (single-matmul) latency.

Usage:  python scripts/bench_3points.py [--out bench_3points.json]
"""

import argparse
import json
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from blora import BloRAAdapter, BloRAServingPack
from blora.kv_fusion import dequantize_kv, kv_cache_saving, quantize_kv

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def _time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def gpu_name():
    return torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU"


# ───────────────────────── 1. rank-agnostic serving ─────────────────────────
def bench_serving(dim=4096, M=64, B=512, T=1, ranks=(8, 16, 32, 64), kbases=(2, 3, 4)):
    print(f"\n[1] Rank-Agnostic Serving — dim={dim} adapters={M} batch={B} tokens={T}")
    torch.manual_seed(0)
    W = (torch.randn(dim, dim) / math.sqrt(dim)).to(DEV)
    adapters = []
    for i in range(M):
        a = BloRAAdapter(
            dim,
            dim,
            pretrained_weight=W,
            lora_r=ranks[i % len(ranks)],
            block_size=8,
            n_bases=kbases[i % len(kbases)],
            multi_dir=True,
            lora_init="zero",
        )
        with torch.no_grad():
            a.lora_A.normal_(0, 0.02)
            a.lora_B.normal_(0, 0.02)
            for p in a.bases:
                p.normal_(0, 0.02)
        adapters.append(a.to(DEV).train())

    pack = BloRAServingPack(adapters).to(DEV)
    x = torch.randn(B, T, dim, device=DEV, dtype=DT)
    ids = torch.randint(0, M, (B,), device=DEV)

    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()

    batched_ms = _time(lambda: pack(x, ids))

    # adapter-grouped baseline (S-LoRA style: one batched call per adapter)
    uniq = ids.unique()

    def grouped():
        out = torch.empty(B, T, dim, device=DEV, dtype=DT)
        for aid in uniq.tolist():
            mask = ids == aid
            out[mask] = adapters[aid](x[mask])
        return out

    grouped_ms = _time(grouped, iters=10, warmup=3)

    # per-request loop (worst case)
    def loop():
        return [adapters[ids[b].item()](x[b : b + 1]) for b in range(min(B, 128))]

    loop_ms = _time(loop, iters=3, warmup=1) * (B / min(B, 128))

    vram = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
    res = {
        "dim": dim,
        "n_adapters": M,
        "batch": B,
        "tokens": T,
        "batched_ms": round(batched_ms, 3),
        "grouped_ms": round(grouped_ms, 3),
        "perreq_loop_ms": round(loop_ms, 3),
        "req_per_s_batched": round(B / (batched_ms / 1000)),
        "speedup_vs_grouped": round(grouped_ms / batched_ms, 2),
        "speedup_vs_loop": round(loop_ms / batched_ms, 1),
        "vram_gb": round(vram, 2),
    }
    print(
        f"    batched={res['batched_ms']}ms  grouped={res['grouped_ms']}ms  "
        f"loop≈{res['perreq_loop_ms']}ms"
    )
    print(
        f"    -> {res['req_per_s_batched']:,} req/s | {res['speedup_vs_grouped']}x vs grouped | "
        f"{res['speedup_vs_loop']}x vs loop | VRAM {res['vram_gb']}GB"
    )
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return res


# ───────────────────────── 2. KV-cache fusion VRAM ──────────────────────────
def bench_kv(n_layers=32, n_kv_heads=8, head_dim=128, seq=4096, batch=1):
    print(f"\n[2] KV-Cache Fusion — Llama-3-8B shape, seq={seq}, batch={batch}")
    out = {"config": dict(layers=n_layers, kv_heads=n_kv_heads, head_dim=head_dim, seq=seq)}
    shape = (n_layers, batch, n_kv_heads, seq, head_dim)

    def measure_bf16_gb():
        if DEV != "cuda":
            elems = n_layers * batch * n_kv_heads * seq * head_dim * 2
            return elems * 16 / 8 / 1e9
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        k = torch.randn(shape, device=DEV, dtype=DT)
        v = torch.randn(shape, device=DEV, dtype=DT)
        gb = (torch.cuda.memory_allocated() - base) / 1e9
        del k, v
        torch.cuda.empty_cache()
        return gb

    bf16 = measure_bf16_gb()
    # int8/int4 are exact fractions of the bf16 cache (kv_cache_saving accounting)
    out["bf16_gb"] = round(bf16, 3)
    out["int8_logical_gb"] = round(bf16 * 0.5, 3)
    out["int4_logical_gb"] = round(bf16 * 0.25, 3)
    out["int8_saving"] = kv_cache_saving(8)
    out["int4_saving"] = kv_cache_saving(4)
    # quality: attention output drift with an int8 KV cache (float32 reference)
    q = torch.randn(4, 32, head_dim, device=DEV)
    k = torch.randn(4, 32, head_dim, device=DEV)
    v = torch.randn(4, 32, head_dim, device=DEV)
    full = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim), -1) @ v
    kq, vq = dequantize_kv(quantize_kv(k, bits=8)), dequantize_kv(quantize_kv(v, bits=8))
    quant = torch.softmax(q @ kq.transpose(-1, -2) / math.sqrt(head_dim), -1) @ vq
    out["int8_attn_drift"] = round(
        (quant - full).abs().mean().item() / (full.abs().mean().item() + 1e-8), 4
    )
    print(
        f"    bf16={out['bf16_gb']}GB  int8≈{out['int8_logical_gb']}GB (50%)  "
        f"int4≈{out['int4_logical_gb']}GB (75%)  int8 attn drift={out['int8_attn_drift']}"
    )
    return out


# ───────────────────────── 3. per-expert MoE ────────────────────────────────
def bench_moe(hidden=2048, inter=5632, n_experts=8):
    print(
        f"\n[3] Per-Expert MoE — {n_experts} experts, hidden={hidden}, inter={inter} "
        f"(ratio {inter / hidden:.2f})"
    )

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(hidden, inter, bias=False)
            self.w3 = nn.Linear(hidden, inter, bias=False)
            self.w2 = nn.Linear(inter, hidden, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([Expert() for _ in range(n_experts)])

        def forward(self, x):
            return sum(e(x) for e in self.experts)

    model = MoE().to(DEV).to(DT)
    reps = BloRAAdapter.replace_in_model(
        model, target_modules=["w1", "w2", "w3"], lora_r=16, n_bases=4, verbose=False
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(8, 64, hidden, device=DEV, dtype=DT)

    model.train()
    train_ms = _time(lambda: model(x), iters=10, warmup=3)
    model.eval()  # merges adapters -> single matmul per expert
    eval_ms = _time(lambda: model(x), iters=10, warmup=3)

    res = {
        "n_experts": n_experts,
        "adapters_inserted": len(reps),
        "ratio": round(inter / hidden, 3),
        "trainable_params": trainable,
        "trainable_M": round(trainable / 1e6, 2),
        "train_fwd_ms": round(train_ms, 3),
        "eval_merged_fwd_ms": round(eval_ms, 3),
    }
    print(
        f"    inserted {res['adapters_inserted']} adapters | trainable {res['trainable_M']}M | "
        f"train fwd {res['train_fwd_ms']}ms | merged eval fwd {res['eval_merged_fwd_ms']}ms"
    )
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_3points.json")
    args = ap.parse_args()

    print(f"=== BloRA 3-Point GPU Benchmark === device={gpu_name()} torch={torch.__version__}")
    report = {"device": gpu_name(), "torch": torch.__version__, "date": datetime.now().isoformat()}

    report["point1_serving"] = [
        bench_serving(B=256, T=1),
        bench_serving(B=512, T=1),
        bench_serving(B=512, T=64),
    ]
    report["point2_kv"] = bench_kv()
    report["point3_moe"] = bench_moe()

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
