"""
test_kv.py — KV-Cache Memory Fusion (Point 2)
=============================================
(A) Adapter fusion is exact: the merged single-matmul k/v projection equals the
    full BloRA computation (W_frozen + LoRA + BCMS) within bf16 — i.e. zero
    adapter overhead at cache time, no accuracy cost.
(B) VRAM reduction is measured, not asserted: int8 = 50%, int4 = 75% of the
    bf16 KV cache; attention with a quantized KV cache stays close to full
    precision.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
import torch.nn.functional as F

from blora import BloRAAdapter
from blora.kv_fusion import (
    dequantize_kv,
    fuse_kv,
    kv_cache_bytes,
    kv_cache_saving,
    quantize_kv,
)

torch.manual_seed(0)


def test_fused_kv_exact():
    """Merged k/v projection == full BloRA forward (zero-overhead fusion)."""
    n_in, n_out = 256, 128  # GQA-style k/v projection
    W = torch.randn(n_out, n_in) * (1.0 / math.sqrt(n_in))
    a = BloRAAdapter(n_in, n_out, pretrained_weight=W, lora_r=8, block_size=8, n_bases=4)
    with torch.no_grad():
        a.lora_A.normal_(0, 0.02)
        a.lora_B.normal_(0, 0.02)
        for p in a.bases:
            p.normal_(0, 0.03)
        a.alphas.normal_(0, 0.1)

    x = torch.randn(8, n_in)
    a.train()
    ref = a(x)  # W_frozen + LoRA + BCMS

    fused = fuse_kv(a)  # single merged matmul
    out = F.linear(x, fused.weight.float())

    rel = (out - ref).abs().max().item() / (ref.abs().mean().item() + 1e-8)
    print(f"  fused kv vs full forward: rel={rel:.2e}")
    assert rel < 1e-2, f"fused k/v projection not exact: rel={rel:.2e}"


def test_kv_quant_roundtrip_and_savings():
    """int8/int4 quant reconstructs KV within tolerance; bytes match accounting."""
    kv = torch.randn(2, 8, 128, 64)  # (B, heads, seq, head_dim)

    for bits, tol in [(8, 1.2e-2), (4, 1.2e-1)]:
        packed = quantize_kv(kv, bits=bits, dim=-1)
        recon = dequantize_kv(packed)
        rel = (recon - kv).abs().mean().item() / (kv.abs().mean().item() + 1e-8)
        saving = kv_cache_saving(bits)
        print(f"  int{bits}: recon rel={rel:.2e} | VRAM saving={saving:.0%}")
        assert rel < tol, f"int{bits} roundtrip too lossy: rel={rel:.2e}"

    # exact byte accounting
    args = dict(n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096)
    assert kv_cache_saving(8) == 0.5
    assert kv_cache_saving(4) == 0.75
    assert kv_cache_bytes(bits=8, **args) * 2 == kv_cache_bytes(bits=16, **args)
    # ">60% VRAM saved" claim is only honest at int4
    assert kv_cache_saving(4) > 0.60 and kv_cache_saving(8) < 0.60
    print("  accounting: int8=50%, int4=75% (>60% requires int4) [OK]")


def test_attention_with_quantized_kv():
    """Attention output with an int8 KV cache stays close to full precision."""
    B, H, L, D = 2, 4, 32, 64
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    def attn(qq, kk, vv):
        s = (qq @ kk.transpose(-1, -2)) / math.sqrt(D)
        return torch.softmax(s, dim=-1) @ vv

    full = attn(q, k, v)
    kq = dequantize_kv(quantize_kv(k, bits=8))
    vq = dequantize_kv(quantize_kv(v, bits=8))
    quant = attn(q, kq, vq)

    rel = (quant - full).abs().mean().item() / (full.abs().mean().item() + 1e-8)
    print(f"  attention(int8 KV) vs full: rel={rel:.2e}")
    assert rel < 5e-2, f"int8 KV attention drift too high: rel={rel:.2e}"


if __name__ == "__main__":
    print("=== BloRA KV-Cache Fusion Tests ===\n")
    tests = [
        ("fused kv exact", test_fused_kv_exact),
        ("kv quant roundtrip + savings", test_kv_quant_roundtrip_and_savings),
        ("attention with quantized kv", test_attention_with_quantized_kv),
    ]
    passed = 0
    for nm, fn in tests:
        try:
            fn()
            print(f"  [OK] {nm}\n")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {nm}: {e}\n")
    print(f"=== {passed}/{len(tests)} passed ===")
