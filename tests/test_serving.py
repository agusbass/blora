"""
test_serving.py — Rank-agnostic batched serving correctness
===========================================================
Proves BloRAServingPack (one batched pass, heterogeneous rank + n_bases per
request) is numerically identical to running each adapter on its own slice.
If this holds, multi-tenant serving costs one GPU pass with zero accuracy loss.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch

from blora import BloRAAdapter
from blora.serving import BloRAServingPack

torch.manual_seed(0)


def _randomize(a: BloRAAdapter):
    """Give the adapter non-trivial trained params so every delta is active."""
    with torch.no_grad():
        a.lora_A.normal_(0, 0.02)
        a.lora_B.normal_(0, 0.02)
        for p in a.bases:
            p.normal_(0, 0.02)
        for p in a.rows:
            p.normal_(0, 0.02)
        for p in a.cols:
            p.normal_(0, 0.02)
        a.alphas.normal_(0, 0.1)
        a.bcms_scale.fill_(0.7)
    a.train()  # train-mode math (W_frozen + lora + bcms)
    return a


def _build_pack(n_in, n_out, ranks, bases_counts):
    W = torch.randn(n_out, n_in) * (1.0 / math.sqrt(n_in))
    adapters = []
    for r, k in zip(ranks, bases_counts):
        a = BloRAAdapter(
            n_in,
            n_out,
            pretrained_weight=W.clone(),
            lora_r=r,
            block_size=8,
            n_bases=k,
            multi_dir=True,
            use_svd_init=True,
            lora_init="zero",
        )
        adapters.append(_randomize(a))
    return adapters, BloRAServingPack(adapters)


def _check(name, n_in, n_out, ranks, bases_counts):
    adapters, pack = _build_pack(n_in, n_out, ranks, bases_counts)
    M = len(adapters)
    B, T = 12, 5
    ids = torch.randint(0, M, (B,))
    x = torch.randn(B, T, n_in)

    batched = pack(x, ids)  # one pass

    ref = torch.empty_like(batched)
    for b in range(B):
        ref[b] = adapters[ids[b].item()](x[b : b + 1])[0]  # per-request reference

    diff = (batched - ref).abs().max().item()
    rel = diff / (ref.abs().mean().item() + 1e-9)
    print(f"  {name}: M={M} ranks={ranks} k={bases_counts} | max_diff={diff:.2e} rel={rel:.2e}")
    assert rel < 1e-4, f"{name}: batched != per-adapter (rel={rel:.2e})"


def test_square_heterogeneous():
    """Square layer (q/o_proj), different rank AND different n_bases per adapter."""
    _check("square", 256, 256, ranks=[4, 8, 16, 8], bases_counts=[2, 4, 3, 4])


def test_rectangular_gqa():
    """Rectangular GQA layer (k/v_proj, out<in), heterogeneous rank."""
    _check("rect-gqa", 256, 128, ranks=[4, 16, 8], bases_counts=[4, 4, 2])


def test_single_pass_no_python_loop():
    """Sanity: pack.forward issues no per-request python loop over adapters."""
    import inspect

    src = inspect.getsource(BloRAServingPack.forward)
    assert "for " not in src, "serving forward must be loop-free over requests"
    print("  loop-free: forward has no per-request python loop [OK]")


if __name__ == "__main__":
    print("=== BloRA Rank-Agnostic Serving Tests ===\n")
    tests = [
        ("square heterogeneous", test_square_heterogeneous),
        ("rectangular GQA", test_rectangular_gqa),
        ("single-pass (no loop)", test_single_pass_no_python_loop),
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
