"""
test_moe.py — General-rectangular BCMS (Per-Expert MoE adaptation, Point 3)
==========================================================================
The BCMS half historically required in % out == 0 (square or integer-ratio
GQA). MoE MLP / expert projections violate this: up-projections (out > in) and
non-integer ratios (e.g. 4096 <-> 14336). These tests prove the general-rect
path is internally consistent (train FFT == eval merged matmul) for both
up- and down-projection, gradients flow, and BloRA can be inserted into MoE
expert linears via replace_in_model.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
import torch.nn as nn

from blora import BloRAAdapter

torch.manual_seed(0)


def _make(n_in, n_out, lora_r=4, n_bases=3):
    W = torch.randn(n_out, n_in) * (1.0 / math.sqrt(n_in))
    a = BloRAAdapter(
        n_in,
        n_out,
        pretrained_weight=W,
        lora_r=lora_r,
        block_size=8,
        n_bases=n_bases,
        multi_dir=True,
        lora_init="zero",
    )
    # activate every delta channel so the test actually exercises the math
    with torch.no_grad():
        a.lora_A.normal_(0, 0.02)
        a.lora_B.normal_(0, 0.02)
        for p in a.bases:
            p.normal_(0, 0.05)
        for p in a.rows:
            p.normal_(0, 0.05)
        for p in a.cols:
            p.normal_(0, 0.05)
        a.alphas.normal_(0, 0.2)
    return a


def _train_eq_eval(name, n_in, n_out):
    a = _make(n_in, n_out)
    assert a.general_rect, f"{name}: expected general_rect path"
    x = torch.randn(16, n_in)
    a.train()
    out_train = a(x)
    a.eval()
    out_eval = a(x)
    diff = (out_train - out_eval).abs().max().item()
    rel = diff / (out_train.abs().mean().item() + 1e-8)
    print(f"  {name}: in={n_in} out={n_out} gen_len={a.gen_len} | rel={rel:.2e}")
    assert rel < 1e-2, f"{name}: train(FFT) != eval(merged), rel={rel:.2e}"


def test_up_projection_train_eq_eval():
    """out > in (up-projection), e.g. MoE gate/up_proj."""
    _train_eq_eval("up-proj", 64, 96)


def test_noninteger_down_train_eq_eval():
    """out < in but non-integer ratio, e.g. MoE down_proj (96/64 = 1.5)."""
    _train_eq_eval("noninteger-down", 96, 64)


def test_gradient_flow_general_rect():
    """Gradients reach BCMS params (bases/rows/cols/alphas) in general-rect."""
    a = _make(64, 96)
    a.train()
    out = a(torch.randn(8, 64))
    out.sum().backward()
    for key in ["bases", "rows", "cols", "alphas"]:
        got = [
            n
            for n, p in a.named_parameters()
            if key in n and p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert got, f"{key} received no gradient"
    print("  general-rect gradient flow: bases/rows/cols/alphas all wired [OK]")


def test_replace_in_moe():
    """BloRA inserts into MoE expert linears (up + down, non-integer ratio)."""

    class Expert(nn.Module):
        def __init__(self, h, inter):
            super().__init__()
            self.w1 = nn.Linear(h, inter, bias=False)  # up   (general_rect)
            self.w3 = nn.Linear(h, inter, bias=False)  # up
            self.w2 = nn.Linear(inter, h, bias=False)  # down (general_rect)

        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([Expert(64, 96) for _ in range(2)])

        def forward(self, x):
            return sum(e(x) for e in self.experts)

    model = MoE()
    reps = BloRAAdapter.replace_in_model(
        model, target_modules=["w1", "w2", "w3"], lora_r=4, n_bases=3, verbose=False
    )
    assert len(reps) == 6, f"expected 6 expert adapters, got {len(reps)}"
    assert all(a.general_rect for a in reps.values()), "expert adapters must be general_rect"

    x = torch.randn(4, 64)
    model.train()
    y_train = model(x)
    model.eval()
    y_eval = model(x)
    assert y_train.shape == (4, 64) and y_eval.shape == (4, 64)
    assert not torch.isnan(y_train).any() and not torch.isnan(y_eval).any()
    print(f"  MoE replace: {len(reps)} expert adapters, train+eval forward OK [OK]")


if __name__ == "__main__":
    print("=== BloRA General-Rectangular / MoE Tests ===\n")
    tests = [
        ("up-projection train==eval", test_up_projection_train_eq_eval),
        ("noninteger-down train==eval", test_noninteger_down_train_eq_eval),
        ("general-rect gradient flow", test_gradient_flow_general_rect),
        ("replace in MoE", test_replace_in_moe),
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
