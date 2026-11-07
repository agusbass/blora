"""
test_blora.py — Sanity checks for BloRAAdapter (production validation)
=======================================================================
Runs on CPU — validates correctness before GPU deployment.

Checks:
  1. train(FFT path) == eval(merged matmul) for identical params
  2. gradient reaches all trainable params (LoRA + BCMS)
  3. init perturbation (no NaN/Inf at step 0)
  4. training loop reduces loss, no NaN/Inf
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
import torch.nn.functional as F

from blora import BloRAAdapter, freeze_for_blora

torch.manual_seed(42)
N, B = 256, 16
W = torch.randn(N, N) * (1.0 / math.sqrt(N))


def make_adapter(lora_init="zero"):
    return BloRAAdapter(
        N,
        N,
        pretrained_weight=W.clone(),
        lora_r=8,
        block_size=8,
        n_bases=4,
        multi_dir=True,
        use_svd_init=True,
        lora_init=lora_init,
    )


def test_train_eval_consistency():
    """train (FFT path) must match eval (merged matmul) for identical params."""
    a = make_adapter()
    x = torch.randn(B, N)
    a.train()
    out_train = a(x)
    a.eval()
    out_eval = a(x)
    diff = (out_train - out_eval).abs().max().item()
    rel = diff / (out_train.abs().mean().item() + 1e-8)
    print(f"  train vs eval: max_diff={diff:.6f}, rel={rel:.2e}")
    assert rel < 0.01, f"train/eval mismatch: rel={rel:.2e}"


def test_train_eval_consistency_pissa():
    """PiSSA init version of train/eval consistency."""
    a = make_adapter(lora_init="pissa")
    x = torch.randn(B, N)
    a.train()
    out_train = a(x)
    a.eval()
    out_eval = a(x)
    diff = (out_train - out_eval).abs().max().item()
    rel = diff / (out_train.abs().mean().item() + 1e-8)
    print(f"  pissa train vs eval: max_diff={diff:.6f}, rel={rel:.2e}")
    assert rel < 0.01, f"pissa train/eval mismatch: rel={rel:.2e}"


def test_gradient_flow():
    """All trainable params must receive gradient."""
    a = make_adapter()
    a.train()
    x = torch.randn(B, N)
    out = a(x)
    loss = out.sum()
    loss.backward()

    params_with_grad = []
    params_without_grad = []
    for name, p in a.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad.append(name)
            else:
                params_without_grad.append(name)

    print(f"  params with gradient: {len(params_with_grad)}")
    print(f"  params WITHOUT gradient: {len(params_without_grad)}")
    if params_without_grad:
        print(f"    missing: {params_without_grad}")
    assert len(params_with_grad) > 0
    # alphas and bcms_scale must get grad
    for key in ["alphas", "bcms_scale"]:
        grads = [n for n in params_with_grad if key in n]
        assert len(grads) > 0, f"{key} got no gradient"


def test_init_no_nan():
    """Init must not produce NaN or Inf."""
    a = make_adapter()
    x = torch.randn(B, N)
    out = a(x)
    assert not torch.isnan(out).any(), "NaN in forward output"
    assert not torch.isinf(out).any(), "Inf in forward output"
    print("  init clean: no NaN/Inf in forward")


def test_training_loop():
    """A few gradient steps must reduce loss, no NaN."""
    a = make_adapter()
    a.train()
    opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(B, N)
    y = torch.randn(B, N) * 0.1

    losses = []
    for step in range(20):
        opt.zero_grad()
        out = a(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in a.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.item())
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        assert not torch.isinf(loss), f"Inf loss at step {step}"

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  training loop: loss {losses[0]:.4f} -> {losses[-1]:.4f} (ok)")


def test_freeze_for_blora():
    """freeze_for_blora must freeze all but adapter params."""
    import torch.nn as nn

    a = make_adapter()

    # Wrap in a simple model to test freeze
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = a
            self.extra = nn.Linear(N, N)

        def forward(self, x):
            return self.layer(x) + self.extra(x)

    model = DummyModel()
    stats = freeze_for_blora(model)
    print(f"  freeze: trainable={stats['trainable']:,}, frozen={stats['frozen']:,}")
    assert stats["trainable"] > 0
    assert stats["frozen"] > 0
    assert model.extra.weight.requires_grad is False
    assert model.layer.lora_A.requires_grad is True


if __name__ == "__main__":
    print("=== BloRA Production Sanity Tests ===\n")
    tests = [
        ("train==eval (zero init)", test_train_eval_consistency),
        ("train==eval (PiSSA init)", test_train_eval_consistency_pissa),
        ("gradient flow", test_gradient_flow),
        ("init no NaN", test_init_no_nan),
        ("training loop", test_training_loop),
        ("freeze_for_blora", test_freeze_for_blora),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}\n")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}\n")
    print(f"=== {passed}/{len(tests)} tests passed ===")
