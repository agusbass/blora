"""
adapter.py — BloRA: BCMS + LoRA, complementary-subspace adapter
================================================================
BloRA = BCMS ⊕ LoRA, split by SVD energy so each mechanism owns a
disjoint subspace instead of competing over the same directions:

    Training:  out = W_frozen·x + (B@A)·lora_scale·x + BCMS_delta·bcms_scale·x
    Eval:      out = W_merged·x      (single matmul)

Subspace separation via PiSSA-style SVD init (no init corruption):
    W = U S Vh
    LoRA  = top-r  : B@A = U[:,:r] S[:r] Vh[:r]   (high-energy directions → PPL)
    W_frozen = W − top-r   (residual stays frozen)        ⇒ W_eff = W at init ✓
    BCMS base init = SVD-guided from the RESIDUAL          (structure LoRA misses)
    alphas init 0  ⇒ BCMS_delta = 0 at init ⇒ W_eff = W EXACTLY at step 0.

Rationale: LoRA's low-rank basis and BCMS's circulant-Kronecker basis are
"blind" to different kinds of structure — what one can't represent, the
other may capture. The PiSSA split assigns each mechanism to a distinct
subspace from step 0 so they fill gaps rather than fight over the same
directions; `ortho_loss()` is available as a regularizer if they start to
overlap during training.

BCMS half reuses the validated v6.3 machinery (hvd multi-dir, FFT circulant,
rectangular GQA) — NO R_MS, NO scipy, train≡eval verified.

Params (N=4096, lora_r=16, k=4, hvd):
    LoRA: 16·4096·2          = 131,072
    BCMS: 4·512·3 + 4 + 1    ≈   6,149   (bases+rows+cols+alphas+scale)
    TOTAL ≈ 137K  (vs LoRA r=32: 262K → ~2x smaller, + complementary structure)

Proprietary technology. Rayalabs.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from blora._bcms_core import (
    _clear_svd_cache,
    _make_ms_rotated,
    _multi_base_forward,
    _multi_base_forward_rect,
    _svd_init_base,
)


class BloRAAdapter(nn.Module):
    # buffer type annotations (mypy widens nn.Module buffers to Tensor|Module|None)
    W_frozen: torch.Tensor
    b_frozen: "torch.Tensor | None"
    _W_merged: "torch.Tensor | None"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pretrained_weight: Optional[torch.Tensor] = None,
        pretrained_bias: Optional[torch.Tensor] = None,
        lora_r: int = 16,
        lora_alpha: Optional[float] = None,
        block_size: int = 8,
        n_bases: int = 4,
        multi_dir: bool = True,
        use_svd_init: bool = True,
        noise_eps: float = 0.05,
        svd_rank: int = 64,
        lora_init: str = "zero",
    ):  # "zero" (B=0, preserves model) | "pissa"
        super().__init__()
        assert in_features % block_size == 0 and out_features % block_size == 0, (
            "block_size must divide both in_features and out_features"
        )

        self.n_in, self.n_out = in_features, out_features
        self.block_size = block_size
        self.nb_in = in_features // block_size
        self.nb_out = out_features // block_size
        self.ratio = self.nb_in // self.nb_out if self.nb_out else 1
        self.is_rect = out_features != in_features
        # general_rect: arbitrary rectangular — up-projection (out > in) or
        # non-integer block ratio (e.g. MoE MLP 4096<->14336). Square and
        # integer-ratio GQA keep the validated circulant / group-sum path.
        self.general_rect = in_features % out_features != 0
        # circulant generator length: max(nb_in, nb_out) in general-rect mode
        # so the (nb_out x nb_in) operator is a slice of one square circulant.
        self.gen_len = max(self.nb_in, self.nb_out) if self.general_rect else self.nb_in
        assert not (self.general_rect and not multi_dir), (
            "general rectangular (MoE / up-projection) requires multi_dir=True"
        )
        self.n_bases = n_bases
        self.multi_dir = multi_dir
        self.lora_r = lora_r
        # PiSSA init makes B@A == top-r exactly, so lora_scale stays 1.0.
        self.lora_scale = 1.0 if lora_alpha is None else (lora_alpha / lora_r)

        # ── LoRA half ────────────────────────────────────────────────────
        self.lora_A = nn.Parameter(torch.zeros(lora_r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))

        # ── BCMS half (generator length gen_len; v6.3 hvd) ───────────────
        self.bases = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.gen_len)) for _ in range(n_bases)]
        )
        self.alphas = nn.Parameter(torch.zeros(n_bases))
        self.bcms_scale = nn.Parameter(torch.tensor(1.0))
        if multi_dir:
            self.rows = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.nb_out)) for _ in range(n_bases)]
            )
            self.cols = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.nb_in)) for _ in range(n_bases)]
            )
        for i in range(n_bases):
            self.register_buffer(f"MS_{i}", _make_ms_rotated(block_size, rotate=i))

        # ── Init: "zero" (standard LoRA B=0, model preserved) or "pissa" (top-r split)
        if use_svd_init and pretrained_weight is not None and lora_init == "pissa":
            # PiSSA: LoRA=top-r, frozen=residual, BCMS=residual structure.
            W = pretrained_weight.float()
            r = min(lora_r, min(W.shape))
            U, S, V = torch.svd_lowrank(W, q=max(r, 8), niter=4)
            U, S, V = U[:, :r], S[:r], V[:, :r]
            sqrt_s = S.clamp_min(0).sqrt()
            lora_A = (sqrt_s.unsqueeze(1) * V.t()).contiguous()
            lora_B = (U * sqrt_s.unsqueeze(0)).contiguous()
            W_residual = W - lora_B @ lora_A
            self.lora_A.data = lora_A.cpu()
            self.lora_B.data = lora_B.cpu()
            ref = W_residual
            for i, base in enumerate(self.bases):
                base.data = (
                    _svd_init_base(
                        ref,
                        getattr(self, f"MS_{i}").to(ref.device),
                        self.nb_in,
                        block_size,
                        rank=svd_rank,
                        noise_eps=noise_eps * (i + 1) * 0.5,
                    )
                    .cpu()
                    .to(base.dtype)
                    if not self.is_rect
                    else torch.randn(self.gen_len) * 0.01
                )
            W_frozen = W_residual
            _clear_svd_cache()
        elif use_svd_init and pretrained_weight is not None:
            # "zero": LoRA B=0 (delta=0 at init → model preserved → HellaSwag safe).
            # BCMS bases SVD-init from full W; LoRA + BCMS learn complementary structure.
            W = pretrained_weight.float()
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            if not self.is_rect:
                for i, base in enumerate(self.bases):
                    base.data = (
                        _svd_init_base(
                            W,
                            getattr(self, f"MS_{i}").to(W.device),
                            self.nb_in,
                            block_size,
                            rank=svd_rank,
                            noise_eps=noise_eps * (i + 1) * 0.5,
                        )
                        .cpu()
                        .to(base.dtype)
                    )
            else:
                for base in self.bases:
                    nn.init.normal_(base, std=0.01)
            W_frozen = W
            _clear_svd_cache()
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            for base in self.bases:
                nn.init.normal_(base, std=0.01)
            W_frozen = (
                pretrained_weight.float()
                if pretrained_weight is not None
                else torch.zeros(out_features, in_features)
            )

        self.register_buffer("W_frozen", W_frozen.to(torch.bfloat16))
        if pretrained_bias is not None:
            self.register_buffer("b_frozen", pretrained_bias.to(torch.bfloat16))
        else:
            self.register_buffer("b_frozen", None)

        # Small nonzero alphas: BCMS engages from step 0 (rows/cols/bases all get
        # gradient). Init perturbation stays ~0.6% of Wx (validated harmless in v6.3).
        nn.init.constant_(self.alphas, 0.1 / n_bases)
        self._W_merged = None

    # ── forward ──────────────────────────────────────────────────────────
    def _bcms_delta(self, x: torch.Tensor) -> torch.Tensor:
        ms = [getattr(self, f"MS_{i}") for i in range(self.n_bases)]
        if self.general_rect:
            # arbitrary rectangular (MoE / up-projection): general-rect path
            # requires multi_dir rows/cols.
            return _multi_base_forward_rect(
                x,
                list(self.bases),
                ms,
                self.alphas,
                self.nb_in,
                self.nb_out,
                self.block_size,
                rows=list(self.rows),
                cols=list(self.cols),
            )
        rows = list(self.rows) if self.multi_dir else None
        cols = list(self.cols) if self.multi_dir else None
        return _multi_base_forward(
            x,
            list(self.bases),
            ms,
            self.alphas,
            self.nb_in,
            self.block_size,
            rows=rows,
            cols=cols,
            ring_pq=None,
            nb_out=self.nb_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        if not self.training and self._W_merged is not None:
            W = self._W_merged.to(device=x.device, dtype=dt)
            b = self.b_frozen.to(x.device, dt) if self.b_frozen is not None else None
            return F.linear(x, W, b)

        b = self.b_frozen.to(x.device, dt) if self.b_frozen is not None else None
        out = F.linear(x, self.W_frozen.to(x.device, dt), b)
        A = self.lora_A.to(x.device, dt)
        B = self.lora_B.to(x.device, dt)
        out = out + (x @ A.t() @ B.t()) * self.lora_scale
        out = out + self._bcms_delta(x) * self.bcms_scale.to(x.device, dt)
        return out

    # ── eval merge (LoRA + BCMS into single weight; no scipy) ─────────────
    @torch.no_grad()
    def _rebuild_eval_cache(self):
        device = self.lora_A.device
        nb_in, nb_out = self.nb_in, self.nb_out
        W = self.W_frozen.float().to(device)
        W = W + (self.lora_B.float() @ self.lora_A.float()) * self.lora_scale

        gl = self.gen_len
        idx = (
            torch.arange(gl, device=device)[:, None] - torch.arange(gl, device=device)[None, :]
        ) % gl
        W_delta = torch.zeros(self.n_out, self.n_in, device=device)
        for i, base in enumerate(self.bases):
            C = base.detach().float()[idx]  # (gl, gl) full circulant
            if self.general_rect:
                # slice the square circulant to (nb_out, nb_in) — matches
                # _multi_base_forward_rect (pad to gl, conv, take first nb_out)
                C = C[:nb_out, :nb_in]
                C = (
                    C
                    + self.rows[i].detach().float().unsqueeze(1)
                    + self.cols[i].detach().float().unsqueeze(0)
                )
            else:
                if self.is_rect:
                    C = C.reshape(nb_out, self.ratio, nb_in).sum(1)  # (nb_out, nb_in)
                if self.multi_dir:
                    C = (
                        C
                        + self.rows[i].detach().float().unsqueeze(1)
                        + self.cols[i].detach().float().unsqueeze(0)
                    )
            MS_i = getattr(self, f"MS_{i}").to(device)
            W_delta = W_delta + self.alphas[i].item() * torch.kron(C, MS_i)
        W = W + W_delta * self.bcms_scale.item()
        self._W_merged = W.to(torch.bfloat16)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._W_merged = None
        else:
            self._rebuild_eval_cache()
        return self

    # ── orthogonality penalty (optional; subspace separation) ─────────────
    def ortho_loss(self, weight: float = 0.0) -> torch.Tensor:
        if weight <= 0 or self.is_rect:
            return torch.zeros((), device=self.lora_A.device)
        W_lora = (self.lora_B @ self.lora_A) * self.lora_scale
        nb = self.nb_in
        idx = (
            torch.arange(nb, device=W_lora.device)[:, None]
            - torch.arange(nb, device=W_lora.device)[None, :]
        ) % nb
        W_bcms = torch.zeros_like(W_lora)
        for i, base in enumerate(self.bases):
            C = base[idx]
            if self.multi_dir:
                C = C + self.rows[i].unsqueeze(1) + self.cols[i].unsqueeze(0)
            W_bcms = W_bcms + self.alphas[i] * torch.kron(C, getattr(self, f"MS_{i}"))
        W_bcms = W_bcms * self.bcms_scale
        ip = (W_lora * W_bcms).sum().abs()
        return weight * ip / (W_lora.norm() * W_bcms.norm() + 1e-8)

    # ── bookkeeping ───────────────────────────────────────────────────────
    @property
    def n_lora_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    @property
    def n_bcms_params(self) -> int:
        n = sum(b.numel() for b in self.bases) + self.alphas.numel() + self.bcms_scale.numel()
        if self.multi_dir:
            n += sum(r.numel() for r in self.rows) + sum(c.numel() for c in self.cols)
        return n

    @property
    def n_trainable(self) -> int:
        return self.n_lora_params + self.n_bcms_params

    @classmethod
    def replace_in_model(
        cls,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        lora_r: int = 16,
        lora_alpha: Optional[float] = None,
        block_size: int = 8,
        n_bases: int = 4,
        multi_dir: bool = True,
        lora_init: str = "zero",
        verbose: bool = True,
    ) -> Dict[str, "BloRAAdapter"]:
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        reps: Dict[str, BloRAAdapter] = {}
        device = next(model.parameters()).device

        def _recurse(module, path=""):
            for name, child in list(module.named_children()):
                full = f"{path}.{name}" if path else name
                is_lin = isinstance(child, nn.Linear)
                is_4 = type(child).__name__ == "Linear4bit"
                is_8 = type(child).__name__ == "Linear8bitLt"
                if is_lin or is_4 or is_8:
                    if name.split(".")[-1] not in target_modules:
                        _recurse(child, full)
                        continue
                    inf, outf = child.in_features, child.out_features
                    # only requirement now: block_size divides both dims.
                    # arbitrary rectangular (MoE up-projection, non-integer
                    # ratio) is handled by the general-rect BCMS path.
                    if inf % block_size or outf % block_size:
                        _recurse(child, full)
                        continue
                    try:
                        if is_4:
                            import bitsandbytes.functional as bF

                            w = bF.dequantize_4bit(
                                child.weight.data, child.weight.quant_state
                            ).float()
                        elif is_8:
                            import bitsandbytes.functional as bF

                            w = bF.dequantize_blockwise(
                                child.weight.data, child.weight.quant_state
                            ).float()
                        else:
                            w = child.weight.data.float()
                    except Exception:
                        w = child.weight.float()
                    adapter = cls(
                        inf,
                        outf,
                        pretrained_weight=w,
                        pretrained_bias=(child.bias.data if child.bias is not None else None),
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        block_size=block_size,
                        n_bases=n_bases,
                        multi_dir=multi_dir,
                        lora_init=lora_init,
                    ).to(device)
                    setattr(module, name, adapter)
                    reps[full] = adapter
                    _clear_svd_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if verbose:
                        print(
                            f"  [{len(reps):3d}] {full}: {outf}x{inf} | "
                            f"LoRA={adapter.n_lora_params:,}+BCMS={adapter.n_bcms_params:,}"
                            f"={adapter.n_trainable:,}{' RECT' if adapter.is_rect else ''}",
                            flush=True,
                        )
                elif not isinstance(child, (nn.Linear, cls)):
                    _recurse(child, full)

        _recurse(model)
        _clear_svd_cache()
        return reps


def freeze_for_blora(model: nn.Module) -> Dict[str, int]:
    """Freeze everything except BloRA adapter params (LoRA half + BCMS half)."""
    keys = ("lora_A", "lora_B", "bases", "alphas", "bcms_scale", "rows", "cols")
    stats = {"frozen": 0, "trainable": 0}
    for name, p in model.named_parameters():
        is_adapter = any(k in name for k in keys)
        p.requires_grad_(is_adapter)
        stats["trainable" if is_adapter else "frozen"] += p.numel()
    return stats
