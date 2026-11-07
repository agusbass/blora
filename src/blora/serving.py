"""
serving.py — Rank-Agnostic Batched Multi-Adapter Serving for BloRA
==================================================================
Serving requirement #1: serve thousands of requests, each using a DIFFERENT
adapter with a DIFFERENT rank, in ONE vectorized GPU pass — no per-request
Python loop, no per-rank queue (the S-LoRA / Punica problem), extended to
BloRA's circulant BCMS half.

Given a SHARED frozen base weight W (one layer of one base model) and a pack
of M heterogeneous BloRA adapters, this computes for a batch of B requests
(each a sequence of T tokens, each request i bound to adapter_id[i]):

    out[i] = W·x[i] + LoRA_{a_i}(x[i]) + BCMS_{a_i}(x[i])

all at once. Heterogeneity is handled by padding to the max rank / max number
of bases and zero-filling the unused slots (zeros contribute nothing), then a
single gather + batched einsum/FFT. The circulant generators differ per
request, but the FFT keeps the request index as a leading batch dim, so
"different rank per request" costs one batched rfft — not a loop.

This mirrors the TRAIN-mode math of BloRAAdapter exactly (verified by
test_serving), so a packed server is numerically identical to running each
adapter on its own slice.

Proprietary technology. Rayalabs.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from blora._bcms_core import _make_ms_rotated
from blora.adapter import BloRAAdapter


class BloRAServingPack(nn.Module):
    """Stacks M BloRA adapters sharing one base weight into padded tensors for
    rank-agnostic batched serving. All adapters must share in/out/block_size
    (same base layer); lora_r and n_bases MAY differ across adapters."""

    # buffer type annotations (nn.Module __getattr__ otherwise widens to
    # Tensor | Module | None, which trips mypy on tensor method calls)
    W_frozen: torch.Tensor
    b_frozen: "torch.Tensor | None"
    MS_stack: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    lscale: torch.Tensor
    bases: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    alphas: torch.Tensor
    bscale: torch.Tensor

    def __init__(self, adapters: List[BloRAAdapter]):
        super().__init__()
        a0 = adapters[0]
        self.n_in, self.n_out = a0.n_in, a0.n_out
        self.block_size = a0.block_size
        self.nb_in, self.nb_out = a0.nb_in, a0.nb_out
        self.ratio = a0.ratio
        self.is_rect = a0.is_rect
        self.M = len(adapters)
        for a in adapters:
            assert (a.n_in, a.n_out, a.block_size) == (self.n_in, self.n_out, self.block_size), (
                "all adapters in a pack must share the base layer shape"
            )
            assert a.multi_dir, "serving pack expects multi_dir adapters (rows/cols present)"

        self.Rmax = max(a.lora_r for a in adapters)
        self.Kmax = max(a.n_bases for a in adapters)

        # shared frozen base (same across adapters) + shared MS kernels
        self.register_buffer("W_frozen", a0.W_frozen.clone())
        self.register_buffer("b_frozen", a0.b_frozen.clone() if a0.b_frozen is not None else None)
        # MS kernels depend only on (block_size, basis index) — rebuild the
        # full Kmax stack so packs with heterogeneous n_bases are covered.
        MS = torch.stack([_make_ms_rotated(self.block_size, rotate=i) for i in range(self.Kmax)], 0)
        self.register_buffer("MS_stack", MS)  # (Kmax, bs, bs)

        # ---- padded LoRA stacks ----
        A = torch.zeros(self.M, self.Rmax, self.n_in)
        B = torch.zeros(self.M, self.n_out, self.Rmax)
        lscale = torch.zeros(self.M)
        # ---- padded BCMS stacks ----
        bases = torch.zeros(self.M, self.Kmax, self.nb_in)
        rows = torch.zeros(self.M, self.Kmax, self.nb_out)
        cols = torch.zeros(self.M, self.Kmax, self.nb_in)
        alphas = torch.zeros(self.M, self.Kmax)
        bscale = torch.zeros(self.M)

        for m, a in enumerate(adapters):
            r, k = a.lora_r, a.n_bases
            A[m, :r] = a.lora_A.detach().float()
            B[m, :, :r] = a.lora_B.detach().float()
            lscale[m] = a.lora_scale
            for i in range(k):
                bases[m, i] = a.bases[i].detach().float()
                rows[m, i] = a.rows[i].detach().float()
                cols[m, i] = a.cols[i].detach().float()
            alphas[m, :k] = a.alphas.detach().float()
            bscale[m] = a.bcms_scale.detach().float()

        for name, t in [
            ("A", A),
            ("B", B),
            ("lscale", lscale),
            ("bases", bases),
            ("rows", rows),
            ("cols", cols),
            ("alphas", alphas),
            ("bscale", bscale),
        ]:
            self.register_buffer(name, t)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_in)   adapter_ids: (B,) int64 -> out: (B, T, n_out)."""
        dev, dt = x.device, x.dtype
        Bb, T, _ = x.shape
        nb_in, nb_out, bs, K = self.nb_in, self.nb_out, self.block_size, self.Kmax

        # ---- shared base ----
        Wf = self.W_frozen.to(dev, dt)
        bf = self.b_frozen.to(dev, dt) if self.b_frozen is not None else None
        out = F.linear(x, Wf, bf)  # (B,T,out)

        xf = x.float()
        ids = adapter_ids.to(dev)

        # ---- rank-agnostic LoRA (one gather + two batched einsums) ----
        A_b = self.A.index_select(0, ids)  # (B,Rmax,in)
        B_b = self.B.index_select(0, ids)  # (B,out,Rmax)
        t = torch.einsum("bti,bri->btr", xf, A_b)  # (B,T,Rmax)
        lora = torch.einsum("btr,bor->bto", t, B_b)  # (B,T,out)
        lora = lora * self.lscale.index_select(0, ids)[:, None, None]
        out = out + lora.to(dt)

        # ---- batched BCMS: per-request circulant generators, one FFT pass ----
        bases_b = self.bases.index_select(0, ids)  # (B,K,nb_in)
        rows_b = self.rows.index_select(0, ids)  # (B,K,nb_out)
        cols_b = self.cols.index_select(0, ids)  # (B,K,nb_in)
        alpha_b = self.alphas.index_select(0, ids)  # (B,K)
        bscale_b = self.bscale.index_select(0, ids)  # (B,)
        MS = self.MS_stack.to(dev).float()  # (K,bs,bs)

        xb = xf.reshape(Bb, T, nb_in, bs)  # (B,T,nb_in,bs)
        # xms[b,t,k,n,o] = xb[b,t,n,:] @ MS[k,o,:]
        xms = torch.einsum("btns,kos->btkno", xb, MS)  # (B,T,K,nb_in,bs)

        # diagonal circulant via per-request rfft over nb_in
        xms_t = xms.permute(0, 1, 2, 4, 3)  # (B,T,K,bs,nb_in)
        xfreq = torch.fft.rfft(xms_t, n=nb_in)  # (B,T,K,bs,F)
        bfreq = torch.fft.rfft(bases_b, n=nb_in)  # (B,K,F)
        of = xfreq * bfreq[:, None, :, None, :]  # broadcast T,bs
        ot = torch.fft.irfft(of, n=nb_in)  # (B,T,K,bs,nb_in)
        block = ot.permute(0, 1, 2, 4, 3)  # (B,T,K,nb_in,bs)

        # rectangular (nb_out<nb_in): group-sum input blocks -> output blocks
        if nb_out != nb_in:
            block = block.reshape(Bb, T, K, nb_out, self.ratio, bs).sum(4)

        # horizontal (rows) + vertical (cols) rank-1 inter-block terms
        pooled = xms.sum(3)  # (B,T,K,bs) sum over input blocks
        horiz = rows_b[:, None, :, :, None] * pooled[:, :, :, None, :]  # (B,T,K,nb_out,bs)
        cpool = torch.einsum("bkn,btkns->btks", cols_b, xms)  # (B,T,K,bs)
        vert = cpool[:, :, :, None, :].expand(-1, -1, -1, nb_out, -1)  # (B,T,K,nb_out,bs)
        block = block + horiz + vert  # (B,T,K,nb_out,bs)

        # weight by per-request alpha, sum bases, scale, flatten to (B,T,out)
        weighted = (alpha_b[:, None, :, None, None] * block).sum(2)  # (B,T,nb_out,bs)
        bcms = weighted.reshape(Bb, T, nb_out * bs) * bscale_b[:, None, None]
        out = out + bcms.to(dt)
        return out
