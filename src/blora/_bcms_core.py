"""
_bcms_core.py — BCMS core machinery for BloRA (self-contained, no bcms import)
================================================================================
Contains:
    - Magic Square kernel builder (MS 3x3 rotated)
    - SVD with caching + SVD-guided base init
    - FFT-based circulant multiply
    - Multi-base batched forward pass

Extracted from bcms.adapter_v6 — validated v6.3 machinery.
No R_MS, no scipy in forward path, train≡eval verified.
Proprietary technology. Rayalabs.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# =======================================================================
# Magic Square Kernel Builder — Rotated per basis
# =======================================================================

_MS3_RAW = np.array([[8.0, 1.0, 6.0], [3.0, 5.0, 7.0], [4.0, 9.0, 2.0]], dtype=np.float32)


def _make_ms_rotated(block_size: int, rotate: int = 0) -> torch.Tensor:
    """Build a Magic Square kernel tiled to block_size, optionally rotated."""
    reps = (block_size // 3) + 1
    tiled = np.tile(_MS3_RAW, (reps, reps))[:block_size, :block_size].copy()
    if rotate > 0:
        tiled = np.roll(tiled, rotate, axis=0)
        tiled = np.roll(tiled, rotate, axis=1)
    ms_norm = (tiled / np.linalg.norm(tiled, "fro")).astype(np.float32)
    return torch.from_numpy(ms_norm)


# =======================================================================
# SVD Initialization
# =======================================================================

_SVD_CACHE: Dict[Tuple, Tuple] = {}


@torch.no_grad()
def _cached_svd(W: torch.Tensor, rank: int = 64):
    """Approximate top-rank SVD via randomized algorithm (svd_lowrank).

    10-50x faster than full SVD for rank << N. Cached per weight tensor.
    Note: svd_lowrank returns V (not Vh) — we transpose here.
    """
    key = (W.data_ptr(), rank)
    if key in _SVD_CACHE:
        return _SVD_CACHE[key]
    U, S, V = torch.svd_lowrank(W.float(), q=rank, niter=4)
    Vh = V.T  # svd_lowrank returns V, transpose to get Vh
    _SVD_CACHE[key] = (U, S, Vh)
    return U, S, Vh


def _clear_svd_cache():
    """Clear the global SVD cache."""
    _SVD_CACHE.clear()


@torch.no_grad()
def _svd_init_base(
    W: torch.Tensor,
    MS: torch.Tensor,
    n_blocks: int,
    block_size: int,
    rank: int = 64,
    noise_eps: float = 0.01,
) -> torch.Tensor:
    """SVD-guided init for one base vector.

    Extracts the circulant generator from the low-rank approximation of W.
    Single GEMM replaces Python loop over rank dimension.
    """
    U, S, Vh = _cached_svd(W, rank)
    ms_norm_sq = (MS * MS).sum()

    # One GEMM: (N, rank) * S^2 @ (rank, N) = (N, N), low-rank W approx
    W_approx = (U * S.pow(2)) @ Vh  # (N, N)
    W_blocks = W_approx.view(n_blocks, block_size, n_blocks, block_size).permute(0, 2, 1, 3)
    ms_flat = MS.reshape(1, 1, -1)
    W_flat = W_blocks.reshape(n_blocks, n_blocks, -1)
    C_accum = (ms_flat * W_flat).sum(dim=-1) / ms_norm_sq  # (nb, nb)

    w_sum = S.sum()
    C_avg = C_accum / w_sum if w_sum > 0 else C_accum

    # Extract circulant generator: average FFT columns -> base vector
    C_fft = torch.fft.fft(C_avg.float(), dim=0)
    base = torch.fft.ifft(C_fft.mean(dim=1)).real

    if noise_eps > 0:
        base = base + noise_eps * torch.randn_like(base)
    return base


# =======================================================================
# FFT-Based Circulant Multiply — O(N log N) vs O(N^2) gather
# =======================================================================


def _circulant_fft(xms: torch.Tensor, base: torch.Tensor, nb: int) -> torch.Tensor:
    """Circulant multiply via FFT: circulant(base) applied along dim-1 of xms.

    xms:  (Bf, nb, bs) — input after MS transform
    base: (nb,) — circulant generator
    Returns: (Bf, nb, bs)

    Circulant-vector product = circular convolution = ifft(fft(base) * fft(v)).
    Complex multiply in frequency domain, O(nb * log(nb)) per channel.
    """
    xms_t = xms.permute(0, 2, 1).float()  # (Bf, bs, nb)
    base_f = torch.fft.rfft(base.float(), n=nb)  # (nb//2+1,)
    xms_f = torch.fft.rfft(xms_t, n=nb)  # (Bf, bs, nb//2+1)
    out_f = xms_f * base_f  # broadcast
    out_t = torch.fft.irfft(out_f, n=nb)  # (Bf, bs, nb)
    return out_t.permute(0, 2, 1).to(xms.dtype)  # (Bf, nb, bs)


def _factor_2d(nb: int) -> Tuple[int, int]:
    """Factor nb into (p, q), p*q=nb, p as close to sqrt(nb) as divides nb."""
    p = int(math.isqrt(nb))
    while nb % p != 0:
        p -= 1
    return p, nb // p


def _multi_base_forward(
    x: torch.Tensor,
    bases: List[torch.Tensor],
    MS_list: List[torch.Tensor],
    alphas: torch.Tensor,
    nb: int,
    bs: int,
    rows: Optional[List[torch.Tensor]] = None,
    cols: Optional[List[torch.Tensor]] = None,
    ring_pq: Optional[Tuple[int, int]] = None,
    nb_out: Optional[int] = None,
) -> torch.Tensor:
    """Batched forward: all k bases in one einsum + one batched FFT.

    W_delta @ x = sum_i alpha_i * kron(M_i, MS_i) @ x, where the inter-block
    operator M_i[a,b] captures up to THREE directions:
        M_i[a,b] = base_i[(a-b) mod nb]   (diagonal  — circulant, FFT)
                 + rows_i[a]              (horizontal — row-constant, broadcast)
                 + cols_i[b]              (vertical   — col-constant, pooled)
    rows/cols are None for pure-circulant (diagonal only).

    Rectangular (nb_out < nb): circulant runs on the nb input-blocks, then
    output blocks are formed by group-summing nb -> nb_out (ratio = nb/nb_out).
    M[a_out, b] = sum_{a_in in group(a_out)} circ[a_in, b]. Square = ratio 1.
    """
    orig_shape = x.shape
    k = len(bases)
    nb_in = nb
    nb_out = nb_out if nb_out is not None else nb_in
    N_in = nb_in * bs
    dev = x.device

    x_flat = x.reshape(-1, N_in).float()  # (Bf, N_in)
    Bf = x_flat.shape[0]
    xb = x_flat.reshape(Bf, nb_in, bs)  # (Bf, nb_in, bs)

    MS_stack = torch.stack(MS_list, 0).to(dev).float()  # (k, bs, bs)
    bases_stack = torch.stack(bases, 0).to(dev).float()  # (k, nb_in)

    # xb @ MS[i].T for all i: einsum 'fns,kts->kfnt'
    xms_k = torch.einsum("fns,kts->kfnt", xb, MS_stack)  # (k, Bf, nb_in, bs)

    # Diagonal: circulant (1D ring, FFT) or BCCB (2D torus, FFT2) over nb_in blocks
    if ring_pq is None:
        xms_k_t = xms_k.permute(0, 1, 3, 2)  # (k, Bf, bs, nb_in)
        bases_f = torch.fft.rfft(bases_stack, n=nb_in)  # (k, nb_in//2+1)
        xms_f = torch.fft.rfft(xms_k_t, n=nb_in)  # (k, Bf, bs, nb_in//2+1)
        out_f = xms_f * bases_f[:, None, None, :]  # broadcast
        out_t = torch.fft.irfft(out_f, n=nb_in)  # (k, Bf, bs, nb_in)
        block_out = out_t.permute(0, 1, 3, 2)  # (k, Bf, nb_in, bs)
    else:
        p, q = ring_pq  # p*q == nb_in
        xms_2d = xms_k.permute(0, 1, 3, 2).reshape(k, Bf, bs, p, q)  # (k,Bf,bs,p,q)
        g_2d = bases_stack.reshape(k, p, q)  # (k, p, q)
        G = torch.fft.rfft2(g_2d, s=(p, q))  # (k, p, q//2+1)
        X = torch.fft.rfft2(xms_2d, s=(p, q))  # (k,Bf,bs,p,q//2+1)
        out_2d = torch.fft.irfft2(X * G[:, None, None, :, :], s=(p, q))  # (k,Bf,bs,p,q)
        block_out = out_2d.reshape(k, Bf, bs, nb_in).permute(0, 1, 3, 2)  # (k,Bf,nb_in,bs)

    # Rectangular: group-sum nb_in -> nb_out on the circulant part (square: ratio 1, no-op)
    if nb_out != nb_in:
        ratio = nb_in // nb_out
        block_out = block_out.reshape(k, Bf, nb_out, ratio, bs).sum(dim=3)  # (k,Bf,nb_out,bs)

    # Horizontal + Vertical: rank-1 inter-block terms, applied on OUTPUT blocks.
    #   M[a_out,b_in] = groupsum(circ) + rows[a_out] + cols[b_in]
    if rows is not None:
        assert cols is not None, "rows and cols must be provided together"
        rows_stack = torch.stack(rows, 0).to(dev).float()  # (k, nb_out)
        cols_stack = torch.stack(cols, 0).to(dev).float()  # (k, nb_in)
        pooled = xms_k.sum(dim=2)  # (k, Bf, bs)  sum over input blocks
        horiz = rows_stack[:, None, :, None] * pooled[:, :, None, :]  # (k, Bf, nb_out, bs)
        cpool = torch.einsum("kb,kfbs->kfs", cols_stack, xms_k)  # (k, Bf, bs)
        vert = cpool[:, :, None, :].expand(-1, -1, nb_out, -1)  # (k, Bf, nb_out, bs)
        block_out = block_out + horiz + vert

    out_k = block_out.reshape(k, Bf, nb_out * bs)  # (k, Bf, N_out)
    weighted = (alphas.float().view(k, 1, 1).to(out_k.device) * out_k).sum(0)  # (Bf, N_out)
    out_shape = (*orig_shape[:-1], nb_out * bs)
    return weighted.to(x.dtype).reshape(out_shape)


def _multi_base_forward_rect(
    x: torch.Tensor,
    bases: List[torch.Tensor],
    MS_list: List[torch.Tensor],
    alphas: torch.Tensor,
    nb_in: int,
    nb_out: int,
    bs: int,
    rows: List[torch.Tensor],
    cols: List[torch.Tensor],
) -> torch.Tensor:
    """General rectangular BCMS forward — ANY (nb_in, nb_out), no divisibility.

    Builds the inter-block operator M_i (nb_out x nb_in) as the top-left slice
    of a full circulant of size nb = max(nb_in, nb_out) generated by base_i
    (length nb), plus the rank-1 row/col terms:

        M_i[a,b] = base_i[(a-b) mod nb] | a<nb_out, b<nb_in
                 + rows_i[a] + cols_i[b]

    The circular convolution runs on the padded length nb (input zero-padded
    from nb_in -> nb), then the first nb_out output blocks are kept. This
    reduces exactly to the square path when nb_in == nb_out, and unlike the
    integer group-sum path it supports up-projection (out > in) and non-integer
    ratios — i.e. MoE MLP / expert shapes (e.g. 4096 <-> 14336).
    """
    orig_shape = x.shape
    k = len(bases)
    nb = max(nb_in, nb_out)  # generator length
    N_in = nb_in * bs
    dev = x.device

    x_flat = x.reshape(-1, N_in).float()  # (Bf, N_in)
    Bf = x_flat.shape[0]
    xb = x_flat.reshape(Bf, nb_in, bs)  # (Bf, nb_in, bs)

    MS_stack = torch.stack(MS_list, 0).to(dev).float()  # (k, bs, bs)
    bases_stack = torch.stack(bases, 0).to(dev).float()  # (k, nb)

    # xb @ MS[i].T for all i -> (k, Bf, nb_in, bs)
    xms_k = torch.einsum("fns,kts->kfnt", xb, MS_stack)

    # zero-pad input blocks nb_in -> nb, then per-base circular conv via FFT
    if nb_in < nb:
        pad = xms_k.new_zeros(k, Bf, nb - nb_in, bs)
        xms_pad = torch.cat([xms_k, pad], dim=2)  # (k, Bf, nb, bs)
    else:
        xms_pad = xms_k
    xms_t = xms_pad.permute(0, 1, 3, 2)  # (k, Bf, bs, nb)
    bases_f = torch.fft.rfft(bases_stack, n=nb)  # (k, nb//2+1)
    xms_f = torch.fft.rfft(xms_t, n=nb)  # (k, Bf, bs, nb//2+1)
    out_t = torch.fft.irfft(xms_f * bases_f[:, None, None, :], n=nb)  # (k, Bf, bs, nb)
    block_full = out_t.permute(0, 1, 3, 2)  # (k, Bf, nb, bs)
    block_out = block_full[:, :, :nb_out, :]  # slice -> (k, Bf, nb_out, bs)

    # rank-1 horizontal (rows over nb_out) + vertical (cols over nb_in)
    rows_stack = torch.stack(rows, 0).to(dev).float()  # (k, nb_out)
    cols_stack = torch.stack(cols, 0).to(dev).float()  # (k, nb_in)
    pooled = xms_k.sum(dim=2)  # (k, Bf, bs)  sum over input blocks
    horiz = rows_stack[:, None, :, None] * pooled[:, :, None, :]  # (k, Bf, nb_out, bs)
    cpool = torch.einsum("kb,kfbs->kfs", cols_stack, xms_k)  # (k, Bf, bs)
    vert = cpool[:, :, None, :].expand(-1, -1, nb_out, -1)  # (k, Bf, nb_out, bs)
    block_out = block_out + horiz + vert

    out_k = block_out.reshape(k, Bf, nb_out * bs)  # (k, Bf, N_out)
    weighted = (alphas.float().view(k, 1, 1).to(out_k.device) * out_k).sum(0)  # (Bf, N_out)
    out_shape = (*orig_shape[:-1], nb_out * bs)
    return weighted.to(x.dtype).reshape(out_shape)
