"""
BloRA — BCMS + LoRA, complementary-subspace PEFT adapter.

Splits the SVD energy of W between two structurally-different bases so
each owns a disjoint subspace instead of competing over the same one:
    LoRA  (low-rank)        -> top-r, high-energy directions  (drives PPL)
    BCMS  (circulant-Kron)  -> residual structure LoRA misses (drives
                               param-efficiency / generalization)

Built on top of validated BCMS v6.3 machinery — self-contained, no external
bcms package dependency.

Public API:
    BloRAAdapter      : LoRA + BCMS hybrid adapter (PiSSA-style subspace split)
    freeze_for_blora  : freeze all but BloRA adapter params
    BloRAServingPack  : rank-agnostic batched multi-adapter serving (one pass)

Proprietary technology. PT Raya Laksana Teknologi (ralafs.com).
"""

from blora.adapter import BloRAAdapter, freeze_for_blora
from blora.kv_fusion import fuse_kv, fuse_kv_in_model
from blora.serving import BloRAServingPack

__all__ = [
    "BloRAAdapter",
    "freeze_for_blora",
    "BloRAServingPack",
    "fuse_kv",
    "fuse_kv_in_model",
]
