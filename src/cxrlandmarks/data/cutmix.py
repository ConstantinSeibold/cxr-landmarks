"""Batch-level CutMix for instance targets (port of qseg ``qseg/data/cutmix.py``).

Applied by the TRAINING LOOP on the collated list of samples (main process, after the
loader), exactly where the paper run applied it: ``apply_cutmix(batch)`` right after
``next(loader)``. For consecutive pairs (0,1), (2,3), ...: with prob ``prob`` (python
``random``) paste a Beta(alpha, alpha)-sized rectangle of sample i+1 into sample i
(numpy ``beta`` + 2x ``randint``), drop sample i's instances whose box centre lies in the
rectangle and append sample i+1's instances whose box centre lies in it (their masks
cropped to the rectangle; boxes/keypoints/valid masks are NOT updated -- as in qseg).
Pairs whose images differ in shape (e.g. after aspect jitter) are skipped after the coin
flip. Sample i+1 is left untouched.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch

_INSTANCE_KEYS = ("classes", "boxes", "masks", "keypoints", "keypoints_valid")


def _box_centers_inside(boxes: torch.Tensor, x1, y1, x2, y2) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    return (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)


def _sample_cut_region(h: int, w: int, alpha: float) -> tuple[int, int, int, int]:
    lam = float(np.random.beta(alpha, alpha))
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h = max(1, int(round(cut_ratio * h)))
    cut_w = max(1, int(round(cut_ratio * w)))
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, y1 + cut_h)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, x1 + cut_w)
    return x1, y1, x2, y2


def _select(sample: dict[str, Any], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: sample[k][keep] for k in _INSTANCE_KEYS}


def _mix_pair(a: dict[str, Any], b: dict[str, Any], alpha: float) -> None:
    img_a, img_b = a["image"], b["image"]
    if img_a.shape != img_b.shape:
        return
    _, H, W = img_a.shape
    x1, y1, x2, y2 = _sample_cut_region(H, W, alpha)
    if x2 <= x1 or y2 <= y1:
        return
    img_a[:, y1:y2, x1:x2] = img_b[:, y1:y2, x1:x2]
    na, nb = len(a["classes"]), len(b["classes"])
    if na + nb == 0:
        return
    inst_a = {k: a[k] for k in _INSTANCE_KEYS}
    if na:
        inst_a = _select(a, ~_box_centers_inside(a["boxes"], x1, y1, x2, y2))
    if nb:
        inst_b = _select(b, _box_centers_inside(b["boxes"], x1, y1, x2, y2))
        if len(inst_b["classes"]):
            m = inst_b["masks"]
            keep = torch.zeros_like(m)
            keep[:, y1:y2, x1:x2] = m[:, y1:y2, x1:x2]
            inst_b["masks"] = keep
            if len(inst_a["classes"]):
                inst_b = {k: torch.cat([inst_a[k], inst_b[k]], dim=0) for k in _INSTANCE_KEYS}
            a.update(inst_b)
            return
    a.update(inst_a)


def apply_cutmix(batch: list[dict[str, Any]], *, enable: bool = True, prob: float = 0.5,
                 alpha: float = 1.0) -> None:
    """In-place CutMix over consecutive pairs of ``batch`` (list of sample dicts)."""
    if not enable or len(batch) < 2 or prob <= 0:
        return
    for i in range(0, len(batch) - 1, 2):
        if random.random() < prob:
            _mix_pair(batch[i], batch[i + 1], alpha)
