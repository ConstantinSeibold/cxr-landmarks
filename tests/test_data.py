"""Standalone checks of the data pipeline (no detectron2 / qseg needed).

Uses the real PAXRay++ val json when available (skipped otherwise)."""
from __future__ import annotations

import builtins
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cxrlandmarks.data import (AugConfig, LandmarkMapper, PaperOrderSampler, apply_cutmix,  # noqa: E402
                               paper_batch_indices, paper_config, pad_batch, training_index_stream)
from cxrlandmarks.data import coco as C  # noqa: E402

CFG = paper_config()
HAVE_DATA = Path(CFG.json_val).exists()


def test_no_forbidden_imports(monkeypatch):
    for m in [k for k in sys.modules if k.startswith("cxrlandmarks.data")]:
        monkeypatch.delitem(sys.modules, m)
    real = builtins.__import__

    def guard(name, *a, **k):
        if name.split(".")[0] in ("detectron2", "fvcore", "hydra", "qseg", "omegaconf"):
            raise ImportError(f"forbidden import {name}")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    import cxrlandmarks.data  # noqa: F401


def test_paper_config_values():
    a = CFG.aug
    assert (a.input_size, a.hflip, a.num_ops_min, a.num_ops, a.magnitude_min, a.magnitude) == (512, "lateral", 2, 4, 0.5, 1.0)
    assert (a.rotation_deg, a.shear_deg, a.perspective_scale, a.aspect_jitter_max) == (10.0, 30.0, 0.02, 0.1)
    assert (a.cutmix, a.cutmix_prob, a.cutmix_alpha) == (True, 0.5, 1.0)
    assert (CFG.batch_size, CFG.seed, CFG.num_workers) == (8, 42, 16)


def test_sampler_is_permutation_per_block():
    n = 50
    s = list(itertools.islice(training_index_stream(n, 42), n))
    assert sorted(s) == list(range(n))
    b = paper_batch_indices(n, 2, num_workers=1)
    assert [i for x in b for i in x] == s[:16]
    flat = list(itertools.islice(iter(PaperOrderSampler(1000, num_workers=16)), 128))
    assert sorted(flat) == sorted(list(itertools.islice(training_index_stream(1000, 42), 128)))


def _fake_sample(n: int, h: int = 64, w: int = 64, k: int = 13, val: int = 0):
    m = torch.zeros((n, h, w), dtype=torch.bool)
    boxes = torch.zeros((n, 4))
    for i in range(n):
        y, x = 8 * i % (h - 8), 8 * i % (w - 8)
        m[i, y:y + 8, x:x + 8] = True
        boxes[i] = torch.tensor([x, y, x + 8, y + 8], dtype=torch.float32)
    return {"image": torch.full((3, h, w), val, dtype=torch.uint8), "classes": torch.arange(n),
            "boxes": boxes, "masks": m, "keypoints": torch.zeros((n, k, 3)), "keypoints_valid": torch.ones((n, k))}


def test_cutmix_consistency():
    random.seed(0)
    np.random.seed(0)
    batch = [_fake_sample(6, val=10), _fake_sample(5, val=200)]
    apply_cutmix(batch, enable=True, prob=1.0, alpha=1.0)
    a = batch[0]
    n = len(a["classes"])
    assert all(a[k].shape[0] == n for k in ("boxes", "masks", "keypoints", "keypoints_valid"))
    assert (a["image"] == 200).any()
    assert torch.equal(batch[1]["image"], torch.full((3, 64, 64), 200, dtype=torch.uint8))


def test_pad_batch():
    x = pad_batch([torch.zeros((3, 500, 470), dtype=torch.uint8), torch.ones((3, 512, 512), dtype=torch.uint8)])
    assert x.shape == (2, 3, 512, 512) and x.dtype == torch.float32


@pytest.mark.skipif(not HAVE_DATA, reason="PAXRay++ val json not available")
def test_mapper_on_val_images():
    raw = C.load_json(CFG.json_val)
    raw["images"] = raw["images"][:4]
    keep = {im["id"] for im in raw["images"]}
    raw["annotations"] = [a for a in raw["annotations"] if a["image_id"] in keep]
    cats = C.scan_categories(raw)
    recs = C.build_records(raw, CFG.image_root, cats)
    for is_train in (False, True):
        m = LandmarkMapper(AugConfig(), is_train=is_train, k_max=cats.k_max)
        for i, r in enumerate(recs):
            np.random.seed(i)
            random.seed(i)
            s = m(r)
            n, (H, W) = len(s["classes"]), s["image"].shape[1:]
            assert s["image"].dtype == torch.uint8 and s["image"].shape[0] == 3
            assert s["masks"].shape == (n, H, W) and s["masks"].dtype == torch.bool
            assert s["keypoints"].shape == (n, cats.k_max, 3) and s["keypoints_valid"].shape == (n, cats.k_max)
            vis = s["keypoints"][..., 2] > 0
            xy = s["keypoints"][..., :2][vis]
            assert (xy >= 0).all() and (xy[:, 0] <= W).all() and (xy[:, 1] <= H).all()
            assert int(s["classes"].max()) < cats.num_classes
            if not is_train:
                assert (H, W) == (512, 512) and s["masks"].any(dim=(1, 2)).all()
