"""Equivalence proof: the ported data pipeline == the paper's qseg/detectron2 pipeline.

Runs only where qseg + detectron2 are importable (the qseg env with
PYTHONPATH=<qseg>/src:<qseg>/third_party/MaskDINO); skipped elsewhere.

For every sample both pipelines start from the SAME python ``random`` / numpy / torch
state; we assert bit-identical images, masks, boxes, classes, visibility and valid masks,
keypoints within 1e-4 px, AND identical RNG states afterwards (same random numbers
consumed in the same order).

Train samples: the first 10 batches (80 samples) the paper run's loader emitted
(``outputs/cache/train_paper_first80.json``, written by ``outputs/scratch/make_train_subset.py``
from train_kp_v1.json; regenerated here from the full json if missing).
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# Reference implementation (paper code base, not public): set QSEG_ROOT and KP_PREP_ROOT to run.
QSEG = Path(os.environ.get("QSEG_ROOT", "qseg"))
for p in (QSEG / "src", QSEG / "third_party" / "MaskDINO", Path(os.environ.get("KP_PREP_ROOT", "kp_prep")) / "baselines"):
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))

pytest.importorskip("detectron2")
qseg_mapper_mod = pytest.importorskip("qseg.data.mapper")
pytest.importorskip("flat_heatmap", reason="paper training code (KP_PREP_ROOT) not available")
if not Path(paper_config_root := os.environ.get("PAXRAY_ROOT", "paxraypp")).exists():
    pytest.skip("PAXRay++ not available (set PAXRAY_ROOT)", allow_module_level=True)

from cxrlandmarks.data import coco as C  # noqa: E402
from cxrlandmarks.data import cutmix as PC  # noqa: E402
from cxrlandmarks.data import sampling as S  # noqa: E402
from cxrlandmarks.data.config import paper_config  # noqa: E402
from cxrlandmarks.data.mapper import LandmarkMapper  # noqa: E402

CFG = paper_config()
SUBSET = REPO / "outputs" / "cache" / "train_paper_first80.json"
REPORT = REPO / "outputs" / "equivalence_report.json"
N_EVAL = 24
RESULTS: dict = {}


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _rng_state():
    return (random.getstate(), np.random.get_state()[1].tobytes(), np.random.get_state()[2],
            torch.get_rng_state().numpy().tobytes())


def _ensure_subset() -> dict:
    if not SUBSET.exists():
        raw = C.load_json(CFG.json_train)
        filt = C.filter_train_records(C.build_records(raw, CFG.image_root, C.scan_categories(raw)))
        order = [filt[i]["image_id"] for b in S.paper_batch_indices(len(filt), 10) for i in b]
        keep = set(order)
        SUBSET.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"info": {"order_image_ids": order}, "categories": raw["categories"],
                   "images": [im for im in raw["images"] if im["id"] in keep],
                   "annotations": [a for a in raw["annotations"] if a["image_id"] in keep]},
                  open(SUBSET, "w"))
    return json.load(open(SUBSET))


@pytest.fixture(scope="module")
def env():
    import flat_heatmap as FH
    from qseg.data.coco_kpt_dataset import build_dataset_dicts
    from qseg.data.utils import scan_coco_json
    from qseg.models.build import _make_d2_cfg

    sub = _ensure_subset()
    scan = scan_coco_json(str(SUBSET), default_sigma=0.05)   # same categories as train json
    qcfg = FH.load_qseg_cfg("experiments/kp_isbi_ours_raddino", ["train.ims_per_batch=8"])
    qcfg.data.num_classes = scan.num_classes
    qcfg.data.has_keypoints = scan.has_keypoints
    qcfg.data.max_keypoints_per_cat = scan.max_keypoints_per_cat
    qcfg.data.max_instances_per_image = scan.max_instances_per_image
    d2 = _make_d2_cfg(qcfg)
    d2.MODEL.MASK_ON = True                                    # as mtl_baselines.py
    cats = C.scan_categories(sub)
    q_train = {d["image_id"]: d for d in build_dataset_dicts(str(SUBSET), CFG.image_root, scan)}
    p_train = {r["image_id"]: r for r in C.build_records(sub, CFG.image_root, cats)}
    val = C.load_json(CFG.json_val)
    q_val = build_dataset_dicts(CFG.json_val, CFG.image_root, scan)
    p_val = C.build_records(val, CFG.image_root, cats)
    return {
        "d2": d2, "scan": scan, "cats": cats, "order": sub["info"]["order_image_ids"],
        "q_train": q_train, "p_train": p_train, "q_val": q_val, "p_val": p_val,
        "q_mapper_train": qseg_mapper_mod.build_qseg_mapper(d2, is_train=True),
        "q_mapper_eval": qseg_mapper_mod.build_qseg_mapper(d2, is_train=False),
        "p_mapper_train": LandmarkMapper(CFG.aug, is_train=True, k_max=cats.k_max),
        "p_mapper_eval": LandmarkMapper(CFG.aug, is_train=False, k_max=cats.k_max),
    }


def _q_targets(out: dict) -> dict:
    """qseg mapper output -> port layout (after _materialize_gt_masks)."""
    from qseg.engine.trainer import _materialize_gt_masks
    _materialize_gt_masks([out])
    inst = out["instances"]
    n = len(inst)
    k = 13
    has_kp = inst.has("gt_keypoints_padded")
    return {
        "image": out["image"],
        "classes": inst.gt_classes,
        "boxes": inst.gt_boxes.tensor,
        "masks": inst.gt_masks if inst.has("gt_masks") else torch.zeros((0,) + tuple(out["image"].shape[1:]), dtype=torch.bool),
        "keypoints": inst.gt_keypoints_padded if has_kp else torch.zeros((n, k, 3)),
        "keypoints_valid": inst.gt_keypoints_valid_mask if has_kp else torch.zeros((n, k)),
    }


def _compare(q: dict, p: dict, stats: dict) -> None:
    assert q["image"].dtype == p["image"].dtype == torch.uint8
    assert q["image"].shape == p["image"].shape
    assert torch.equal(q["image"], p["image"]), "image differs"
    assert torch.equal(q["classes"], p["classes"]), "classes differ"
    assert q["masks"].dtype == p["masks"].dtype == torch.bool
    assert torch.equal(q["masks"], p["masks"]), "masks differ"
    assert torch.equal(q["boxes"], p["boxes"]), "boxes differ"
    assert q["keypoints"].shape == p["keypoints"].shape
    d = (q["keypoints"][..., :2] - p["keypoints"][..., :2]).abs()
    kd = float(d.max()) if d.numel() else 0.0
    assert kd <= 1e-4, f"keypoints differ by {kd}"
    assert torch.equal(q["keypoints"][..., 2], p["keypoints"][..., 2]), "visibility differs"
    assert torch.equal(q["keypoints_valid"], p["keypoints_valid"]), "valid mask differs"
    stats["max_kpt_abs_diff_px"] = max(stats.get("max_kpt_abs_diff_px", 0.0), kd)
    stats["kpt_bit_exact"] = stats.get("kpt_bit_exact", True) and torch.equal(q["keypoints"], p["keypoints"])
    stats["instances"] = stats.get("instances", 0) + int(q["classes"].numel())
    stats["visible_kpts"] = stats.get("visible_kpts", 0) + int((q["keypoints"][..., 2] > 0).sum())


def _record_keys_equal(q: dict, p: dict) -> None:
    for k in ("file_name", "image_id", "height", "width"):
        assert q[k] == p[k], k
    qa = [a for a in q["annotations"]]
    pa = p["annotations"]
    assert len(qa) == len(pa)
    for a, b in zip(qa, pa):
        assert a["category_id"] == b["category_id"]
        assert a["bbox"] == b["bbox"] and a["bbox_mode"] == 1
        assert a["iscrowd"] == b["iscrowd"]
        assert a["segmentation"] == b["segmentation"]
        assert a["keypoints"] == b["keypoints"]
        assert a["keypoints_valid_mask"] == b["keypoints_valid_mask"]


def test_dataset_records_identical(env):
    for iid, q in env["q_train"].items():
        _record_keys_equal(q, env["p_train"][iid])
    assert len(env["q_val"]) == len(env["p_val"])
    for q, p in zip(env["q_val"], env["p_val"]):
        _record_keys_equal(q, p)
    assert env["cats"].k_max == env["scan"].max_keypoints_per_cat == 13
    assert env["cats"].cat_ids == [env["scan"].contig_to_cat_id[i] for i in range(env["scan"].num_classes)]
    RESULTS["records"] = {"train_subset_images": len(env["q_train"]), "val_images": len(env["q_val"])}


def test_train_filters_identical(env):
    from detectron2.data.build import (filter_images_with_few_keypoints,
                                       filter_images_with_only_crowd_annotations)
    q = filter_images_with_few_keypoints(filter_images_with_only_crowd_annotations(env["q_val"]), 1)
    p = C.filter_train_records(env["p_val"])
    assert [d["image_id"] for d in q] == [d["image_id"] for d in p]
    RESULTS["filters_on_val"] = {"kept": len(p), "of": len(env["p_val"])}


def _map_both(env, which: str, qd: dict, pd: dict, seed: int):
    _seed(seed)
    qo = env[f"q_mapper_{which}"](copy.deepcopy(qd))
    sq = _rng_state()
    _seed(seed)
    po = env[f"p_mapper_{which}"](pd)
    sp = _rng_state()
    assert sq == sp, "RNG consumption differs"
    return qo, po


def test_train_mapper_equivalent(env, monkeypatch):
    from collections import Counter
    from cxrlandmarks.data import augmentations as PA
    ops: Counter = Counter()
    orig_scale, orig_flip = PA._scale_aug_magnitude, PA.LateralGatedFlip.get_transform

    def scale_spy(aug, mag):
        ops[type(aug).__name__] += 1
        return orig_scale(aug, mag)

    def flip_spy(self, image):
        t = orig_flip(self, image)
        ops["LateralHFlip(applied)"] += int(isinstance(t, PA.HFlipTransform))
        return t

    monkeypatch.setattr(PA, "_scale_aug_magnitude", scale_spy)
    monkeypatch.setattr(PA.LateralGatedFlip, "get_transform", flip_spy)
    stats: dict = {"samples": 0, "laterals": 0, "filtered_instances": 0, "samples_with_filtered_instances": 0,
                   "shape_changed": 0}
    for i, iid in enumerate(env["order"]):
        qo, po = _map_both(env, "train", env["q_train"][iid], env["p_train"][iid], seed=1000 + i)
        _compare(_q_targets(qo), po, stats)
        stats["samples"] += 1
        stats["filtered_instances"] += po["num_filtered"]
        stats["samples_with_filtered_instances"] += int(po["num_filtered"] > 0)
        stats["laterals"] += int(po["view"] == "lateral")
        stats["shape_changed"] += int(tuple(po["image"].shape[1:]) != (512, 512))
    assert stats["samples"] >= 50
    stats["op_counts"] = dict(sorted(ops.items()))
    pool = {"RandomRotationQSeg", "RandomShear", "RandomPerspective", "RandomAspectResize",
            "RandomGaussianNoise", "RandomSaltPepperNoise", "RandomGaussianBlur", "RandomMotionBlur",
            "RandomGammaJitter", "RandomCLAHE", "RandomPoissonNoise", "RandomIntensityRemap",
            "RandomFourierAmplitude", "RandomElasticTransform", "RandomInversion", "RandomErasing"}
    assert pool <= set(ops), f"pool ops never exercised: {pool - set(ops)}"
    assert ops["LateralHFlip(applied)"] > 0
    RESULTS["train_mapper"] = stats


def test_eval_mapper_equivalent(env):
    stats: dict = {"samples": 0}
    for i in range(N_EVAL):
        qo, po = _map_both(env, "eval", env["q_val"][i], env["p_val"][i], seed=5000 + i)
        _compare(_q_targets(qo), po, stats)
        stats["samples"] += 1
    assert stats["samples"] >= 20
    RESULTS["eval_mapper"] = stats


@pytest.mark.parametrize("prob", [0.5, 1.0])
def test_cutmix_equivalent(env, prob):
    from qseg.data.cutmix import apply_cutmix as q_cutmix
    from qseg.engine.trainer import _materialize_gt_masks
    order = env["order"]
    stats = {"batches": 0, "pairs_mixed": 0, "pairs_skipped_shape": 0}
    for b in range(len(order) // 8):
        qb, pb = [], []
        for j, iid in enumerate(order[8 * b: 8 * b + 8]):
            qo, po = _map_both(env, "train", env["q_train"][iid], env["p_train"][iid], seed=20000 + 8 * b + j)
            qb.append(qo)
            pb.append(po)
        before = [p["image"].clone() for p in pb]
        _seed(777 + b)
        q_cutmix(qb, enable=True, prob=prob, alpha=CFG.aug.cutmix_alpha)
        _materialize_gt_masks(qb)
        sq = _rng_state()
        _seed(777 + b)
        PC.apply_cutmix(pb, enable=True, prob=prob, alpha=CFG.aug.cutmix_alpha)
        assert _rng_state() == sq, "CutMix RNG consumption differs"
        cmp_stats: dict = {}
        for q, p in zip(qb, pb):
            _compare(_q_targets(q), p, cmp_stats)
        for i in range(0, 8, 2):
            if not torch.equal(before[i], pb[i]["image"]):
                stats["pairs_mixed"] += 1
            elif before[i].shape != before[i + 1].shape:
                stats["pairs_skipped_shape"] += 1
        stats["batches"] += 1
    assert stats["batches"] >= 10
    RESULTS[f"cutmix_prob{prob}"] = stats


def test_sample_order_matches_detectron2_loader():
    """Port's emitted index order == a real detectron2 train loader (16 workers, batch 8,
    TrainingSampler(seed=42), aspect grouping) over a dummy dataset."""
    from detectron2.data import build_detection_train_loader
    from detectron2.data.samplers import TrainingSampler
    import itertools
    n = 301
    data = [{"idx": i, "width": 512, "height": 512} for i in range(n)]
    for workers in (16, 3):
        loader = build_detection_train_loader(dataset=data, mapper=lambda d: d,
                                              sampler=TrainingSampler(n, seed=42), total_batch_size=8,
                                              aspect_ratio_grouping=True, num_workers=workers)
        got = [[d["idx"] for d in b] for b in itertools.islice(iter(loader), 60)]
        assert got == S.paper_batch_indices(n, 60, seed=42, batch_size=8, num_workers=workers)
        flat = list(itertools.islice(iter(S.PaperOrderSampler(n, seed=42, num_workers=workers)), 480))
        assert flat == [i for b in got for i in b]
    RESULTS["sample_order"] = {"batches_checked": 60, "workers": [16, 3], "n": n}


def test_zz_write_report():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(RESULTS, indent=2, default=str))
    print(json.dumps(RESULTS, indent=2, default=str))
