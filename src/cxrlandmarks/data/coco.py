"""COCO keypoint-json reading: category scan, per-image records, train-set filtering.

Port of qseg ``data/utils.py:scan_coco_json`` (class/keypoint schema part),
``data/coco_kpt_dataset.py:build_dataset_dicts`` (binary=False, no SSL/extra keys) and
detectron2 0.6 ``data/build.py`` filters (``filter_images_with_only_crowd_annotations``,
``filter_images_with_few_keypoints``).

Records follow the detectron2 dataset-dict layout so the mapper port can be checked
field-by-field against qseg: per annotation ``category_id`` is the CONTIGUOUS class index
(categories sorted by COCO id), ``bbox`` is XYWH_ABS, ``keypoints`` is padded to
``3*K_max`` and ``keypoints_valid_mask`` marks the slots that belong to the category.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_util

IGNORE_CATEGORY_NAME = "__ignore__"


@dataclass
class CategoryInfo:
    cat_ids: list[int]                       # contiguous index -> COCO category id
    names: list[str]
    keypoint_names: list[list[str]]          # per contiguous class ([] if none)
    k_max: int

    @property
    def num_classes(self) -> int:
        return len(self.cat_ids)

    @property
    def cat_id_to_contig(self) -> dict[int, int]:
        return {c: i for i, c in enumerate(self.cat_ids)}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def scan_categories(coco: dict[str, Any]) -> CategoryInfo:
    cats = sorted([c for c in coco.get("categories", []) if c.get("name") != IGNORE_CATEGORY_NAME],
                  key=lambda c: c["id"])
    kp = [list(c.get("keypoints") or []) for c in cats]
    return CategoryInfo(
        cat_ids=[int(c["id"]) for c in cats],
        names=[c["name"] for c in cats],
        keypoint_names=kp,
        k_max=max([len(k) for k in kp] + [0]),
    )


def _seg_to_record(seg: Any, h: int, w: int) -> Any:
    if isinstance(seg, list):
        return seg
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            return mask_util.frPyObjects(seg, h, w)
        return seg
    raise ValueError(f"Unrecognized segmentation type: {type(seg)}")


def build_records(coco: dict[str, Any], image_root: str | Path,
                  cats: CategoryInfo | None = None) -> list[dict[str, Any]]:
    """Per-image records in json image order (== qseg build_dataset_dicts, binary=False).

    ``cats`` must come from the TRAIN json (qseg scans json_train for every split).
    """
    image_root = Path(image_root)
    cats = cats or scan_categories(coco)
    c2i = cats.cat_id_to_contig
    k_max = cats.k_max
    has_kp = k_max > 0
    k_per_cat = [len(k) for k in cats.keypoint_names]

    by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        by_img[ann["image_id"]].append(ann)

    records: list[dict[str, Any]] = []
    for img in coco.get("images", []):
        rec: dict[str, Any] = {
            "file_name": str(image_root / img["file_name"]),
            "image_id": int(img["id"]),
            "height": int(img["height"]),
            "width": int(img["width"]),
        }
        anns = []
        for ann in by_img.get(img["id"], []):
            cat_id = int(ann["category_id"])
            if cat_id not in c2i:
                continue
            contig = c2i[cat_id]
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue
            a: dict[str, Any] = {
                "category_id": contig,
                "bbox": list(map(float, bbox)),         # XYWH_ABS
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
            seg = ann.get("segmentation")
            if seg is not None:
                a["segmentation"] = _seg_to_record(seg, img["height"], img["width"])
            if has_kp:
                kpts = ann.get("keypoints")
                k_c = k_per_cat[contig]
                padded = np.zeros((k_max, 3), dtype=np.float32)
                valid = np.zeros((k_max,), dtype=np.float32)
                if kpts and k_c > 0:
                    arr = np.asarray(kpts, dtype=np.float32).reshape(-1, 3)
                    if arr.shape[0] != k_c:
                        arr = arr[:k_c]                  # qseg tolerates schema drift this way
                    padded[:arr.shape[0]] = arr
                    valid[:arr.shape[0]] = 1.0
                a["keypoints"] = padded.reshape(-1).tolist()
                a["keypoints_valid_mask"] = valid.tolist()
            anns.append(a)
        rec["annotations"] = anns
        records.append(rec)
    return records


def filter_train_records(records: list[dict[str, Any]], *, filter_empty: bool = True,
                         min_keypoints: int = 1) -> list[dict[str, Any]]:
    """detectron2 train-set filters, in detectron2's order (crowd-only, then few keypoints)."""
    if filter_empty:
        records = [r for r in records if any(a.get("iscrowd", 0) == 0 for a in r["annotations"])]
    if min_keypoints > 0:
        def n_vis(r):
            return sum((np.array(a["keypoints"][2::3]) > 0).sum()
                       for a in r["annotations"] if "keypoints" in a)
        records = [r for r in records if n_vis(r) >= min_keypoints]
    return records


def view_of(file_name: str, tag: str = "lateral") -> str:
    """View from the file-name BASENAME (a parent dir called 'lateral' must not count)."""
    import os
    return "lateral" if tag in os.path.basename(str(file_name)).lower() else "frontal"
