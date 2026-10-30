"""``PaxRayLandmarkDataset`` + collate / batch padding helpers.

Sample dict (one image, after augmentation; all spatial values in INPUT pixels):
  image            uint8 (3, H, W)   RGB, H/W = augmented input size (512 +- aspect jitter)
  file_name, image_id, height, width   original record fields (height/width = ORIGINAL size)
  view             "frontal" | "lateral" (basename contains "lateral")
  classes          int64 (N,)        contiguous class index, categories sorted by COCO id
  boxes            float32 (N, 4)    xyxy; train: tight box of the mask; eval: transformed bbox
  masks            bool (N, H, W)    per-instance binary masks
  keypoints        float32 (N, K_max, 3)  (x, y, v); v=0 slots have x=y=0
  keypoints_valid  float32 (N, K_max)     1 where the slot exists for the class (see mapper
                                          for the qseg fallback-to-ones quirk)

Training-loop responsibilities (outside the dataset, in this order, as the paper run did):
  batch = next(loader)                       # list of 8 sample dicts (collate_list)
  apply_cutmix(batch, prob=0.5, alpha=1.0)   # cutmix.py, main process, python random + numpy
  x = pad_batch([d["image"] for d in batch]) # float32, zero-padded to a multiple of 32
Masks are already dense (qseg's ``_materialize_gt_masks`` = BitMasks -> tensor is a no-op).
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .coco import CategoryInfo, build_records, filter_train_records, load_json, scan_categories
from .config import DataConfig
from .mapper import LandmarkMapper


class PaxRayLandmarkDataset(torch.utils.data.Dataset):
    """Map-style dataset over a PAXRay++ keypoint json.

    ``split`` picks the json from ``cfg`` ("train" | "val" | "test") unless ``json_path`` is
    given. Categories / K_max always come from ``categories_from`` (default: the train json;
    qseg scans json_train for every split) -- pass a ``CategoryInfo`` to avoid re-reading it.
    Train records are filtered like detectron2's train loader (non-crowd annotation present,
    >= 1 visible keypoint); eval records are not filtered.
    """

    def __init__(self, cfg: DataConfig | None = None, split: str = "train", *,
                 json_path: str | None = None, categories: CategoryInfo | None = None,
                 records: list[dict[str, Any]] | None = None, is_train: bool | None = None):
        self.cfg = cfg or DataConfig()
        self.split = split
        self.is_train = (split == "train") if is_train is None else bool(is_train)
        if records is None:
            path = json_path or {"train": self.cfg.json_train, "val": self.cfg.json_val,
                                 "test": self.cfg.json_test}[split]
            coco = load_json(path)
            if categories is None:
                categories = (scan_categories(coco) if path == self.cfg.json_train
                              else scan_categories(load_json(self.cfg.json_train)))
            records = build_records(coco, self.cfg.image_root, categories)
            del coco
            if self.is_train:
                records = filter_train_records(records, filter_empty=self.cfg.filter_empty,
                                               min_keypoints=self.cfg.min_keypoints_per_image)
        assert categories is not None, "pass categories together with records"
        self.categories = categories
        self.records = records
        self.mapper = LandmarkMapper(self.cfg.aug, is_train=self.is_train, k_max=categories.k_max)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.mapper(self.records[idx])


def collate_list(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No collation (detectron2 ``trivial_batch_collator``): instance counts differ per image."""
    return batch


def pad_batch(images: list[torch.Tensor], mult: int = 32) -> torch.Tensor:
    """float32 (B, 3, H, W), zero-padded bottom/right to a multiple of ``mult`` (paper's
    ``flat_heatmap.pad_batch``); pixel values stay in [0, 255] (normalise in the model)."""
    H = math.ceil(max(i.shape[1] for i in images) / mult) * mult
    W = math.ceil(max(i.shape[2] for i in images) / mult) * mult
    out = images[0].new_zeros((len(images), 3, H, W), dtype=torch.float32)
    for b, im in enumerate(images):
        out[b, :, :im.shape[1], :im.shape[2]] = im.float()
    return out
