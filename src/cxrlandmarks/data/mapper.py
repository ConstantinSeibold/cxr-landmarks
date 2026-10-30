"""Per-image mapping: read -> augment -> transform annotations -> dense targets.

Port of qseg ``QSegDatasetMapper`` (``qseg/data/mapper.py``) on top of detectron2 0.6
``DatasetMapper.__call__`` / ``_transform_annotations`` and ``detection_utils``
(``read_image``, ``transform_instance_annotations``, ``transform_keypoint_annotations``,
``annotations_to_instances`` with mask_format="bitmask", ``filter_empty_instances``) and
``BitMasks.get_bounding_boxes``. Behaviour, including its quirks, is kept identical:

* boxes: XYWH->XYXY is done in float32 (detectron2 ``BoxMode.convert`` on a python list
  goes through ``torch.tensor``), each transform re-boxes the 4 corners, then the box is
  clipped to the image; empty-instance filtering runs on THESE boxes (threshold 1e-5,
  float32) and on mask non-emptiness; only AFTER filtering are train boxes recomputed
  from the masks (qseg's 0-instance guard reorders recompute after the filter).
* keypoints: float64 through the transforms; a keypoint outside [0,W]x[0,H] gets v=0 and
  every v==0 keypoint is zeroed (x=y=0); the lateral flip needs no slot swap (identity
  flip indices); stored as float32.
* valid mask: qseg zips the per-annotation ``keypoints_valid_mask`` with the instances
  ONLY if no instance was filtered out; otherwise it falls back to all-ones (K_max)
  for every instance of that image. Reproduced (``valid_mask_fallback_ones``).
"""
from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

from .augmentations import (
    LateralGatedFlip, RandAugment, RandomAspectResize, RandomCLAHE, RandomElasticTransform,
    RandomErasing, RandomFourierAmplitude, RandomGammaJitter, RandomGaussianBlur,
    RandomGaussianNoise, RandomIntensityRemap, RandomInversion, RandomMotionBlur,
    RandomPerspective, RandomPoissonNoise, RandomRotationQSeg, RandomSaltPepperNoise,
    RandomShear,
)
from .coco import view_of
from .config import AugConfig
from .transforms_base import AugInput, AugmentationList, HFlipTransform, ResizeShortestEdge, TransformList

_EXIF_ORIENT = 274


# ----------------------------------------------------------------------------- transforms
def build_train_augmentations(aug: AugConfig) -> tuple[list, LateralGatedFlip | None]:
    """Train aug list in qseg order: resize, (lateral) flip, RandAugment over the pool."""
    augs: list = [ResizeShortestEdge(short_edge_length=aug.input_size, max_size=aug.input_size * 2,
                                     sample_style="choice")]
    lateral_flip = None
    if aug.hflip == "lateral":
        lateral_flip = LateralGatedFlip(prob=aug.hflip_prob)
        augs.append(lateral_flip)
    elif aug.hflip not in ("none", "false", False, None):
        raise ValueError(f"aug.hflip={aug.hflip!r}: only 'lateral' or 'none' are supported")
    pool: list = []
    if aug.rotation_deg > 0:
        pool.append(RandomRotationQSeg(aug.rotation_deg))
    if aug.shear_deg > 0:
        pool.append(RandomShear(aug.shear_deg))
    if aug.perspective_scale > 0:
        pool.append(RandomPerspective(aug.perspective_scale))
    if aug.aspect_jitter_max > 0:
        pool.append(RandomAspectResize(float(aug.aspect_jitter_max), prob=1.0))
    if aug.gaussian_noise_std > 0:
        pool.append(RandomGaussianNoise(aug.gaussian_noise_std, prob=1.0))
    if aug.salt_pepper_density > 0:
        pool.append(RandomSaltPepperNoise(aug.salt_pepper_density, prob=1.0))
    if aug.gaussian_blur_sigma_max > 0:
        pool.append(RandomGaussianBlur(aug.gaussian_blur_sigma_max, prob=1.0))
    if aug.motion_blur_max_kernel >= 3:
        pool.append(RandomMotionBlur(aug.motion_blur_max_kernel, prob=1.0))
    if aug.gamma_max > 0:
        pool.append(RandomGammaJitter(float(aug.gamma_max), prob=1.0))
    if aug.clahe_clip_max > 1.0:
        pool.append(RandomCLAHE(float(aug.clahe_clip_max), prob=1.0))
    if aug.poisson_strength_max > 0:
        pool.append(RandomPoissonNoise(float(aug.poisson_strength_max), prob=1.0))
    if aug.intensity_remap_strength_max > 0:
        pool.append(RandomIntensityRemap(float(aug.intensity_remap_strength_max), prob=1.0))
    if aug.fourier_amp_strength_max > 0:
        pool.append(RandomFourierAmplitude(float(aug.fourier_amp_strength_max), prob=1.0))
    if aug.elastic_alpha_max > 0:
        pool.append(RandomElasticTransform(alpha_max=float(aug.elastic_alpha_max),
                                           sigma=float(aug.elastic_sigma), prob=1.0))
    if aug.inversion_enable:
        pool.append(RandomInversion(prob=1.0))
    if aug.erasing_area_ratio_max > 0:
        pool.append(RandomErasing(
            area_ratio_max=float(aug.erasing_area_ratio_max),
            area_ratio_min=float(aug.erasing_area_ratio_min),
            aspect_ratio_min=float(aug.erasing_aspect_min),
            aspect_ratio_max=float(aug.erasing_aspect_max),
            fill=str(aug.erasing_fill), prob=1.0,
        ))
    if aug.randaugment and pool:
        augs.append(RandAugment(pool=pool, num_ops_min=aug.num_ops_min, num_ops_max=aug.num_ops,
                                magnitude=aug.magnitude, magnitude_min=aug.magnitude_min))
    else:
        augs.extend(pool)
    return augs, lateral_flip


def build_eval_augmentations(aug: AugConfig) -> list:
    return [ResizeShortestEdge(short_edge_length=aug.input_size, max_size=aug.input_size * 2,
                               sample_style="choice")]


# ----------------------------------------------------------------------------- annotations
def read_image_rgb(file_name: str) -> np.ndarray:
    """detectron2 ``read_image(format="RGB")``: PIL open, EXIF transpose, convert RGB."""
    with open(file_name, "rb") as f:
        image = Image.open(f)
        try:
            exif = image.getexif()
        except Exception:
            exif = None
        if exif is not None:
            method = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM,
                      5: Image.TRANSPOSE, 6: Image.ROTATE_270, 7: Image.TRANSVERSE,
                      8: Image.ROTATE_90}.get(exif.get(_EXIF_ORIENT))
            if method is not None:
                image = image.transpose(method)
        return np.asarray(image.convert("RGB"))


def _xywh_to_xyxy_f32(bbox: list[float]) -> list[float]:
    b = np.asarray(bbox, dtype=np.float32)
    b[2] += b[0]
    b[3] += b[1]
    return b.tolist()


def transform_keypoints(keypoints: list[float], transforms: TransformList,
                        image_size: tuple[int, int], hflip_indices: list[int] | None) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints_xy = transforms.apply_coords(keypoints[:, :2])
    inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array(image_size[::-1]))
    inside = inside.all(axis=1)
    keypoints[:, :2] = keypoints_xy
    keypoints[:, 2][~inside] = 0
    do_hflip = sum(isinstance(t, HFlipTransform) for t in transforms.transforms) % 2 == 1
    if do_hflip:
        if hflip_indices is None:
            raise ValueError("Cannot flip keypoints without providing flip indices!")
        if len(keypoints) != len(hflip_indices):
            raise ValueError(f"{len(keypoints)} keypoints vs {len(hflip_indices)} flip indices")
        keypoints = keypoints[np.asarray(hflip_indices, dtype=np.int32), :]
    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints


def transform_annotation(anno: dict, transforms: TransformList, image_size: tuple[int, int],
                         hflip_indices: list[int] | None) -> dict:
    bbox = _xywh_to_xyxy_f32(anno["bbox"])
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    anno["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    seg = anno.get("segmentation")
    if seg is not None:
        if isinstance(seg, list):
            polygons = [np.asarray(p).reshape(-1, 2) for p in seg]
            anno["segmentation"] = [p.reshape(-1) for p in transforms.apply_polygons(polygons)]
        elif isinstance(seg, dict):
            mask = transforms.apply_segmentation(mask_util.decode(seg))
            assert tuple(mask.shape[:2]) == image_size
            anno["segmentation"] = mask
        else:
            raise ValueError(f"Cannot transform segmentation of type {type(seg)}")
    if "keypoints" in anno:
        anno["keypoints"] = transform_keypoints(anno["keypoints"], transforms, image_size, hflip_indices)
    return anno


def _polygons_to_bitmask(polygons: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if len(polygons) == 0:
        return np.zeros((h, w)).astype(bool)
    rles = mask_util.frPyObjects(polygons, h, w)
    return mask_util.decode(mask_util.merge(rles)).astype(bool)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """detectron2 ``BitMasks.get_bounding_boxes`` (tight, exclusive max; empty -> zeros)."""
    boxes = torch.zeros(masks.shape[0], 4, dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(masks.shape[0]):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[i, :] = torch.as_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32)
    return boxes


def annotations_to_targets(annos: list[dict], image_size: tuple[int, int], k_max: int) -> dict[str, torch.Tensor]:
    h, w = image_size
    n = len(annos)
    boxes = (np.stack([a["bbox"] for a in annos]) if n else np.zeros((0, 4)))
    out = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "classes": torch.tensor([int(a["category_id"]) for a in annos], dtype=torch.int64),
    }
    masks = []
    for a in annos:
        seg = a.get("segmentation")
        if isinstance(seg, list):
            masks.append(_polygons_to_bitmask(seg, h, w))
        elif isinstance(seg, dict):
            masks.append(mask_util.decode(seg))
        elif isinstance(seg, np.ndarray):
            masks.append(seg)
        else:
            raise ValueError(f"annotation without a usable segmentation: {type(seg)}")
    out["masks"] = (torch.stack([torch.from_numpy(np.ascontiguousarray(m)) for m in masks]).to(torch.bool)
                    if n else torch.zeros((0, h, w), dtype=torch.bool))
    out["keypoints"] = (torch.as_tensor(np.stack([np.asarray(a["keypoints"]).reshape(-1, 3) for a in annos]),
                                        dtype=torch.float32)
                        if n and "keypoints" in annos[0] else torch.zeros((n, k_max, 3), dtype=torch.float32))
    return out


class LandmarkMapper:
    """Callable record -> sample dict (the qseg QSegDatasetMapper equivalent)."""

    def __init__(self, aug: AugConfig, *, is_train: bool, k_max: int,
                 valid_mask_fallback_ones: bool = True):
        self.is_train = is_train
        self.k_max = int(k_max)
        self.aug_cfg = aug
        if is_train:
            augs, self._lateral_flip = build_train_augmentations(aug)
            self.recompute_boxes = bool(aug.recompute_boxes_train)
        else:
            augs, self._lateral_flip = build_eval_augmentations(aug), None
            self.recompute_boxes = False
        self.augmentations = AugmentationList(augs)
        self.lateral_tag = str(aug.hflip_lateral_tag).lower()
        # qseg passes identity flip indices whenever a flip is configured (no L/R slot swap)
        self.hflip_indices = list(range(self.k_max)) if (aug.hflip == "lateral" and self.k_max > 0) else None
        self.valid_mask_fallback_ones = valid_mask_fallback_ones

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        record = copy.deepcopy(record)
        base = os.path.basename(str(record.get("file_name", ""))).lower()
        if self._lateral_flip is not None:
            self._lateral_flip.active = self.lateral_tag in base
        image = read_image_rgb(record["file_name"])
        if (image.shape[1], image.shape[0]) != (record["width"], record["height"]):
            raise ValueError(f"image size mismatch for {record['file_name']}: "
                             f"{image.shape[1::-1]} vs {(record['width'], record['height'])}")
        aug_input = AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        non_crowd = [a for a in record.pop("annotations", []) if a.get("iscrowd", 0) == 0]
        valid_masks = [a.get("keypoints_valid_mask") for a in non_crowd]
        annos = [transform_annotation(a, transforms, image_shape, self.hflip_indices) for a in non_crowd]
        t = annotations_to_targets(annos, image_shape, self.k_max)
        if len(annos):
            keep = (((t["boxes"][:, 2] - t["boxes"][:, 0]) > 1e-5)
                    & ((t["boxes"][:, 3] - t["boxes"][:, 1]) > 1e-5)
                    & t["masks"].flatten(1).any(dim=1))
            t = {k: v[keep] for k, v in t.items()}
        n = int(t["classes"].shape[0])
        if self.recompute_boxes and n > 0:
            t["boxes"] = masks_to_boxes(t["masks"])
        K = int(t["keypoints"].shape[1])
        if n and len(valid_masks) == n and all(v is not None for v in valid_masks):
            vm = torch.as_tensor(valid_masks, dtype=torch.float32)
            if vm.shape[1] != K:
                vm = torch.nn.functional.pad(vm, (0, max(0, K - vm.shape[1])))[:, :K]
        elif self.valid_mask_fallback_ones:
            vm = torch.ones((n, K), dtype=torch.float32)
        else:
            raise RuntimeError("instances were filtered; strict valid-mask mode cannot realign them")
        return {
            "image": torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "file_name": record["file_name"],
            "image_id": record["image_id"],
            "height": record["height"],
            "width": record["width"],
            "view": view_of(record["file_name"], self.lateral_tag),
            "classes": t["classes"],
            "boxes": t["boxes"],
            "masks": t["masks"],
            "keypoints": t["keypoints"],
            "keypoints_valid": vm,
            "num_filtered": len(annos) - n,
        }
