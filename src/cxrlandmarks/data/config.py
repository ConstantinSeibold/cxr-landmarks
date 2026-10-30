"""Data / augmentation config with the EXACT effective values of the paper run.

Paper run: ``kp_prep/baselines/mtl_baselines.py --arch dense`` (defaults: --batch 8,
--workers 16, --seed 42), which composes the qseg Hydra config
``experiments/kp_isbi_ours_raddino`` (= ``configs/base.yaml`` + ``backbone/raddino`` +
the experiment yaml) and translates it with ``qseg.models.build._make_d2_cfg`` into
``d2.INPUT.QSEG_AUG`` (read by ``qseg.data.mapper.build_qseg_mapper``).

Origin legend used below:
  [exp]   configs/experiments/kp_isbi_ours_raddino.yaml (``aug:`` / ``data:`` block)
  [base]  configs/base.yaml default (not overridden by the experiment)
  [d2cfg] hard-coded in qseg.models.build._make_d2_cfg / detectron2 0.6 defaults
  [mapper] hard-coded in qseg.data.mapper.build_qseg_mapper
  [mtl]   kp_prep/baselines/mtl_baselines.py (argparse default / code)
  [scan]  derived from train_kp_v1.json by qseg.data.utils.scan_coco_json
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

PAXRAY_ROOT = os.environ.get("PAXRAY_ROOT", "paxraypp")          # PAXRay++ root (images + annotations/)


@dataclass
class AugConfig:
    # resize: ResizeShortestEdge(short=input_size, max=2*input_size, "choice") [mapper]
    input_size: int = 512                    # [exp] aug.input_size=[512,512] -> INPUT.IMAGE_SIZE
    # flip: aug.hflip="lateral" -> LateralGatedFlip(prob=0.5) gated on basename tag [exp][mapper]
    hflip: str = "lateral"                   # "lateral" | "none" (all-image flip unsupported: unsafe on frontals)
    hflip_prob: float = 0.5                  # [mapper] LateralGatedFlip(prob=0.5)
    hflip_lateral_tag: str = "lateral"       # [base] aug.hflip_lateral_tag
    # RandAugment: k ~ randint(num_ops_min, num_ops), magnitude ~ U(magnitude_min, magnitude)
    randaugment: bool = True                 # [exp] aug.randaugment.enable
    num_ops: int = 4                         # [exp] aug.randaugment.num_ops
    num_ops_min: int = 2                     # [exp] aug.randaugment.num_ops_min
    magnitude: float = 1.0                   # [exp] aug.randaugment.magnitude
    magnitude_min: float = 0.5               # [exp] aug.randaugment.magnitude_min
    # pool members (an op joins the pool iff its strength enables it; order = pool order)
    rotation_deg: float = 10.0               # [exp]
    shear_deg: float = 30.0                  # [exp]
    perspective_scale: float = 0.02          # [exp]
    aspect_jitter_max: float = 0.1           # [exp]
    gaussian_noise_std: float = 8.0          # [exp]
    salt_pepper_density: float = 0.005       # [exp]
    gaussian_blur_sigma_max: float = 1.5     # [exp]
    motion_blur_max_kernel: int = 7          # [exp]
    gamma_max: float = 0.5                   # [exp]
    clahe_clip_max: float = 4.0              # [exp] (joins pool iff > 1)
    poisson_strength_max: float = 0.6        # [exp]
    intensity_remap_strength_max: float = 0.15  # [exp]
    fourier_amp_strength_max: float = 0.3    # [exp]
    elastic_alpha_max: float = 60.0          # [exp]
    elastic_sigma: float = 40.0              # [exp]
    inversion_enable: bool = True            # [exp]
    erasing_area_ratio_max: float = 0.2      # [exp]
    erasing_area_ratio_min: float = 0.02     # [exp]
    erasing_aspect_min: float = 0.3          # [exp]
    erasing_aspect_max: float = 3.3          # [exp]
    erasing_fill: str = "random"             # [exp]
    # NOT in the paper pool (0 = off): windowing_strength_max, lens_distortion_max,
    # scale_translate_max, grid_distortion_max [base]; lsj is dead code in qseg (no caller).
    # Per-op *_prob values are ignored under RandAugment (all pool ops use prob=1.0) [mapper].
    # CutMix (batch level, applied by the training loop after collation) [exp]
    cutmix: bool = True                      # [exp] aug.cutmix.enable
    cutmix_prob: float = 0.5                 # [exp] aug.cutmix.prob
    cutmix_alpha: float = 1.0                # [exp] aug.cutmix.alpha
    # box handling
    recompute_boxes_train: bool = True       # [d2cfg] MODEL.MASK_ON=True (set by mtl) -> mapper recompute_boxes


@dataclass
class DataConfig:
    image_root: str = PAXRAY_ROOT                                         # [exp] data.image_root
    json_train: str = f"{PAXRAY_ROOT}/annotations/train_kp_v1.json"      # [exp]
    json_val: str = f"{PAXRAY_ROOT}/annotations/val_kp_v1.json"          # [exp]
    json_test: str = f"{PAXRAY_ROOT}/annotations/test_kp_v1.json"        # [exp]
    batch_size: int = 8                      # [mtl] --batch 8 -> SOLVER.IMS_PER_BATCH
    num_workers: int = 16                    # [mtl] --workers 16 -> DATALOADER.NUM_WORKERS
    seed: int = 42                           # [mtl] --seed 42 -> d2.SEED (TrainingSampler seed)
    # train-set filtering (detectron2 get_detection_dataset_dicts) [d2cfg]
    filter_empty: bool = True                # DATALOADER.FILTER_EMPTY_ANNOTATIONS
    min_keypoints_per_image: int = 1         # ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE (KEYPOINT_ON)
    aspect_ratio_grouping: bool = True       # DATALOADER.ASPECT_RATIO_GROUPING (no-op: all 512x512)
    # image format / masks [d2cfg]: INPUT.FORMAT="RGB", INPUT.MASK_FORMAT="bitmask"
    # K_max = 13 [scan] is read from the json at dataset construction.
    aug: AugConfig = field(default_factory=AugConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def paper_config(paxray_root: str | None = None) -> DataConfig:
    """The paper's effective data config (all defaults), with the PAXRay++ root substituted."""
    r = paxray_root or PAXRAY_ROOT
    return DataConfig(image_root=r, json_train=f"{r}/annotations/train_kp_v1.json",
                      json_val=f"{r}/annotations/val_kp_v1.json", json_test=f"{r}/annotations/test_kp_v1.json")
