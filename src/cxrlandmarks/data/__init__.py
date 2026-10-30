"""Training data pipeline of the paper's joint dense model (detectron2/qseg-free port)."""
from __future__ import annotations

from .coco import CategoryInfo, build_records, filter_train_records, load_json, scan_categories, view_of
from .config import AugConfig, DataConfig, paper_config
from .cutmix import apply_cutmix
from .dataset import PaxRayLandmarkDataset, collate_list, pad_batch
from .mapper import LandmarkMapper, build_eval_augmentations, build_train_augmentations
from .sampling import PaperOrderSampler, paper_batch_indices, training_index_stream, worker_init_fn

__all__ = [
    "AugConfig", "CategoryInfo", "DataConfig", "LandmarkMapper", "PaperOrderSampler",
    "PaxRayLandmarkDataset", "apply_cutmix", "build_eval_augmentations", "build_records",
    "build_train_augmentations", "collate_list", "filter_train_records", "load_json",
    "pad_batch", "paper_batch_indices", "paper_config", "scan_categories",
    "training_index_stream", "view_of", "worker_init_fn",
]
