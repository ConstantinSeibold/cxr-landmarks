"""Convert a training checkpoint ({"model": EMA state_dict, "epoch": ...}) into the release format:
<out>/model.safetensors + <out>/config.json (+ the vocabulary files the weights were trained with).

    python scripts/export_weights.py --ckpt /path/to/model_best.pth --out weights/joint_dense_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from importlib import resources
from pathlib import Path

import torch
from safetensors.torch import save_file

from cxrlandmarks.landmarks import Channels
from cxrlandmarks.model import JointDenseModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.contiguous() for k, v in ck["model"].items()}
    ch = Channels()
    model = JointDenseModel(ch.n, ch.n_classes, pretrained_encoder=False)
    model.load_state_dict(sd, strict=True)                            # raises on any key or shape mismatch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    vocab = Path(str(resources.files("cxrlandmarks") / "vocab"))
    for f in ("final_vocab_v1.json", "categories_kp_v1.json"):
        shutil.copy(vocab / f, out / f)
    cfg = {"architecture": "JointDenseModel", "encoder": "microsoft/rad-dino", "n_landmarks": ch.n,
           "n_classes": ch.n_classes, "k_max": ch.k_max, "heatmap_stride": 4, "vocabulary": "final_vocab_v1",
           "input": {"format": "RGB uint8", "resize_shortest_edge": 512, "max_edge": 1024, "pad_multiple": 32},
           "training_epoch": ck.get("epoch"), "source_checkpoint_sha256": hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()}
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    print("wrote", out, json.dumps(cfg))


if __name__ == "__main__":
    main()
