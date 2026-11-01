"""Write the gold-standard reading frame: each MIMIC-CXR-JPG image of the PhysioNet release resized to
512x512 (aspect ratio not preserved, as read), named <dicom_id>_<view>.jpg as in gold_kp.json.

    python scripts/prepare_gold_images.py --release /data/cxr-landmarks-gold \
        --mimic-cxr-jpg /data/mimic-cxr-jpg/2.1.0 --out /data/gold_512
    cxrlm-predict --gt gold_kp.json --image-root /data/gold_512 --out-dir preds/gold

The readers saw an existing 512x512 copy whose interpolation is not recorded; any standard resize
gives the same coordinate frame.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--mimic-cxr-jpg", required=True, type=Path, help="folder containing files/p10/...")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.release / "images.csv")))
    for r in rows:
        im = Image.open(a.mimic_cxr_jpg / r["mimic_cxr_jpg_path"]).convert("RGB")
        im.resize((int(r["width"]), int(r["height"])), Image.BICUBIC).save(a.out / f"{r['dicom_id']}_{r['view']}.jpg", quality=95)
    print(f"wrote {len(rows)} images to {a.out}")


if __name__ == "__main__":
    main()
