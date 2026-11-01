"""Build the gold-standard json from the two readers' annotations (PhysioNet release).

    cxrlm-build-gold --release /data/cxr-landmarks-gold --out gold_kp.json [--sigma-out sigma.csv]

Release layout (MIMIC-CXR-derived, credentialed access on PhysioNet):
  images.csv          image_id, dicom_id, ..., view (frontal | lateral); coordinates are in the
                      512x512 frame of the image (see orig_width / orig_height to map back)
  readers/R1.csv      one row per (image, landmark) a reader answered: dicom_id, view, class,
  readers/R2.csv      landmark, status (ok | moved | added | not_visible), x, y, ...
  scale.csv           dicom_id, class, area, bbox_x, bbox_y, bbox_w, bbox_h: the object scale (s_j^2
                      in Eq. 2, PCK box) from the structure mask the readers saw (no GT masks exist)

Per (image, class, landmark) read by both readers:
  both visible         -> mean of the two points, v=2
  either not visible   -> v=0 (not scored)
--include-single also keeps landmarks only one reader answered (that reader's point; not in the paper).
Per-landmark tolerance sigma_yk (Eq. 2): from the paired differences d of landmarks both readers saw,
sigma^2 = E[d^2] / (4 s^2); landmarks with fewer than --min-pairs pairs get the median sigma. Stored as
categories[].oks_sigmas (pycocotools uses kappa = 2 sigma).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..landmarks import _vocab_file

READERS = ("R1", "R2")


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build(release: Path, include_single: bool = False, min_pairs: int = 5) -> tuple[dict, list[dict]]:
    images = _read(release / "images.csv")
    scale = {(r["dicom_id"], r["class"]): (float(r["area"]), [float(r[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")])
             for r in _read(release / "scale.csv")}
    cats = json.load(open(_vocab_file("categories_kp_v1.json")))
    cats = cats if isinstance(cats, list) else cats["categories"]
    by_name = {c["name"]: c for c in cats}

    pts: dict[tuple, dict] = defaultdict(dict)                 # (dicom, view, class, kp) -> reader -> (vis, x, y)
    for r in READERS:
        for row in _read(release / "readers" / f"{r}.csv"):
            vis = row["status"] != "not_visible"
            xy = (float(row["x"]), float(row["y"])) if vis else (np.nan, np.nan)
            pts[(row["dicom_id"], row["view"], row["class"], row["landmark"])][r] = (vis, *xy)

    # sigma from landmarks both readers saw
    dn = defaultdict(list)
    for (dic, _, cls, kp), rr in pts.items():
        if len(rr) == 2 and all(v[0] for v in rr.values()):
            (_, x1, y1), (_, x2, y2) = rr["R1"], rr["R2"]
            s = np.sqrt(scale[(dic, cls)][0]) if (dic, cls) in scale else np.nan
            dn[(cls, kp)].append(np.hypot(x1 - x2, y1 - y2) / s)
    table = [{"class": c, "landmark": k, "n_pairs": len(v), "sigma": float(np.sqrt(np.nanmean(np.square(v)) / 4))}
             for (c, k), v in sorted(dn.items())]
    med = float(np.median([t["sigma"] for t in table if t["n_pairs"] >= min_pairs]))
    for t in table:
        t["sigma_used"] = t["sigma"] if t["n_pairs"] >= min_pairs else med
    sig = {(t["class"], t["landmark"]): t["sigma_used"] for t in table}

    iid = {im["dicom_id"]: int(im["image_id"]) for im in images}
    anns: dict[tuple, dict] = {}
    for (dic, view, cls, kp), rr in sorted(pts.items()):
        if len(rr) < 2 and not include_single:
            continue
        cat = by_name[cls]
        key = (dic, cat["id"])
        if key not in anns:
            if (dic, cls) not in scale:
                continue
            area, bbox = scale[(dic, cls)]
            anns[key] = {"image_id": iid[dic], "category_id": cat["id"], "keypoints": [0.0] * (3 * len(cat["keypoints"])),
                         "area": area, "bbox": bbox, "iscrowd": 0, "num_keypoints": 0}
        if not all(v[0] for v in rr.values()):
            continue
        j = cat["keypoints"].index(kp)
        x = float(np.mean([v[1] for v in rr.values()]))
        y = float(np.mean([v[2] for v in rr.values()]))
        a = anns[key]
        a["keypoints"][3 * j:3 * j + 3] = [x, y, 2]
        a["num_keypoints"] += 1
    out_anns = [a for a in anns.values() if a["num_keypoints"] > 0]
    for n, a in enumerate(out_anns, start=1):
        a["id"] = n
    out_cats = copy.deepcopy(cats)
    for c in out_cats:
        if c.get("keypoints"):
            c["oks_sigmas"] = [float(sig.get((c["name"], k), med)) for k in c["keypoints"]]
    ims = [{"id": int(im["image_id"]), "file_name": f"{im['dicom_id']}_{im['view']}.jpg", "dicom_id": im["dicom_id"],
            "view": im["view"], "width": int(im["width"]), "height": int(im["height"])} for im in images]
    gold = {"info": {"description": "cxr-landmarks gold standard: mean of two readers"
                     + (" (incl. single reads)" if include_single else ""), "sigma_median": med},
            "images": ims, "annotations": out_anns, "categories": out_cats}
    return gold, table


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path, help="unpacked PhysioNet release")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sigma-out", type=Path)
    ap.add_argument("--include-single", action="store_true")
    ap.add_argument("--min-pairs", type=int, default=5)
    a = ap.parse_args(argv)
    gold, table = build(a.release, a.include_single, a.min_pairs)
    json.dump(gold, open(a.out, "w"))
    if a.sigma_out:
        with open(a.sigma_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    n_lm = sum(a_["num_keypoints"] for a_ in gold["annotations"])
    print(f"{len({a_['image_id'] for a_ in gold['annotations']})} images, {len(gold['annotations'])} annotations, "
          f"{n_lm} landmarks; median sigma {gold['info']['sigma_median']:.4f}")


if __name__ == "__main__":
    main()
