"""Compare a regenerated *_kp_v1.json against a reference one.

Usage: python scripts/compare_labels.py NEW.json REFERENCE.json
Exit code 0 iff categories, annotation set, keypoints, num_keypoints and info.side_relabel match.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def compare(new_path: str, ref_path: str) -> bool:
    new, ref = json.load(open(new_path)), json.load(open(ref_path))
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'OK ' if cond else 'DIFF'}] {name}{(': ' + detail) if detail else ''}")

    check("categories (names, keypoints, skeleton, keypoint_groups, all fields)",
          new["categories"] == ref["categories"], f"{len(new['categories'])} vs {len(ref['categories'])}")
    nk = [len(c["keypoints"]) for c in new["categories"]]
    print(f"  K_max (merged frontal+lateral slots per category) = {max(nk)}")
    check("images", new["images"] == ref["images"], f"{len(new['images'])} vs {len(ref['images'])}")

    na = {a["id"]: a for a in new["annotations"]}
    ra = {a["id"]: a for a in ref["annotations"]}
    check("annotation id set", set(na) == set(ra), f"{len(na)} vs {len(ra)}")
    common = set(na) & set(ra)
    kp_mis = nk_mis = other_mis = len_mis = 0
    maxdiff = 0.0
    for i in common:
        a, b = na[i], ra[i]
        ka, kb = a.get("keypoints"), b.get("keypoints")
        if ka is None or kb is None or len(ka) != len(kb):
            len_mis += 1
            kp_mis += 1
        elif ka != kb:
            kp_mis += 1
            maxdiff = max(maxdiff, float(np.abs(np.asarray(ka, float) - np.asarray(kb, float)).max()))
        if a.get("num_keypoints") != b.get("num_keypoints"):
            nk_mis += 1
        rest_a = {k: v for k, v in a.items() if k not in ("keypoints", "num_keypoints")}
        rest_b = {k: v for k, v in b.items() if k not in ("keypoints", "num_keypoints")}
        if rest_a != rest_b:
            other_mis += 1
    check("annotation keypoints", kp_mis == 0,
          f"{kp_mis} mismatching annotations ({len_mis} length mismatches), max abs diff {maxdiff:g}")
    check("num_keypoints", nk_mis == 0, f"{nk_mis} mismatching")
    check("other annotation fields (category_id after relabel, bbox, segmentation, ...)", other_mis == 0,
          f"{other_mis} mismatching")
    check("info.side_relabel", new.get("info", {}).get("side_relabel") == ref.get("info", {}).get("side_relabel"),
          json.dumps(new.get("info", {}).get("side_relabel", {}).get("swapped")))
    check("info (whole block)", new.get("info") == ref.get("info"))
    check("whole json identical", new == ref)
    return ok


if __name__ == "__main__":
    sys.exit(0 if compare(sys.argv[1], sys.argv[2]) else 1)
