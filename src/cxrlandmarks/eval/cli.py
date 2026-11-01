"""Evaluate landmark (and optionally mask) predictions against a kp json, as in the paper's Table 1.

    cxrlm-evaluate --gt test_kp_v1.json --keypoints predictions/keypoints.json \
        [--segm predictions/segm.json] [--bootstrap 1000] [--out metrics.json]

Gold standard (real CXRs, --gold): the gold json is partial (only some landmark groups were read), so
detections are restricted to the (image, class) pairs it annotates, OKS uses the per-landmark
inter-reader sigmas stored in its categories ("oks_sigmas"), classes with fewer than --min-anns
annotations are dropped, a class absent from a resample is skipped in PCK, and the mean radial error
(MRE, pixels) of the top-scoring detection is reported.

    cxrlm-evaluate --gold --gt gold_kp.json --keypoints predictions/keypoints.json --bootstrap 1000

Bootstrap: 95% percentile CIs over test images; resamples are drawn as in the paper (images in
sorted-id order, numpy default_rng(seed), B resamples), so CIs of different methods are paired.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile

import numpy as np

from . import keypoints as KP


def resample_weights(image_ids: list[int], B: int, seed: int = 0) -> np.ndarray:
    ids = sorted(image_ids)
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                     for _ in range(B)]).astype(np.float64)


def _gold_subset(gt: dict, min_anns: int) -> tuple[dict, list[str]]:
    n = {}
    for a in gt["annotations"]:
        n[a["category_id"]] = n.get(a["category_id"], 0) + 1
    names = {c["id"]: c["name"] for c in gt["categories"]}
    dropped = sorted(f"{names[c]} ({k})" for c, k in n.items() if k < min_anns)
    return dict(gt, annotations=[a for a in gt["annotations"] if n[a["category_id"]] >= min_anns]), dropped


def evaluate(gt_json: str, kp_preds: list[dict] | str, segm_preds=None, B: int = 0, seed: int = 0,
             gold: bool = False, min_anns: int = 5) -> dict:
    gt = json.load(open(gt_json))
    tmp = None
    if gold:
        gt, dropped = _gold_subset(gt, min_anns)
        fd, tmp = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(gt, f)
        gt_json = tmp
    K = {c["id"]: len(c["keypoints"]) for c in gt["categories"] if c.get("keypoints")}
    # gold: resample the annotated images only (the json may list images without gold landmarks)
    ids = sorted({a["image_id"] for a in gt["annotations"]} if gold else (im["id"] for im in gt["images"]))
    img_index = {i: n for n, i in enumerate(ids)}
    if isinstance(kp_preds, str):
        kp_preds = json.load(open(kp_preds))
    preds = KP.load_list(kp_preds, K)
    sigmas = None
    if gold:
        have = {(a["image_id"], a["category_id"]) for a in gt["annotations"]}
        preds = [p for p in preds if (p["image_id"], p["category_id"]) in have]
        sigmas = KP.class_sigmas(gt)
    try:
        out = {"oks_ap": KP.macro_oks_ap(gt_json, preds)["keypoints/AP"], "pck": KP.pck(preds, gt)}
    finally:
        if tmp:
            os.remove(tmp)
    if gold:
        mp = KP.mre_parts(preds, gt, img_index)
        out["mre"] = float(KP.mre_weighted(mp, np.ones((1, len(ids))))[0])
        out["mre_coverage"] = float(mp[2].sum() / max(mp[3].sum(), 1))
        out["dropped_classes"] = dropped
    W = resample_weights(ids, B, seed) if B else None
    if W is not None:
        prep = KP.prepare(preds, gt, img_index, sigmas)
        ap1 = float(KP.ap_weighted(prep, np.ones((1, len(ids))))[0])
        assert abs(ap1 - out["oks_ap"]) < 1e-6, (ap1, out["oks_ap"])     # exact COCOeval reproduction
        ap_b = np.concatenate([KP.ap_weighted(prep, W[s:s + 100]) for s in range(0, B, 100)])
        pck_b = KP.pck_weighted(KP.pck_parts(preds, gt, img_index), W, skip_absent=gold)
        out["oks_ap_ci"] = np.percentile(ap_b, [2.5, 97.5]).tolist()
        out["pck_ci"] = np.percentile(pck_b, [2.5, 97.5]).tolist()
        if gold:
            out["mre_ci"] = np.percentile(KP.mre_weighted(mp, W), [2.5, 97.5]).tolist()
    if isinstance(segm_preds, str):
        segm_preds = json.load(open(segm_preds))
    if segm_preds is not None and len(segm_preds) == 0:          # no masks at all (e.g. an untrained model)
        out["segm_ap"] = 0.0
    elif segm_preds is not None:
        from pycocotools.coco import COCO
        from . import segmentation as SG
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(gt_json)
        prep, ref = SG.prepare(coco, segm_preds, img_index)
        out["segm_ap"] = ref
        if W is not None:
            seg_b = SG.ap_weighted(prep, W)
            out["segm_ap_ci"] = np.percentile(seg_b, [2.5, 97.5]).tolist()
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="ground-truth kp json (e.g. test_kp_v1.json)")
    ap.add_argument("--keypoints", required=True, help="COCO keypoint results json")
    ap.add_argument("--segm", help="COCO segmentation results json")
    ap.add_argument("--bootstrap", type=int, default=0, help="number of image resamples for 95%% CIs (paper: 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gold", action="store_true", help="partial gold standard with per-landmark sigmas (see above)")
    ap.add_argument("--min-anns", type=int, default=5, help="--gold: drop classes with fewer annotations")
    ap.add_argument("--out", help="write the metrics here")
    a = ap.parse_args(argv)
    m = evaluate(a.gt, a.keypoints, a.segm, a.bootstrap, a.seed, gold=a.gold, min_anns=a.min_anns)
    if m.get("dropped_classes"):
        print("dropped (too few annotations):", ", ".join(m["dropped_classes"]))
    for k in ("oks_ap", "pck", "mre", "segm_ap"):
        if k in m:
            ci = m.get(f"{k}_ci")
            print(f"{k:8s} {m[k]:6.2f}" + (f"  [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
    if a.out:
        json.dump(m, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
