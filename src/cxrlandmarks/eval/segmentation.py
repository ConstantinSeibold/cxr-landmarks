"""Segmentation AP (COCO mask AP, macro over classes) with bootstrap resampling over images.

COCOeval(segm) runs once; its per-(image, class) matches are then re-weighted per resample. A
resampled image drawn w times contributes w consecutive copies of its detections; the weighted
cumulative sums below are exact for that (within a run of copies the interpolated precision equals
the value at the run's end). With unit weights it reproduces COCOeval exactly.
"""
from __future__ import annotations

import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REC = np.linspace(0, 1, 101)


def prepare(gt: COCO, res_path, img_index: dict, cat_ids=None):
    """res_path: COCO segm results (path, list, or a COCO dataset dict with 'annotations')."""
    with contextlib.redirect_stdout(io.StringIO()):
        res = json.load(open(res_path)) if isinstance(res_path, str) else res_path
        dt = gt.loadRes(res["annotations"] if isinstance(res, dict) else res)
        E = COCOeval(gt, dt, "segm")
        if cat_ids is not None:
            E.params.catIds = sorted(cat_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
    p = E.params
    n_a, n_i = len(p.areaRng), len(p.imgIds)
    out = {}
    for ci, c in enumerate(p.catIds):
        ev = [E.evalImgs[ci * n_a * n_i + ii] for ii in range(n_i)]       # areaRng index 0 = all
        img, sc, tp, ig, gt_img = [], [], [], [], []
        for e in ev:
            if e is None:
                continue
            k = img_index[e["image_id"]]
            gt_img += [k] * int((~np.asarray(e["gtIgnore"], bool)).sum())
            d = len(e["dtScores"])                                       # already capped at maxDet
            img += [k] * d
            sc += list(e["dtScores"])
            tp.append(np.asarray(e["dtMatches"]).reshape(len(p.iouThrs), d) > 0)
            ig.append(np.asarray(e["dtIgnore"]).reshape(len(p.iouThrs), d).astype(bool))
        if not gt_img:
            continue
        o = np.argsort(-np.asarray(sc), kind="mergesort")
        T = len(p.iouThrs)
        tp = np.concatenate(tp, 1) if tp else np.zeros((T, 0), bool)
        ig = np.concatenate(ig, 1) if ig else np.zeros((T, 0), bool)
        out[c] = dict(img=np.asarray(img, int)[o], tp=tp[:, o], ig=ig[:, o], gt_img=np.asarray(gt_img, int))
    return out, 100.0 * E.stats[0]


def ap_weighted(prep, W):
    """W (B, n_images) image multiplicities -> macro segm AP (B,), in %."""
    B = W.shape[0]
    per = []
    for d in prep.values():
        npig = W[:, d["gt_img"]].sum(1)
        wd = W[:, d["img"]]
        aps = np.zeros(B)
        T = d["tp"].shape[0]
        for t in range(T):
            w = wd * ~d["ig"][t]
            tp = np.cumsum(w * d["tp"][t], 1)
            fp = np.cumsum(w * ~d["tp"][t], 1)
            rc = tp / np.maximum(npig, 1)[:, None]
            pr = tp / (tp + fp + np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], 1)[:, ::-1]
            for b in range(B):
                ix = np.searchsorted(rc[b], REC, side="left")
                aps[b] += pr[b, ix[ix < rc.shape[1]]].sum() / len(REC)
        aps /= T
        aps[npig == 0] = np.nan
        per.append(aps)
    return 100.0 * np.nanmean(per, 0)

