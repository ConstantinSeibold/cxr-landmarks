"""Landmark metrics as reported in the paper (Table 1).

OKS-AP: COCO keypoint AP per landmark class (uniform OKS sigma 0.05, all predictions), macro mean
over classes. PCK@0.1: top-scoring prediction per (image, class); a landmark is correct within
0.1 x the diagonal of its structure's ground-truth box; a missed structure counts all its landmarks
as wrong; mean over landmarks per class, macro over classes.

Bootstrap CIs: an exact re-implementation of pycocotools' keypoint evaluation for this setting
(at most one GT per image x class; maxDets 20; area range "all"), factored so that resampling
images is a reweighting of a fixed, score-sorted detection list; with unit weights it reproduces
COCOeval exactly (checked in `bootstrap`).

Gold standard (real CXRs): per-landmark tolerances from the categories' "oks_sigmas" (inter-reader
sigma; pass them as `sigmas`), and sparse annotations, so `pck_weighted(..., skip_absent=True)`
skips a class that a resample does not contain instead of scoring it 0.
"""
from __future__ import annotations

import contextlib
import io
import json
from collections import defaultdict

import numpy as np

THRS = np.linspace(0.5, 0.95, 10)
REC = np.linspace(0.0, 1.0, 101)


def macro_oks_ap(gt_json: str, preds: list[dict], default_sigma: float = 0.05) -> dict:
    """Per-category COCOeval(keypoints), macro mean; mirrors qseg KeypointOKSEvaluator."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        dt = gt.loadRes(preds) if preds else None
    if dt is None:
        return {"keypoints/AP": float("nan")}
    res, aps = {}, []
    for cat in gt.loadCats(gt.getCatIds()):
        cid, K = cat["id"], len(cat.get("keypoints") or [])
        if K == 0 or cid not in dt.getCatIds():
            continue
        ev = COCOeval(gt, dt, iouType="keypoints")
        ev.params.kpt_oks_sigmas = np.array(cat.get("oks_sigmas") or [default_sigma] * K, dtype=np.float32)
        ev.params.catIds = [cid]
        ids = sorted(set(gt.getImgIds(catIds=[cid])) | {a["image_id"] for a in dt.loadAnns(dt.getAnnIds(catIds=[cid]))})
        ev.params.imgIds = ids
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(); ev.accumulate(); ev.summarize()
        s = [x * 100 if x >= 0 else x for x in ev.stats[:3]]
        res[f"keypoints/{cat['name']}/AP"] = s[0]
        aps.append(s)
    ok = [a for a in aps if a[0] >= 0]
    for j, k in enumerate(("AP", "AP50", "AP75")):
        res[f"keypoints/{k}"] = float(np.mean([a[j] for a in ok])) if ok else float("nan")
    res["keypoints/num_cats_with_data"] = len(ok)
    return res



def load(path, K):
    """-> COCO keypoint results with exactly K_c (x, y, v) triplets. qseg's predictions.json
    stores (x, y) for all K_max slots; the others already store K_c triplets."""
    return load_list(json.load(open(path)), K)


def load_list(preds, K):
    out = []
    for p in preds:
        c = p["category_id"]
        if c not in K or not p.get("keypoints"):
            continue
        a = np.asarray(p["keypoints"], dtype=float)
        xy = a.reshape(-1, 3)[:, :2] if len(a) == 3 * K[c] else a.reshape(-1, 2)[:K[c]]
        out.append(dict(image_id=p["image_id"], category_id=c, score=float(p["score"]),
                        keypoints=np.c_[xy, np.full(len(xy), 2.0)].ravel().tolist()))
    return out


def pck(preds, gt, alpha=0.1):
    best = {}
    for p in preds:
        k = (p["image_id"], p["category_id"])
        if k not in best or p["score"] > best[k]["score"]:
            best[k] = p
    hits = defaultdict(list)
    for a in gt["annotations"]:
        if a.get("num_keypoints", 0) == 0:
            continue
        g = np.asarray(a["keypoints"], dtype=float).reshape(-1, 3)
        v = g[:, 2] > 0
        diag = float(np.hypot(a["bbox"][2], a["bbox"][3]))
        p = best.get((a["image_id"], a["category_id"]))
        if p is None:
            hits[a["category_id"]] += [0.0] * int(v.sum())
            continue
        d = np.hypot(*(np.asarray(p["keypoints"]).reshape(-1, 3)[:, :2] - g[:, :2]).T)
        hits[a["category_id"]] += (d[v] <= alpha * diag).astype(float).tolist()
    return 100.0 * float(np.mean([np.mean(h) for h in hits.values() if h]))


def _oks(dxy, g, area, sigma=0.05):
    xg, yg, vg = g[:, 0], g[:, 1], g[:, 2]
    var = (2 * sigma) ** 2
    if (vg > 0).any():
        dx, dy = dxy[:, 0] - xg, dxy[:, 1] - yg
        e = (dx ** 2 + dy ** 2) / var / (area + np.spacing(1)) / 2
        e = e[vg > 0]
    else:
        return None                                   # box-based OKS, only for ignored GT
    return float(np.exp(-e).sum() / e.shape[0])


def _oks_ignored(dxy, gt_ann, sigma=0.05):
    bb = gt_ann["bbox"]
    x0, x1, y0, y1 = bb[0] - bb[2], bb[0] + bb[2] * 2, bb[1] - bb[3], bb[1] + bb[3] * 2
    z = np.zeros(len(dxy))
    dx = np.maximum(z, x0 - dxy[:, 0]) + np.maximum(z, dxy[:, 0] - x1)
    dy = np.maximum(z, y0 - dxy[:, 1]) + np.maximum(z, dxy[:, 1] - y1)
    e = (dx ** 2 + dy ** 2) / (2 * sigma) ** 2 / (gt_ann["area"] + np.spacing(1)) / 2
    return float(np.exp(-e).sum() / e.shape[0])


def class_sigmas(gt: dict) -> dict | None:
    """Per-class OKS sigma vectors from the categories' "oks_sigmas", or None (uniform 0.05)."""
    s = {c["id"]: np.asarray(c["oks_sigmas"], dtype=float) for c in gt["categories"] if c.get("oks_sigmas")}
    return s or None


def prepare(preds, gt, img_index, sigmas: dict | None = None):
    """-> per class: dict(img (n,), tp (T, n), ig (T, n), gt_img (m,)) in COCO's order.
    sigmas: class -> per-landmark sigma vector (default: uniform 0.05)."""
    gts = {(a["image_id"], a["category_id"]): a for a in gt["annotations"]}
    by = defaultdict(list)
    for p in preds:
        by[(p["image_id"], p["category_id"])].append(p)
    K = {c["id"] for c in gt["categories"] if c.get("keypoints")}
    out = {}
    for c in sorted(K):
        sg = sigmas[c] if sigmas and c in sigmas else 0.05
        # no detections for a class: loadRes copies all categories, so COCOeval still
        # evaluates it and scores 0 AP (not -1) -> keep it with an empty detection list
        imgs = sorted({k[0] for k in by if k[1] == c} | {k[0] for k in gts if k[1] == c})
        img_l, sc_l, tp_l, ig_l, gt_img = [], [], [], [], []
        for i in imgs:
            g = gts.get((i, c))
            g_ign = g is not None and (g.get("num_keypoints", 0) == 0 or g.get("iscrowd", 0))
            if g is not None and not g_ign:
                gt_img.append(img_index[i])
            ds = by.get((i, c), [])
            order = np.argsort([-d["score"] for d in ds], kind="mergesort")[:20]
            ds = [ds[j] for j in order]
            if g is not None:
                gk = np.asarray(g["keypoints"], dtype=float).reshape(-1, 3)
            matched = np.zeros(len(THRS), bool)
            for d in ds:
                tp = np.zeros(len(THRS), bool)
                ig = np.zeros(len(THRS), bool)
                if g is not None:
                    dxy = np.asarray(d["keypoints"], dtype=float).reshape(-1, 3)[:, :2]
                    o = _oks_ignored(dxy, g, sg) if not (gk[:, 2] > 0).any() else _oks(dxy, gk, g["area"], sg)
                    hit = (~matched) & (o >= THRS)
                    matched |= hit
                    tp, ig = (hit & (not g_ign)), (hit & bool(g_ign))
                img_l.append(img_index[i]); sc_l.append(d["score"]); tp_l.append(tp); ig_l.append(ig)
        o = np.argsort(-np.asarray(sc_l), kind="mergesort")
        T = len(THRS)
        out[c] = dict(img=np.asarray(img_l, dtype=int)[o],
                      tp=np.asarray(tp_l, dtype=bool).reshape(-1, T).T[:, o],
                      ig=np.asarray(ig_l, dtype=bool).reshape(-1, T).T[:, o],
                      gt_img=np.asarray(gt_img, dtype=int))
    return out


def ap_weighted(prep, W):
    """W (B, n_images) image multiplicities -> macro OKS-AP (B,), in %."""
    B = W.shape[0]
    per = []
    for c, d in prep.items():
        npig = W[:, d["gt_img"]].sum(1)                               # (B,)
        if len(d["gt_img"]) == 0:
            continue
        # A resampled image drawn w times contributes w consecutive copies of its detections
        # (equal scores, stable sort keeps them adjacent), exactly as COCOeval would see it.
        wd = W[:, d["img"]].astype(int)                               # (B, n)
        aps = np.zeros(B)
        for b in range(B):
            rep = np.repeat(np.arange(len(d["img"])), wd[b])
            if npig[b] == 0:
                aps[b] = np.nan
                continue
            for t in range(len(THRS)):
                f = d["tp"][t][rep][~d["ig"][t][rep]]
                tp = np.cumsum(f)
                fp = np.cumsum(~f)
                rc = tp / npig[b]
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                ix = np.searchsorted(rc, REC, side="left")
                aps[b] += pr[ix[ix < len(rc)]].sum() / len(REC)
        per.append(aps / len(THRS))
    return 100.0 * np.nanmean(per, 0)


def pck_parts(preds, gt, img_index, alpha=0.1):
    """-> per class (img (m,), hits (m,), n (m,)) for PCK reweighting."""
    best = {}
    for p in preds:
        k = (p["image_id"], p["category_id"])
        if k not in best or p["score"] > best[k]["score"]:
            best[k] = p
    parts = defaultdict(lambda: ([], [], []))
    for a in gt["annotations"]:
        if a.get("num_keypoints", 0) == 0:
            continue
        g = np.asarray(a["keypoints"], dtype=float).reshape(-1, 3)
        v = g[:, 2] > 0
        p = best.get((a["image_id"], a["category_id"]))
        if p is None:
            h = 0
        else:
            d = np.hypot(*(np.asarray(p["keypoints"]).reshape(-1, 3)[:, :2] - g[:, :2]).T)
            h = int((d[v] <= alpha * float(np.hypot(a["bbox"][2], a["bbox"][3]))).sum())
        L = parts[a["category_id"]]
        L[0].append(img_index[a["image_id"]]); L[1].append(h); L[2].append(int(v.sum()))
    return {c: tuple(np.asarray(x) for x in L) for c, L in parts.items()}


def pck_weighted(parts, W, skip_absent: bool = False):
    per = []
    for i, h, n in parts.values():
        den = (W[:, i] * n).sum(1)
        r = (W[:, i] * h).sum(1) / np.maximum(den, 1e-12)
        per.append(np.where(den > 0, r, np.nan) if skip_absent else r)
    return 100.0 * (np.nanmean(per, 0) if skip_absent else np.mean(per, 0))


def mre_parts(preds, gt, img_index):
    """-> (img (m,), summed error (m,), n landmarks with a prediction (m,), n visible (m,)) per annotation;
    error in pixels between the top-scoring prediction and the GT, over visible landmarks."""
    best = {}
    for p in preds:
        k = (p["image_id"], p["category_id"])
        if k not in best or p["score"] > best[k]["score"]:
            best[k] = p
    img, dist, n, nv = [], [], [], []
    for a in gt["annotations"]:
        if a.get("num_keypoints", 0) == 0:
            continue
        g = np.asarray(a["keypoints"], dtype=float).reshape(-1, 3)
        v = g[:, 2] > 0
        p = best.get((a["image_id"], a["category_id"]))
        img.append(img_index[a["image_id"]]); nv.append(int(v.sum()))
        if p is None:
            dist.append(0.0); n.append(0)
            continue
        d = np.hypot(*(np.asarray(p["keypoints"]).reshape(-1, 3)[:, :2] - g[:, :2]).T)[v]
        dist.append(float(d.sum())); n.append(int(v.sum()))
    return tuple(np.asarray(x) for x in (img, dist, n, nv))


def mre_weighted(parts, W):
    i, d, n, _ = parts
    return (W[:, i] * d).sum(1) / np.maximum((W[:, i] * n).sum(1), 1e-12)

