"""
Write derived keypoints into the PAXRay++ COCO JSONs (expands existing
annotations in place into new *_kp.json files).

Per COCO, keypoints are defined per CATEGORY. Frontal and lateral use different
keypoint sets, but the two views share one category list in each JSON, so each
category gets the UNION of its frontal+lateral keypoints (a fixed vector); an
annotation fills only the kpts valid for its image's view, the rest get v=0.

For each annotation we set:
    keypoints     : [x1,y1,v1, ...]  (len 3*K, K = merged kpt count for its category)
    num_keypoints : count of v>0
Each category gets: keypoints (names), skeleton (1-indexed pairs),
keypoint_groups (shared-landmark id per kpt, "" if none; extra metadata).
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
import numpy as np
from pycocotools import mask as M

from . import schema as KS
from . import extract as E

NWORKERS = os.cpu_count() or 1
IMAGE_TASKS = []          # set per-file before Pool (workers inherit via fork)

# ---- visibility (v=1) policy --------------------------------------------- #
# A present keypoint defaults to v=2 (visible). It is downgraded to v=1
# (present-but-not-truly-visible / inferred) when:
#   (a) it belongs to a lung/lobe/zone silhouette class AND falls behind a
#       dense overlying structure (heart u mediastinum u diaphragm) -> the
#       border there is not a visible edge (retrocardiac / subdiaphragmatic); or
#   (b) it belongs to an atlas-projected abdominal organ that is essentially
#       invisible on a CXR.
APPLY_OCCLUSION = True
# retrocardiac = behind the HEART silhouette (the only place a lung soft-tissue
# border is genuinely replaced/hidden). NOT the diaphragm (that wrongly hides
# the visible costophrenic recess) nor the peripheral mediastinum.
OCCLUDER_CLASSES = {"heart"}
ATLAS_INVISIBLE = {"kidney left", "kidney right", "pancreas", "small bowel", "duodenum"}
OCCLUDABLE_SUBSTR = ("lung", "lobe", "zone")
# landmarks that stay VISIBLE even over the heart shadow: the mediastinal
# stripes (air-soft-tissue interfaces seen through the heart), the hila
# (vascular), fissures, and the lateral/apical silhouette + CTR width point.
VISIBLE_NAMES = ("apex", "lateral_margin", "costophrenic", "hilum", "ctr_thorax",
                 "para_aortic", "paraspinal", "paratracheal", "fissure")


# ---- merged (frontal ∪ lateral) keypoint definition per category ---------- #
def _merged(cname):
    f = KS.SCHEMA_FRONTAL[cname]
    l = KS.SCHEMA_LATERAL[cname]
    names = f.names() + [n for n in l.names() if n not in set(f.names())]
    idx = {n: i + 1 for i, n in enumerate(names)}
    edges = set()
    for sch in (f, l):
        for a, b in sch.skeleton:
            if a in idx and b in idx:
                edges.add((idx[a], idx[b]))
    grp = {}
    for sch in (f, l):
        for k in sch.keypoints:
            if k.group and k.name not in grp:
                grp[k.name] = k.group
    groups = [grp.get(n, "") for n in names]
    return names, sorted([list(e) for e in edges]), groups


MERGED = {name: _merged(name) for _, name in KS.CATEGORIES}


def _decode(seg):
    c = dict(seg)
    if isinstance(c["counts"], str):
        c["counts"] = c["counts"].encode()
    return M.decode(c).astype(bool)


def _worker(i):
    iid, W, H, view, anns = IMAGE_TASKS[i]
    schema = KS.SCHEMA_FRONTAL if view == "frontal" else KS.SCHEMA_LATERAL
    masks = {}
    for _, nm, seg in anns:
        mk = _decode(seg)
        if mk.any():
            masks[nm] = E.clean_mask(mk)
    ctx = E.make_ctx(masks, W, H)
    occ = None
    if APPLY_OCCLUSION:
        for oc in OCCLUDER_CLASSES:
            if oc in masks:
                occ = masks[oc].copy() if occ is None else (occ | masks[oc])
    out = []
    for ann_id, nm, _ in anns:
        names = MERGED[nm][0]
        kp = E.extract_for_class(nm, masks[nm], schema, ctx, masks) if nm in masks else {}
        is_atlas = nm in ATLAS_INVISIBLE
        occludable = any(s in nm for s in OCCLUDABLE_SUBSTR)
        flat, nv = [], 0
        for n in names:
            p = kp.get(n)
            if p and p[2] > 0 and 0 <= p[0] < W and 0 <= p[1] < H:
                v = 2
                if is_atlas:
                    v = 1
                elif (occludable and occ is not None and occ[int(p[1]), int(p[0])]
                      and not any(w in n for w in VISIBLE_NAMES)):
                    v = 1                       # retrocardiac (behind heart), not a visible edge
                flat += [int(p[0]), int(p[1]), v]
                nv += 1
            else:
                flat += [0, 0, 0]
        out.append((ann_id, flat, nv))
    return out


# ---- side relabelling (frontal only) -------------------------------------- #
# PAXRay++ names these pairs inconsistently: hemidiaphragm and less-obstructed
# lung are named in image coordinates (swapped in 100% of frontals), clavicles in
# ~59%. Patient-left is image-right in every frontal, so the side is reassigned by
# position (bbox centre vs. partner, or vs. the spine when the partner is absent).
# Laterals carry no positional side cue and are left untouched.
RELABEL_PAIRS = [("clavicle left", "clavicle right"),
                 ("left hemidiaphragm", "right hemidiaphragm"),
                 ("less obstructed left lung", "less obstructed right lung")]


def _cx(a):
    x, _, w, _ = a["bbox"]
    return x + w / 2


def relabel_sides(coco, imgs, by_img):
    cid = {c["name"]: c["id"] for c in coco["categories"]}
    cname = {v: k for k, v in cid.items()}
    stats = {l: [0, 0] for l, _ in RELABEL_PAIRS}       # [swapped, seen]
    for iid, anns in by_img.items():
        im = imgs[iid]
        if "frontal" not in im["file_name"]:
            continue
        verts = [_cx(a) for a in anns if cname[a["category_id"]].startswith("vertebrae T")]
        mid = float(np.median(verts)) if verts else im.get("width", 512) / 2
        for l, r in RELABEL_PAIRS:
            al = [a for a in anns if a["category_id"] == cid[l]]
            ar = [a for a in anns if a["category_id"] == cid[r]]
            if len(al) > 1 or len(ar) > 1 or not (al or ar):
                continue
            stats[l][1] += 1
            if al and ar:
                wrong = _cx(al[0]) < _cx(ar[0])
            else:
                a = (al or ar)[0]
                wrong = (_cx(a) > mid) != bool(al)
            if wrong:
                stats[l][0] += 1
                for a in al:
                    a["category_id"] = cid[r]
                for a in ar:
                    a["category_id"] = cid[l]
    for l, (s, n) in stats.items():
        print(f"  relabel {l!r}: swapped {s}/{n} frontal images", flush=True)
    coco.setdefault("info", {})["side_relabel"] = {
        "pairs": RELABEL_PAIRS, "rule": "frontal only; patient-left = image-right (bbox centre)",
        "swapped": {l: s for l, (s, _) in stats.items()}, "seen": {l: n for l, (_, n) in stats.items()}}


def process(in_path, out_path):
    global IMAGE_TASKS
    t0 = time.time()
    print(f"[{os.path.basename(in_path)}] loading ...", flush=True)
    coco = json.load(open(in_path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    print(f"  {len(imgs)} images, {len(coco['annotations'])} annotations "
          f"(loaded {time.time()-t0:.0f}s)", flush=True)
    relabel_sides(coco, imgs, by_img)

    for c in coco["categories"]:
        names, skel, groups = MERGED[c["name"]]
        c["keypoints"], c["skeleton"], c["keypoint_groups"] = names, skel, groups

    IMAGE_TASKS = []
    for iid, anns in by_img.items():
        im = imgs[iid]
        H, W = im.get("height", 512), im.get("width", 512)
        view = "frontal" if "frontal" in im["file_name"] else "lateral"
        IMAGE_TASKS.append((iid, W, H, view,
                            [(a["id"], cats[a["category_id"]], a["segmentation"]) for a in anns]))

    results = {}
    n = len(IMAGE_TASKS)
    # workers read the module-global IMAGE_TASKS, so they must inherit it via fork
    with mp.get_context("fork").Pool(NWORKERS) as pool:
        done = 0
        for res in pool.imap_unordered(_worker, range(n), chunksize=8):
            for ann_id, flat, nv in res:
                results[ann_id] = (flat, nv)
            done += 1
            if done % 1000 == 0:
                el = time.time() - t0
                print(f"  {done}/{n} imgs ({el:.0f}s, eta {el/done*(n-done)/60:.0f}m)", flush=True)

    for a in coco["annotations"]:
        flat, nv = results[a["id"]]
        a["keypoints"] = flat
        a["num_keypoints"] = nv

    print(f"  writing {out_path} ...", flush=True)
    json.dump(coco, open(out_path, "w"))
    print(f"  DONE {os.path.basename(out_path)} in {(time.time()-t0)/60:.1f}m", flush=True)


SPLITS = ("test", "val", "train")


def main(argv=None):
    """Write <split><suffix>.json (default *_kp_v1.json) from the PAXRay++ COCO jsons
    <paxray-root>/annotations/<split>.json."""
    global NWORKERS
    ap = argparse.ArgumentParser(
        prog="cxrlm-make-labels",
        description="Derive the silver landmark labels (kp_v1) from the PAXRay++ anatomy masks.")
    ap.add_argument("--paxray-root", required=True,
                    help="PAXRay++ root containing annotations/{train,val,test}.json")
    ap.add_argument("--out-dir", default=None, help="output directory (default: <paxray-root>/annotations)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS),
                    help="splits to process (default: test val train)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--suffix", default=os.environ.get("KP_SUFFIX", "_kp_v1"),
                    help="output file suffix (default: $KP_SUFFIX or _kp_v1)")
    args = ap.parse_args(argv)
    NWORKERS = max(1, args.workers)
    out_dir = args.out_dir or os.path.join(args.paxray_root, "annotations")
    os.makedirs(out_dir, exist_ok=True)
    # v1 = Delphi round-1 vocabulary + revised rules (schema.FINAL_VOCAB_PATH)
    print(f"vocabulary: {KS.FINAL_VOCAB_PATH}", flush=True)
    for split in [s for s in SPLITS if s in args.splits]:
        process(os.path.join(args.paxray_root, "annotations", f"{split}.json"),
                os.path.join(out_dir, f"{split}{args.suffix}.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
