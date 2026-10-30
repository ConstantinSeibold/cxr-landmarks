"""
Mask -> keypoint extractor for PAXRay++ (see schema.py).

Implements one primitive per `method` token, computed deterministically from
the binary RLE masks. Relational landmarks use:
  * directed single-mask extremes (axis = spine midline; lateral = away from it)
  * `nearest`         (point on this mask closest to a partner mask)
  * `overlap_centroid`(centroid of this mask & a partner mask)
because anatomically adjacent masks do NOT touch in this dataset.

`__main__` runs an overlay validation on a few real frontal images for the
costophrenic / cardiophrenic decomposed groups.
"""

import os
import json
import math
import collections
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from pycocotools import mask as M

from . import schema as KS

# PAXRay++ root / image dir: only used by the overlay demo in __main__
# (set from the command line there); the extraction rules never read them.
DATA = os.environ.get("PAXRAY_ROOT", "")
IMGDIR = os.path.join(DATA, "images_patlas")


# ---------------- mask helpers ---------------- #
def decode_rle(seg):
    c = dict(seg)
    if isinstance(c["counts"], str):
        c["counts"] = c["counts"].encode()
    return M.decode(c).astype(bool)


def clean_mask(mask, frac=0.15):
    """Drop tiny speckle components but KEEP all substantial ones, so bilateral
    combined masks (clavicles, scapulae, both lungs) retain both sides. (Using
    only the largest CC would discard one side.)"""
    lbl, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= frac * sizes.max()]
    return np.isin(lbl, keep)


# ---------------- directed extreme ---------------- #
_DIRS = {  # name -> (vy, hx)   vy:+1 lower/-1 upper   hx:+1 lateral/-1 medial
    "top": (-1, 0), "bottom": (1, 0),
    "lateral": (0, 1), "medial": (0, -1),
    "infero_lateral": (1, 1), "infero_medial": (1, -1),
    "supero_lateral": (-1, 1), "supero_medial": (-1, -1),
    # lateral-view aliases (anterior/posterior handled via ap_sign in ctx)
    "anterior": (0, 1), "posterior": (0, -1),
    "antero_superior": (-1, 1), "antero_inferior": (1, 1),
    "postero_superior": (-1, -1), "postero_inferior": (1, -1),
}
_LATERAL_DIRS = {"anterior", "posterior", "antero_superior", "antero_inferior",
                 "postero_superior", "postero_inferior"}


def directed_extreme(mask, direction, ctx, side, from_name):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    vy, hx = _DIRS[direction]
    if direction in _LATERAL_DIRS:                       # lateral-view AP axis
        pos = ctx["ant_sign"] * (xs - ctx["mid_ap"])    # +axis = anterior (sternum)
    else:
        mid = ctx["mid_x"]
        if side in ("left", "right") and from_name:
            # side came from the kpt NAME suffix -> structure follows the
            # anatomical convention (heart, combined lung): map via left_sign
            # over the WHOLE mask (the two borders stay opposite). Half-restrict
            # only for infero/supero CORNERS on a genuinely spanning mask.
            # name-sided kpts only occur on bilateral "both" classes -> always
            # split at the midline so _left/_right can't collapse to one point.
            s = ctx["left_sign"] if side == "left" else -ctx["left_sign"]
            keep = (s * (xs - mid)) >= 0                 # restrict to the named half
            if keep.sum() >= 10:
                xs, ys = xs[keep], ys[keep]
        else:
            # single-side mask (lung, hemidiaphragm): derive lateral direction
            # from the mask's OWN position (the L/R label can be flipped, e.g.
            # the diaphragm), so never trust the name here.
            s = 1 if xs.mean() >= mid else -1
        pos = s * (xs - mid)
    score = vy * ys.astype(float) + hx * pos.astype(float)
    j = int(np.argmax(score))
    return (int(xs[j]), int(ys[j]), 2)


def superior_boundary(mask):
    """Topmost mask pixel per column -> (xs, ys) of the upper ridge."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    x0 = xs.min()
    top = np.full(xs.max() - x0 + 1, 1 << 20)
    np.minimum.at(top, xs - x0, ys)
    valid = top < (1 << 20)
    bx = np.arange(x0, xs.max() + 1)[valid]
    by = top[valid]
    return bx, by


def ridge_extreme(mask, which, ctx, side=None):
    """Point on the SUPERIOR ridge of a filled region (the diaphragm = the dome
    curve atop the abdominal mask). The mask itself is abdomen, so we only ever
    read its upper boundary. dome = highest ridge point; lateral/medial ends and
    anterior/posterior ends (lateral view) and q_lateral/q_medial samples for
    capturing dome curvature (flattening).

    For a midline-spanning mask (combined `diaphragm`) with an explicit side,
    restrict to that half first (via the lung-derived left_sign). For a
    single-side mask, lateral direction is auto-derived from the mask's own
    position (the L/R label can be flipped vs lungs)."""
    mid = ctx["mid_x"]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    frac_left = (xs < mid).mean()
    straddle = min(frac_left, 1 - frac_left) > 0.30
    if side in ("left", "right") and straddle:
        ssign = ctx["left_sign"] if side == "left" else -ctx["left_sign"]
        keep = (ssign * (xs - mid)) >= 0
        if keep.sum() > 50:
            m2 = np.zeros_like(mask)
            m2[ys[keep], xs[keep]] = True
            mask = m2
        lat_sign = ssign
    else:
        lat_sign = None                    # auto, decided from the boundary
    bx, by = superior_boundary(mask)
    if bx is None:
        return None
    if which == "dome":
        j = int(np.argmin(by))
    elif which in ("anterior", "posterior"):
        ap = ctx["ant_sign"] * (bx - ctx["mid_ap"])
        j = int(np.argmax(ap)) if which == "anterior" else int(np.argmin(ap))
    else:  # lateral / medial / q_lateral / q_medial  (LR axis)
        s = lat_sign if lat_sign is not None else (1 if bx.mean() >= mid else -1)
        lat = s * (bx - mid)
        order = np.argsort(lat)            # medial -> lateral
        if which == "lateral":
            j = int(order[-1])
        elif which == "medial":
            j = int(order[0])
        else:
            # quarter points sampled RELATIVE to the dome's position so the
            # ridge chain (medial -> q_medial -> dome -> q_lateral -> lateral)
            # stays monotonic in lateral order (no self-crossing on viz).
            dome_rank = int(np.where(order == int(np.argmin(by)))[0][0])
            n = len(order)
            if which == "q_medial":
                j = int(order[max(0, dome_rank // 2)])
            else:                          # q_lateral
                j = int(order[min(n - 1, (dome_rank + n - 1) // 2)])
    return (int(bx[j]), int(by[j]), 2)


def _peak_band(profile, frac=0.45, smooth=5):
    """Contiguous run around the profile's peak where value >= frac*peak."""
    p = np.convolve(profile.astype(float), np.ones(smooth) / smooth, "same")
    if p.max() <= 0:
        return 0, len(p) - 1
    peak = int(np.argmax(p))
    thr = frac * p[peak]
    a = peak
    while a > 0 and p[a - 1] >= thr:
        a -= 1
    b = peak
    while b < len(p) - 1 and p[b + 1] >= thr:
        b += 1
    return a, b


def vertebral_body(mask, lateral=False, ctx=None):
    """Isolate the vertebral BODY from the full vertebra mask (which includes
    spinous / transverse / articular processes).

    FRONTAL: body = contiguous tall-and-wide core (row band around peak width
    drops the spinous tail; column band around peak height drops the transverse
    processes).

    LATERAL: the posterior arch/spinous extend POSTERIORLY (not laterally), so
    isolate the body as the ANTERIOR ~55% of the AP extent (toward the sternum).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    if lateral:
        asign = ctx["ant_sign"] if ctx else 1
        ap = asign * xs                                 # larger = anterior
        cut = ap.max() - 0.55 * (ap.max() - ap.min())
        keep = ap >= cut
        body = np.zeros_like(mask)
        body[ys[keep], xs[keep]] = True
        return body if body.any() else mask
    rw = np.bincount(ys, minlength=mask.shape[0])      # width per row
    ch = np.bincount(xs, minlength=mask.shape[1])       # height per column
    ytop, ybot = _peak_band(rw)
    xl, xr = _peak_band(ch)
    body = mask.copy()
    body[:ytop] = False
    body[ybot + 1:] = False
    body[:, :xl] = False
    body[:, xr + 1:] = False
    return body if body.any() else mask


def rect_corner(mask, token, ctx):
    """Vertebral-body endplate corner. Isolates the body first (vertebral_body),
    then takes the corner via top/bottom edge band + horizontal extreme.
    token letters: S/I (sup/inf endplate) + L/R (anatomical) or A/P (ant/post)."""
    lateral = any(c in token for c in "AP")
    body = vertebral_body(mask, lateral, ctx)
    ys, xs = np.where(body)
    if len(xs) == 0:
        return None
    ymin, ymax = ys.min(), ys.max()
    h = max(1, ymax - ymin)
    band = ys <= ymin + 0.25 * h if "S" in token else ys >= ymax - 0.25 * h
    bx, by = xs[band], ys[band]
    hor = next(c for c in token if c in "LRAP")
    if hor in "LR":
        val = ctx["left_sign"] * (bx - ctx["mid_x"])           # larger = anatomical left
        k = int(np.argmax(val)) if hor == "L" else int(np.argmin(val))
    else:
        val = ctx["ant_sign"] * (bx - ctx["mid_ap"])           # larger = anterior
        k = int(np.argmax(val)) if hor == "A" else int(np.argmin(val))
    return (int(bx[k]), int(by[k]), 2)


def skeleton_point(mask, which):
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    ys, xs = np.where(sk)
    if len(xs) == 0:
        ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if which == "top":
        j = int(np.argmin(ys))
    elif which == "bottom":
        j = int(np.argmax(ys))
    else:                                  # mid: skeleton pixel nearest median y
        j = int(np.argmin(np.abs(ys - np.median(ys))))
    return (int(xs[j]), int(ys[j]), 2)


def _skeleton_path(mask):
    """Ordered medial-axis polyline (the longest geodesic path through the
    skeleton's largest component). Returns (M,2) array of (y,x), oriented
    top->bottom, or None."""
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    if sk.sum() < 2:
        return None
    lbl, n = ndi.label(sk, structure=np.ones((3, 3), int))   # 8-conn (skeletons are diagonal)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        sk = lbl == (1 + int(np.argmax(sizes)))
    pts = np.column_stack(np.where(sk))                 # (y,x)
    if len(pts) < 2:
        return None
    adj = collections.defaultdict(list)
    for a, b in cKDTree(pts).query_pairs(1.5):
        adj[a].append(b); adj[b].append(a)

    def bfs(src):
        dist = {src: 0}; par = {src: -1}; q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; par[v] = u; q.append(v)
        far = max(dist, key=dist.get)
        return far, par
    a, _ = bfs(0)
    b, par = bfs(a)
    path = []
    cur = b
    while cur != -1:
        path.append(cur); cur = par[cur]
    path = pts[path[::-1]]
    if path[0, 0] > path[-1, 0]:                         # orient top -> bottom
        path = path[::-1]
    return path


def _arclen(path):
    return np.r_[0, np.cumsum(np.hypot(np.diff(path[:, 1]), np.diff(path[:, 0])))]


def centerline_point(mask, frac):
    """Point at arc-length `frac` (0..1) along the structure's centerline."""
    path = _skeleton_path(mask)
    if path is None:
        return centroid(mask)
    d = _arclen(path)
    t = frac * d[-1]
    i = int(np.clip(np.searchsorted(d, t), 1, len(path) - 1))
    seg = d[i] - d[i - 1]
    w = 0.0 if seg == 0 else (t - d[i - 1]) / seg
    y = path[i - 1, 0] + w * (path[i, 0] - path[i - 1, 0])
    x = path[i - 1, 1] + w * (path[i, 1] - path[i - 1, 1])
    return (int(x), int(y), 2)


def centerline_curvature(mask, lo=0.12, hi=0.55):
    """Max-curvature point along the centerline within [lo,hi] arc fraction.
    Used for the sternal angle (manubriosternal bend, upper-mid sternum) --
    far more robust than whole-contour curvature, which finds the bottom tip."""
    path = _skeleton_path(mask)
    if path is None or len(path) < 7:
        return centerline_point(mask, 0.3)
    d = _arclen(path); fr = d / d[-1]
    w = max(2, len(path) // 8)
    best, bj = -1.0, None
    for j in range(w, len(path) - w):
        if not (lo <= fr[j] <= hi):
            continue
        v1 = path[j] - path[j - w]; v2 = path[j + w] - path[j]
        a1 = math.atan2(v1[0], v1[1]); a2 = math.atan2(v2[0], v2[1])
        turn = abs(math.atan2(math.sin(a2 - a1), math.cos(a2 - a1)))
        if turn > best:
            best, bj = turn, j
    if bj is None:
        return centerline_point(mask, 0.3)
    return (int(path[bj, 1]), int(path[bj, 0]), 2)


def curvature_max(mask, ctx):
    """Sharpest-curvature point on the contour (e.g. sternal angle). Approximate:
    turning angle over a fixed arc window on the smoothed largest contour."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    n = len(c)
    if n < 30:
        return centroid(mask)
    w = max(3, n // 25)
    v1 = c - np.roll(c, w, axis=0)
    v2 = np.roll(c, -w, axis=0) - c
    a1 = np.arctan2(v1[:, 1], v1[:, 0])
    a2 = np.arctan2(v2[:, 1], v2[:, 0])
    turn = np.abs(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1)))
    j = int(np.argmax(turn))
    return (int(c[j, 0]), int(c[j, 1]), 2)


def centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()), 2)


def restrict_half(mask, side, ctx):
    """Restrict a midline-spanning mask to its left/right half (via left_sign)
    so per-side kpts on a combined bilateral class (clavicles/scapulas/breast)
    don't collapse to one point. No-op for single-side masks."""
    if side not in ("left", "right"):
        return mask
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    mid = ctx["mid_x"]
    ssign = ctx["left_sign"] if side == "left" else -ctx["left_sign"]
    keep = (ssign * (xs - mid)) >= 0
    if keep.sum() < 30:                                  # half essentially empty -> keep full
        return mask
    m2 = np.zeros_like(mask)
    m2[ys[keep], xs[keep]] = True
    return m2


def overlap_centroid(mask, via_mask):
    if via_mask is None:
        return None
    inter = mask & via_mask
    return centroid(inter) if inter.any() else None


def nearest_point(mask, via_mask):
    if via_mask is None or not via_mask.any() or not mask.any():
        return None
    dt = ndi.distance_transform_edt(~via_mask)
    ys, xs = np.where(mask)
    j = int(np.argmin(dt[ys, xs]))
    return (int(xs[j]), int(ys[j]), 2)


def contact_point(mask, other, where, ctx, d=4):
    """Point on the INTERFACE between two masks (the dilated-overlap zone), used
    for fissures (lobe<->lobe) and mediastinal stripes (structure<->lung).
    `where` selects mid / top / bottom / lateral / medial of the interface."""
    if other is None or not other.any() or not mask.any():
        return None
    band = ndi.binary_dilation(mask, iterations=d) & ndi.binary_dilation(other, iterations=d)
    ys, xs = np.where(band)
    if len(xs) < 5:
        return None
    if where == "mid":
        return (int(xs.mean()), int(ys.mean()), 2)
    if where == "top":
        j = int(np.argmin(ys))
    elif where == "bottom":
        j = int(np.argmax(ys))
    else:                                       # lateral / medial w.r.t. midline
        mid = ctx["mid_x"]
        s = 1 if xs.mean() >= mid else -1
        lat = s * (xs - mid)
        j = int(np.argmax(lat)) if where == "lateral" else int(np.argmin(lat))
    return (int(xs[j]), int(ys[j]), 2)


def rib_costal(mask, via_masks, ctx):
    """Costal limb of the costophrenic angle = infero-lateral-most point of the
    rib cage on that side, from the UNION of the given rib masks (user's idea:
    use the rib outlines / outermost extent rather than one specific rib)."""
    u = None
    for vm in via_masks:
        if vm is not None and vm.any():
            u = vm.copy() if u is None else (u | vm)
    if u is None or not u.any():
        return None
    return directed_extreme(u, "lateral", ctx, None, False)   # lateral chest wall


def bronchus_point(mask, side, ctx):
    """Main-bronchus point for the carinal angle = centroid of the DISTAL HALF
    of the bronchial limb (below the carina, named anatomical side). The
    carina->point vector follows the established bronchial axis -> realistic
    angle (~59 deg median). The distal infero-lateral tips splay wider and
    overestimate (~90); the full-limb centroid includes the near-vertical
    proximal part and underestimates (~52)."""
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return None
    ay = ys.min()
    ax = xs[ys == ay].mean()                     # carina apex
    below = ys > ay + 2
    bx, by = xs[below], ys[below]
    if len(bx) < 5:
        return centroid(mask)
    s = ctx["left_sign"] if side == "left" else -ctx["left_sign"]
    sel = (s * (bx - ax)) >= 0                    # limb on the named anatomical side
    if sel.sum() < 5:
        return None
    sx, sy = bx[sel], by[sel]
    d = np.hypot(sx - ax, sy - ay)               # distance from carina
    keep = d >= np.quantile(d, 0.6)              # distal portion -> bronchial axis
    if keep.sum() < 3:
        keep = np.ones(len(sx), bool)
    return (int(sx[keep].mean()), int(sy[keep].mean()), 2)


def ctr_thorax(mask, dome_mask, ctx):
    """Inner thoracic margin at the (right-)hemidiaphragm DOME level -> the
    standard CTR denominator endpoint. Lateral-most lung point at the dome row."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    if dome_mask is not None and dome_mask.any():
        dome = ridge_extreme(dome_mask, "dome", ctx)
        if dome is not None:
            band = np.abs(ys - dome[1]) <= 8
            if band.sum() >= 5:
                ys, xs = ys[band], xs[band]
    mid = ctx["mid_x"]
    s = 1 if xs.mean() >= mid else -1
    j = int(np.argmax(s * (xs - mid)))
    return (int(xs[j]), int(ys[j]), 2)



# ---------------- v1 rules (Delphi round 1: resident + attending revisions) ---------------- #
def _anat_left(xs, ctx):
    """Signed distance from the midline, positive towards the patient's LEFT."""
    return ctx["left_sign"] * (np.asarray(xs, float) - ctx["mid_x"])


def _union(masks, names):
    u = None
    for n in names:
        m = masks.get(n)
        if m is not None and m.any():
            u = m.copy() if u is None else (u | m)
    return u


def hemi_on(side, masks, ctx):
    """Hemidiaphragm mask lying on the patient's `side` (the L/R labels of the
    hemidiaphragm classes are flipped in part of the data, so decide by position)."""
    s = 1 if side == "left" else -1
    for nm in ("left hemidiaphragm", "right hemidiaphragm"):
        m = masks.get(nm)
        if m is not None and m.any() and s * _anat_left(np.where(m)[1], ctx).mean() > 0:
            return m
    return None


def _diaphragm_top(masks):
    """Per-column top of the (sub-)diaphragmatic region: dict x -> y."""
    u = _union(masks, ("left hemidiaphragm", "right hemidiaphragm", "diaphragm"))
    if u is None:
        return {}
    bx, by = superior_boundary(u)
    return dict(zip(bx.tolist(), by.tolist()))


def _row_profile(mask, lat):
    """Per row, the pixel maximising `lat(xs)` (outer silhouette edge). -> dict y -> (x, value)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {}
    v = lat(xs)
    prof = {}
    for y, x, val in zip(ys, xs, v):
        if y not in prof or val > prof[y][1]:
            prof[y] = (int(x), float(val))
    return prof


def _chord_knee(prof, y0, y1):
    """Deepest point of a concavity: the profile row (in [y0, y1]) lying furthest
    INSIDE the straight chord between the profile points at y0 and y1."""
    rows = [y for y in sorted(prof) if min(y0, y1) <= y <= max(y0, y1)]
    if len(rows) < 3:
        return None
    ya, yb = rows[0], rows[-1]
    va, vb = prof[ya][1], prof[yb][1]
    best, by = -1e9, None
    for y in rows:
        chord = va + (vb - va) * (y - ya) / max(1, yb - ya)
        depth = chord - prof[y][1]
        if depth > best:
            best, by = depth, y
    return (prof[by][0], int(by), 2)


def _pt_extreme_side(mask, ctx, side, vy=0, hx=1):
    """Directed extreme on the named ANATOMICAL side (lateral = away from midline)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    s = 1 if side == "left" else -1
    lat = s * _anat_left(xs, ctx)
    keep = lat >= 0
    if keep.sum() >= 10:
        xs, ys, lat = xs[keep], ys[keep], lat[keep]
    j = int(np.argmax(vy * ys + hx * lat))
    return (int(xs[j]), int(ys[j]), 2)


def visible_corner(lung, direction, ctx, masks, fallback):
    """Costophrenic corner of the VISIBLE lung, i.e. where the dome contour meets the
    chest wall: the lung clipped at the diaphragm's superior contour, so the recess
    projected behind the diaphragm is ignored."""
    top = _diaphragm_top(masks)
    if lung is None or not lung.any() or not top:
        return fallback()
    ys, xs = np.where(lung)
    keep = np.array([y < top.get(x, 1 << 20) for x, y in zip(xs, ys)])
    if keep.sum() < 20:
        return fallback()
    vm = np.zeros_like(lung)
    vm[ys[keep], xs[keep]] = True
    return directed_extreme(vm, direction, ctx, None, False)


def ridge_under_lung(mask, lung, which, ctx):
    """Lateral-view costophrenic sulcus: anterior / posterior end of the diaphragm's
    superior contour, restricted to columns where the visible lung lies directly above
    it (within 1 cm), i.e. under the visible costophrenic angle."""
    bx, by = superior_boundary(mask)
    if bx is None:
        return None
    if lung is not None and lung.any():
        d = ndi.distance_transform_edt(~lung)[by, bx]
        sel = d <= ctx["px_per_cm"]
        if sel.sum() >= 5:
            bx, by = bx[sel], by[sel]
    ap = ctx["ant_sign"] * (bx - ctx["mid_ap"])
    j = int(np.argmax(ap)) if which == "anterior" else int(np.argmin(ap))
    return (int(bx[j]), int(by[j]), 2)


def apex_lateral(mask, ctx, masks):
    """Lateral-view cardiac apex: antero-inferior tip of the heart above the diaphragm."""
    top = _diaphragm_top(masks)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if top:
        keep = np.array([y < top.get(x, 1 << 20) for x, y in zip(xs, ys)])
        if keep.sum() >= 20:
            xs, ys = xs[keep], ys[keep]
    m2 = np.zeros_like(mask)
    m2[ys, xs] = True
    return directed_extreme(m2, "antero_inferior", ctx, None, False)


# ---- angle limbs: assistive points that make an angle measurable (vertex + 2 limbs) ----
LIMB_CM = 2.0


def _clip_above_diaphragm(mask, masks):
    """Part of `mask` above the diaphragm's superior contour (the visible lung / heart)."""
    top = _diaphragm_top(masks)
    if mask is None or not mask.any() or not top:
        return mask
    ys, xs = np.where(mask)
    keep = np.array([y < top.get(x, 1 << 20) for x, y in zip(xs, ys)])
    if keep.sum() < 20:
        return mask
    m2 = np.zeros_like(mask)
    m2[ys[keep], xs[keep]] = True
    return m2


def _contour_walk(region, vertex, d):
    """The two points at arc length d from the contour point nearest `vertex`, walking both
    ways along the region's outer contour."""
    if region is None or not region.any() or vertex is None:
        return None, None
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    n = len(c)
    if n < 10:
        return None, None
    j = int(np.argmin(np.hypot(c[:, 0] - vertex[0], c[:, 1] - vertex[1])))
    seg = np.hypot(*(np.roll(c, -1, 0) - c).T)
    out = []
    for step in (1, -1):
        acc, k = 0.0, j
        for _ in range(n - 1):
            acc += seg[k] if step == 1 else seg[(k - 1) % n]
            k = (k + step) % n
            if acc >= d:
                break
        out.append((int(c[k, 0]), int(c[k, 1]), 2))
    return out[0], out[1]


def _steeper(p, q, v):
    """Order two limb points as (more vertical limb, more horizontal limb) w.r.t. vertex v."""
    def slope(t):
        return abs(t[1] - v[1]) / (abs(t[0] - v[0]) + 1e-6)
    return (p, q) if slope(p) >= slope(q) else (q, p)


def _ridge_walk(hemi, vertex, d, ctx, outward_sign):
    """Point on the diaphragm's superior contour at arc length d from the ridge point nearest
    the vertex, walking in the anatomical direction `outward_sign` (+1 = towards patient left)."""
    bx, by = superior_boundary(hemi)
    if bx is None or vertex is None:
        return None
    o = np.argsort(bx)
    bx, by = bx[o], by[o]
    j = int(np.argmin(np.hypot(bx - vertex[0], by - vertex[1])))
    step = 1 if outward_sign * ctx["left_sign"] > 0 else -1
    acc, k = 0.0, j
    while 0 <= k + step < len(bx) and acc < d:
        acc += float(np.hypot(bx[k + step] - bx[k], by[k + step] - by[k]))
        k += step
    return (int(bx[k]), int(by[k]), 2)


def visible_lung(side, masks, ctx, fallback=None):
    """The 'less obstructed' (visible) lung of the patient's `side`, chosen by position. Its outline
    is the radiographic one: bottom = dome contour, medial edge = cardiac border. The plain lung
    and heart masks both continue below the diaphragm (CT projection), so angle points taken on
    them sit inside the abdomen."""
    s = 1 if side == "left" else -1
    for nm in ("less obstructed left lung", "less obstructed right lung"):
        m = masks.get(nm)
        if m is not None and m.any() and s * _anat_left(np.where(m)[1], ctx).mean() > 0:
            return m
    return _clip_above_diaphragm(fallback, masks) if fallback is not None else None


def _side_of_mask(mask, ctx):
    return "left" if _anat_left(np.where(mask)[1], ctx).mean() >= 0 else "right"


def posterior_sulcus(vis, ctx, frac=1 / 3):
    """Lateral view: deepest (lowest) point of the visible lung within its posterior third (the
    posterior costophrenic sulcus); a diagonal extreme can stop above it on a curved base."""
    ys, xs = np.where(vis)
    ap = ctx["ant_sign"] * (xs - ctx["mid_ap"])           # larger = anterior
    sel = ap <= ap.min() + frac * (ap.max() - ap.min())
    xs, ys, ap = xs[sel], ys[sel], ap[sel]
    j = int(np.lexsort((ap, -ys))[0])                     # lowest, then most posterior
    return (int(xs[j]), int(ys[j]), 2)


def angle_trio(vis, corner, ctx, lateral=False):
    """Vertex (corner of the visible lung) and the two limb points LIMB_CM along its outline,
    ordered (vertex, steeper limb, flatter limb): steeper = chest wall / cardiac border,
    flatter = diaphragm."""
    if vis is None or not vis.any():
        return None, None, None
    if corner == "postero_inferior":
        v = posterior_sulcus(vis, ctx)
    else:
        v = directed_extreme(vis, corner, ctx, None, False)
    p, q = _contour_walk(vis, v, LIMB_CM * ctx["px_per_cm"])
    if p is None:
        return v, None, None
    steep, flat = _steeper(p, q, v)
    return v, steep, flat


def costophrenic_limb(mask, which, ctx, masks):
    """Frontal costophrenic angle limbs on the visible lung: 'wall' (up the lateral chest wall)
    and 'diaphragm' (along the dome), LIMB_CM from the vertex."""
    vis = visible_lung(_side_of_mask(mask, ctx), masks, ctx, mask)
    _, wall, dia = angle_trio(vis, "infero_lateral", ctx)
    return wall if which == "wall" else dia


def sulcus_limb(mask, sulcus, which, ctx, masks):
    """Lateral-view costophrenic sulcus limbs (anterior / posterior) on the visible lung outline."""
    vis = masks.get("less obstructed lung")
    _, wall, dia = angle_trio(vis, "antero_inferior" if sulcus == "anterior" else "postero_inferior", ctx)
    return wall if which == "wall" else dia


def sulcus_vertex(mask, sulcus, ctx, masks):
    vis = masks.get("less obstructed lung")
    if vis is None or not vis.any():
        return ridge_under_lung(mask, None, sulcus, ctx)
    if sulcus == "posterior":
        return posterior_sulcus(vis, ctx)
    return directed_extreme(vis, "antero_inferior", ctx, None, False)


def cardiophrenic_limb(mask, side, which, ctx, masks):
    """Cardiophrenic angle (vertex / 'heart' / 'diaphragm' limb) on the visible lung of `side`:
    its infero-medial corner, and points LIMB_CM up the cardiac border and along the dome."""
    vis = visible_lung(side, masks, ctx, masks.get(f"{side} lung"))
    v, heart, dia = angle_trio(vis, "infero_medial", ctx)
    return {"vertex": v, "heart": heart, "diaphragm": dia}[which]


def plateau_top(mask, tol_frac=0.04):
    """Middle of the apical plateau: among the pixels within tol of the highest
    row, take the median column (the single top pixel is arbitrary on a flat apex)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    tol = max(2, tol_frac * (ys.max() - ys.min()))
    cand = ys <= ys.min() + tol
    xm = int(np.median(xs[cand]))
    col = ys[xs == xm]
    return (xm, int(col.min()) if len(col) else int(ys.min()), 2)


def hilar_point(mask, pa_mask, ctx, masks=None):
    """Hilar point proxy (pulmonary veins are not segmented): lateral edge of the main pulmonary
    arterial trunk next to the mediastinum, at the level where it meets the airway -- the left
    PA arches over the left main bronchus (row of its midpoint), the right interlobar artery runs
    below the right main bronchus (1 cm lower). The PA masks are whole arterial trees, so a
    tree-level rule (e.g. its upper half) lands on peripheral branches."""
    if pa_mask is None or not pa_mask.any():
        return overlap_centroid(mask, pa_mask)
    masks = masks or {}
    side = _side_of_mask(mask, ctx)
    bif = masks.get("tracheal bifurcation")
    br = bronchus_mid(bif, side, ctx) if bif is not None and bif.any() else None
    med = _union(masks, ("heart", "cardiomediastinum"))
    if br is None or med is None:
        return _hilar_point_tree(mask, pa_mask, ctx, masks)
    lbl, n = ndi.label(pa_mask)
    if n > 1:                                             # main tree only (drop speckle)
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        pa_mask = lbl == (1 + int(np.argmax(sizes)))
    y_ref = br[1] + (0 if side == "left" else int(ctx["px_per_cm"]))
    ys, xs = np.where(pa_mask)
    dmed = ndi.distance_transform_edt(~med)[ys, xs]
    sel = (np.abs(ys - y_ref) <= 3) & (dmed <= 3.0 * ctx["px_per_cm"])
    if sel.sum() < 3:
        return _hilar_point_tree(mask, pa_mask, ctx, masks)
    o = 1 if side == "left" else -1
    best = None
    for y in np.unique(ys[sel]):
        r = np.sort(xs[sel & (ys == y)])
        runs = np.split(r, np.where(np.diff(r) > 1)[0] + 1)
        run = min(runs, key=lambda q: np.min(np.abs(_anat_left(q, ctx))))      # trunk = medial-most run
        x = run[np.argmax(o * _anat_left(run, ctx))]                          # its lateral edge
        if best is None or abs(y - y_ref) < abs(best[1] - y_ref):
            best = (int(x), int(y), 2)
    return best


def _hilar_point_tree(mask, pa_mask, ctx, masks=None):
    """Fallback: lateral-most PA point in the tree's upper half within 2 cm of the mediastinum."""
    ys, xs = np.where(pa_mask)
    up = ys <= ys.min() + 0.5 * (ys.max() - ys.min())
    med = _union(masks or {}, ("heart", "cardiomediastinum"))
    if med is not None:
        near = ndi.distance_transform_edt(~med)[ys, xs] <= 2.0 * ctx["px_per_cm"]
        if (up & near).sum() >= 5:
            up = up & near
    xs, ys = xs[up], ys[up]
    own = 1 if _anat_left(np.where(mask)[1], ctx).mean() >= 0 else -1
    j = int(np.argmax(own * _anat_left(xs, ctx)))
    return (int(xs[j]), int(ys[j]), 2)


def lateral_at_row(mask, y, ctx, band=6):
    """Lateral-most pixel of a (single-side) lung within +-band rows of y."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or y is None:
        return None
    sel = np.abs(ys - y) <= band
    if sel.sum() < 3:
        return None
    xs, ys = xs[sel], ys[sel]
    own = 1 if _anat_left(np.where(mask)[1], ctx).mean() >= 0 else -1
    j = int(np.argmax(own * _anat_left(xs, ctx)))
    return (int(xs[j]), int(ys[j]), 2)


def ctr_thorax_rib(mask, masks, ctx):
    """CTR denominator endpoint on the side opposite the reference: inner thoracic
    margin at the level where the SAME-NUMBERED posterior rib meets the lateral
    chest wall as on the right (ribs slope, so a shared image row is wrong)."""
    rl = masks.get("right lung")
    ref = ctr_thorax(rl, masks.get("right hemidiaphragm"), ctx) if rl is not None and rl.any() else None
    if ref is None:
        return ctr_thorax(mask, masks.get("right hemidiaphragm"), ctx)
    best, bn = 1e9, None
    for n in range(4, 12):
        rm = masks.get(f"posterior {KS._ordinal(str(n))} rib right")
        if rm is None or not rm.any():
            continue
        # outside: distance to the rib; inside: minus the depth (prefer the rib whose
        # centre line the point lies on when two ribs overlap at the chest wall)
        d = ndi.distance_transform_edt(~rm)[ref[1], ref[0]] - ndi.distance_transform_edt(rm)[ref[1], ref[0]]
        if d < best:
            best, bn = d, n
    lm = masks.get(f"posterior {KS._ordinal(str(bn))} rib left") if bn else None
    if lm is None or not lm.any():
        return ctr_thorax(mask, masks.get("right hemidiaphragm"), ctx)
    own = 1 if _anat_left(np.where(mask)[1], ctx).mean() >= 0 else -1
    prof = _row_profile(mask, lambda xs: own * _anat_left(xs, ctx))
    dt = ndi.distance_transform_edt(~lm)
    rows = sorted(prof)
    d = np.array([dt[y, prof[y][0]] for y in rows])
    hit = [y for y, v in zip(rows, d) if v <= d.min() + 1]
    y = int(np.median(hit))
    # sanity: asymmetric rib projection can move the matched rib far from the CTR level ->
    # fall back to the left dome level
    if abs(y - ref[1]) > 2.5 * ctx["px_per_cm"]:
        hl = hemi_on("left", masks, ctx)
        dome = dome_lateral(hl, masks.get("heart"), ctx) if hl is not None else None
        y = dome[1] if dome is not None else ref[1]
        y = min(prof, key=lambda r: abs(r - y))
    p = (prof[y][0], int(y), 2)
    heart = masks.get("heart")
    if heart is not None and heart.any():                # never inside the cardiac border
        hb = _pt_extreme_side(heart, ctx, "left")
        if hb is not None and own * _anat_left([p[0]], ctx)[0] <= own * _anat_left([hb[0]], ctx)[0]:
            y = max(prof, key=lambda r: prof[r][1])      # widest inner thoracic width
            p = (prof[y][0], int(y), 2)
    return p


def dome_lateral(mask, heart, ctx):
    """Hemidiaphragm dome taken on the ridge LATERAL to the cardiac silhouette (the
    medial ridge under the heart is not the radiographic dome)."""
    bx, by = superior_boundary(mask)
    if bx is None:
        return None
    if heart is not None and heart.any():
        hy, hx = np.where(heart)
        hx = hx[hy < by.min()]                 # cardiac border above the diaphragm only
        if len(hx) == 0:
            hx = np.where(heart)[1]
        own = 1 if _anat_left(bx, ctx).mean() >= 0 else -1
        hl = (own * _anat_left(hx, ctx)).max()
        sel = own * _anat_left(bx, ctx) > hl
        if sel.sum() >= 10:
            bx, by = bx[sel], by[sel]
    j = int(np.argmin(by))
    return (int(bx[j]), int(by[j]), 2)


def heart_apex(mask, hemi, ctx):
    """Infero-lateral tip of the left heart contour ABOVE the left dome."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if hemi is not None and hemi.any():
        bx, by = superior_boundary(hemi)
        top = dict(zip(bx.tolist(), by.tolist()))
        keep = np.array([y < top.get(x, 1 << 20) for x, y in zip(xs, ys)])
        if keep.sum() >= 20:
            xs, ys = xs[keep], ys[keep]
    m2 = np.zeros_like(mask)
    m2[ys, xs] = True
    return _pt_extreme_side(m2, ctx, "left", vy=1, hx=1)


def cardiophrenic_ivc(mask, ivc, hemi, ctx):
    """Right cardiophrenic angle where the right heart / IVC silhouette meets the right
    hemidiaphragm: among silhouette pixels nearest the hemidiaphragm, the right-most."""
    sil = mask | ivc if ivc is not None else mask
    if hemi is None or not hemi.any():
        return _pt_extreme_side(sil, ctx, "right", vy=1, hx=1)
    bx, by = superior_boundary(hemi)
    top = dict(zip(bx.tolist(), by.tolist()))
    # walk DOWN the right heart/IVC border (outer silhouette edge per row) from the
    # right cardiac border until it reaches the diaphragm contour
    prof = _row_profile(sil, lambda q: -_anat_left(q, ctx))
    start = _pt_extreme_side(mask, ctx, "right")
    last = None
    for y in sorted(r for r in prof if r >= start[1]):
        x = prof[y][0]
        if y >= top.get(x, 1 << 20):
            break
        last = (x, int(y), 2)
    return last or _pt_extreme_side(sil, ctx, "right", vy=1, hx=1)


LMC_PARTS = ("cardiomediastinum", "heart", "aortic arch", "pulmonary trunc", "upper mediastinum")


def left_mediastinal(which, masks, ctx):
    """Landmarks on the LEFT cardiomediastinal contour (outer silhouette edge per row):
    ap_window = deepest point between knob and PA segment; pa_segment = PA convexity;
    laa = contour at the left-atrial-appendage level; waist = deepest point below the
    chord from knob to LV border."""
    sil = _union(masks, LMC_PARTS)
    arch, pa, heart = masks.get("aortic arch"), masks.get("pulmonary trunc"), masks.get("heart")
    if sil is None or arch is None or heart is None or not arch.any() or not heart.any():
        return None
    prof = _row_profile(sil, lambda xs: _anat_left(xs, ctx))
    knob = knob_on_silhouette(arch, ctx, masks)
    lv = _pt_extreme_side(heart, ctx, "left")
    if knob is None or lv is None:
        return None
    if which == "waist":
        return _chord_knee(prof, knob[1], lv[1])
    pa_y = _pt_extreme_side(pa, ctx, "left")[1] if pa is not None and pa.any() else (knob[1] + lv[1]) // 2
    if which == "ap_window":
        return _chord_knee(prof, knob[1], pa_y)
    ap = _chord_knee(prof, knob[1], pa_y)
    ap_y = ap[1] if ap else knob[1]
    rows = [y for y in sorted(prof) if ap_y < y < lv[1]]
    if not rows:
        return None
    # the heart mask already covers the PA / LA-appendage region, so the silhouette has
    # no separable moguls: anchor each on the contour at its own structure's level.
    if which == "pa_segment":
        y = min(rows, key=lambda r: abs(r - pa_y))
        return (prof[y][0], int(y), 2)
    if which == "laa":
        la = masks.get("heart atrium left")
        lo = min(rows, key=lambda r: abs(r - pa_y))
        y = _pt_extreme_side(la, ctx, "left")[1] if la is not None and la.any() else (lo + lv[1]) // 2
        if not lo + 0.15 * (lv[1] - lo) < y < lv[1] - 0.15 * (lv[1] - lo):
            y = (lo + lv[1]) // 2                  # LA edge outside the PA-LV span
        y = min(rows, key=lambda r: abs(r - y))
        return (prof[y][0], int(y), 2)
    return None


def knob_on_silhouette(mask, ctx, masks):
    """Aortic knob on the LEFT mediastinal outline: within 3 cm below the arch summit, the most
    lateral outline point where the arch itself forms the outline; otherwise the first outward
    bulge of the outline below the summit. (The arch mask can stay inside the silhouette, and it
    also reaches down along the descending aorta where the cardiac outline is wider.)"""
    sil = _union(masks, LMC_PARTS)
    if sil is None or not mask.any():
        return _pt_extreme_side(mask, ctx, "left")
    top = summit_distal(mask, ctx)
    prof = _row_profile(sil, lambda q: _anat_left(q, ctx))
    win = [y for y in sorted(prof) if top[1] <= y <= top[1] + 3.0 * ctx["px_per_cm"]]
    if not win:
        return _pt_extreme_side(mask, ctx, "left")
    dt = ndi.distance_transform_edt(~mask)
    own = [y for y in win if dt[y, prof[y][0]] <= 4]
    if own:
        y = max(own, key=lambda r: prof[r][1])
        return (prof[y][0], int(y), 2)
    v = np.convolve([prof[y][1] for y in win], np.ones(5) / 5, "same")
    peaks = [k for k in range(2, len(win) - 2) if v[k] >= v[k - 1] and v[k] >= v[k + 1]]
    y = win[peaks[0]] if peaks else win[min(len(win) - 1, int(ctx["px_per_cm"]))]
    return (prof[y][0], int(y), 2)


def anat_right_extreme(mask, ctx):
    """Right-most (patient's right) point of the mask, e.g. the left-atrial double density."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    j = int(np.argmin(_anat_left(xs, ctx)))
    return (int(xs[j]), int(ys[j]), 2)


def summit_distal(mask, ctx, frac=0.4):
    """Top of the aortic knob: highest point of the DISTAL (lateral, lung-bordered)
    part of the arch rather than its summit under the great vessels."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    lat = _anat_left(xs, ctx)
    keep = lat >= lat.max() - frac * (lat.max() - lat.min())
    j = int(np.argmin(np.where(keep, ys, 1 << 20)))
    return (int(xs[j]), int(ys[j]), 2)


def mw_right(mask, arch, ctx, masks=None):
    """Right mediastinal-width endpoint: right edge of the upper mediastinum (SVC) at
    the aortic-knob level."""
    if arch is None or not arch.any():
        return None
    knob = knob_on_silhouette(arch, ctx, masks or {})
    ys, xs = np.where(mask)
    sel = np.abs(ys - knob[1]) <= 3
    if sel.sum() < 2:
        return None
    xs, ys = xs[sel], ys[sel]
    j = int(np.argmin(_anat_left(xs, ctx)))
    return (int(xs[j]), int(ys[j]), 2)


def carina_notch(mask):
    """Carina = apex of the notch between the main bronchi (top of the subcarinal
    contour): highest point of the concavity hull(mask) minus mask below the top."""
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
    gap = hull.astype(bool) & ~mask
    gap[: ys.min() + 2] = False
    lbl, n = ndi.label(gap)
    if n == 0:
        return None
    xm = np.median(xs)
    best, bj = 1e9, None
    for i in range(1, n + 1):
        gy, gx = np.where(lbl == i)
        if len(gx) < 5:
            continue
        if gy.max() < ys.max() - 0.2 * (ys.max() - ys.min()):
            continue                           # notch must open at the bottom of the Y
        score = abs(gx.mean() - xm)
        if score < best:
            best, bj = score, i
    if bj is None:
        return None
    gy, gx = np.where(lbl == bj)
    top = gy == gy.min()
    return (int(np.median(gx[top])), int(gy.min()), 2)


def bronchus_mid(mask, side, ctx):
    """Midpoint of the main-bronchus axis (half arc length along the limb centerline
    from the carina) on the named anatomical side."""
    car = carina_notch(mask)
    if car is None:
        return bronchus_point(mask, side, ctx)
    ys, xs = np.where(mask)
    s = 1 if side == "left" else -1
    sel = (ys >= car[1] - 2) & (s * ctx["left_sign"] * (xs - car[0]) > 0)
    if sel.sum() < 10:
        return bronchus_point(mask, side, ctx)
    limb = np.zeros_like(mask)
    limb[ys[sel], xs[sel]] = True
    return centerline_point(limb, 0.5)


def _bronchus_axis(mask, side, ctx):
    """Two points on the main-bronchus centerline (30% and 70% arc length from the carina)."""
    car = carina_notch(mask)
    if car is None:
        return None
    ys, xs = np.where(mask)
    s = 1 if side == "left" else -1
    sel = (ys >= car[1] - 2) & (s * ctx["left_sign"] * (xs - car[0]) > 0)
    if sel.sum() < 10:
        return None
    limb = np.zeros_like(mask)
    limb[ys[sel], xs[sel]] = True
    a, b = centerline_point(limb, 0.3), centerline_point(limb, 0.7)
    return None if a is None or b is None or a[:2] == b[:2] else (np.array(a[:2], float), np.array(b[:2], float))


def bronchial_axis_vertex(mask, ctx):
    """Vertex of the carinal (interbronchial) angle: intersection of the two main-bronchus
    axes. The angle is measured between the bronchial axes, which meet above the notch apex
    (the carina), so the notch apex itself underestimates the splay."""
    L, R = _bronchus_axis(mask, "left", ctx), _bronchus_axis(mask, "right", ctx)
    if L is None or R is None:
        return None
    (p1, p2), (q1, q2) = L, R
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
    x = p1 + t * d1
    if not (np.isfinite(x).all() and abs(x[1] - p1[1]) < 200):
        return None
    return (int(round(x[0])), int(round(x[1])), 2)


def subcarinal_limb(mask, side, ctx):
    """Subcarinal angle limb: point LIMB_CM from the carina (notch apex) along the inferior wall of
    the `side` main bronchus. Carina + both limbs give the splaying angle (left-atrial
    enlargement); measuring at the notch from the bronchus midpoints would flatten it."""
    car = carina_notch(mask)
    if car is None:
        return None
    p, q = _contour_walk(mask, car, LIMB_CM * ctx["px_per_cm"])
    if p is None:
        return None
    s = 1 if side == "left" else -1
    return p if s * (_anat_left([p[0]], ctx)[0] - _anat_left([q[0]], ctx)[0]) > 0 else q


def _path_walk_from(path, start_xy, d):
    """Point at arc length d along a centerline path (y,x rows), starting from the path end
    nearest `start_xy` and walking towards the other end."""
    if path is None or len(path) < 2:
        return None
    e0 = np.hypot(path[0, 1] - start_xy[0], path[0, 0] - start_xy[1])
    e1 = np.hypot(path[-1, 1] - start_xy[0], path[-1, 0] - start_xy[1])
    if e1 < e0:
        path = path[::-1]
    arc = _arclen(path)
    k = int(np.clip(np.searchsorted(arc, d), 0, len(path) - 1))
    return (int(path[k, 1]), int(path[k, 0]), 2)


def rib_slope_limb(mask, ctx, cm=3.0):
    """Rib-slope limb: point `cm` along the posterior rib's centerline from the rib head (its
    medial end); head + limb give the rib's inclination (horizontal ribs = hyperinflation)."""
    head = directed_extreme(mask, "medial", ctx, None, False)
    if head is None:
        return None
    return _path_walk_from(_skeleton_path(mask), head[:2], cm * ctx["px_per_cm"])


def manubriosternal_limb(mask, which, ctx):
    """Lateral view: sternal-centerline point 2 cm above ('manubrium') or 3 cm below ('body')
    the sternal angle; with the sternal angle they give the manubriosternal angle."""
    path = _skeleton_path(mask)
    sa = centerline_curvature(mask)
    if path is None or sa is None:
        return None
    j = int(np.argmin(np.hypot(path[:, 1] - sa[0], path[:, 0] - sa[1])))
    arc = _arclen(path) - _arclen(path)[j]
    d = (-2.0 if which == "manubrium" else 3.0) * ctx["px_per_cm"]   # path runs top -> bottom
    k = int(np.argmin(np.abs(arc - d)))
    return (int(path[k, 1]), int(path[k, 0]), 2)


def _posterior_sternal_surface(st, ctx):
    """dict row -> x of the posterior-most sternum pixel (lateral view)."""
    ys, xs = np.where(st)
    ap = ctx["ant_sign"] * (xs - ctx["mid_ap"])
    out = {}
    for y, x, a in zip(ys, xs, ap):
        if y not in out or a < out[y][1]:
            out[y] = (int(x), a)
    return out


def sternodiaphragmatic(mask, which, ctx, masks, cm=3.0):
    """Lateral-view sternodiaphragmatic angle (>=90 deg in hyperinflation), built from the sternum
    and the diaphragm contour directly: vertex = most anterior point of the diaphragm's superior
    contour still behind the posterior sternal surface; 'sternum' limb = posterior sternal surface
    `cm` above it; 'diaphragm' limb = `cm` posteriorly along the diaphragm contour."""
    st = masks.get("sternum")
    bx, by = superior_boundary(mask)
    if st is None or not st.any() or bx is None:
        return None
    # smooth the contour over ~1 cm so narrow slits between mask and anterior wall do not
    # pull the vertex down into a gutter
    o0 = np.argsort(bx)
    bx, by = bx[o0], ndi.median_filter(by[o0].astype(float), size=max(3, int(ctx["px_per_cm"])) | 1).astype(int)
    post = _posterior_sternal_surface(st, ctx)
    rows = np.array(sorted(post))
    ap = ctx["ant_sign"] * (bx - ctx["mid_ap"])
    def surf(y):
        r = int(rows[np.argmin(np.abs(rows - y))])
        return post[r]
    behind = np.array([a < surf(y)[1] for a, y in zip(ap, by)])
    lung = masks.get("less obstructed lung")
    if lung is not None and lung.any():                  # the contour under the visible lung only
        under = ndi.distance_transform_edt(~lung)[by, bx] <= ctx["px_per_cm"]
        if (behind & under).sum() >= 5:
            behind = behind & under
    if behind.sum() < 5:
        return None
    o = np.argsort(-ap)                                     # anterior -> posterior
    o = o[behind[o]]
    v = (int(bx[o[0]]), int(by[o[0]]), 2)
    d = cm * ctx["px_per_cm"]
    if which == "vertex":
        return v
    if which == "sternum":
        y = v[1] - d
        x, _ = surf(y)
        r = int(rows[np.argmin(np.abs(rows - y))])
        return (x, r, 2)
    # diaphragm limb: contour point `cm` behind the vertex (horizontal distance), so the limb
    # follows the dome and does not climb the anterior wall
    k = int(np.argmin(np.abs(ap[o] - (ap[o[0]] - d))))
    return (int(bx[o[k]]), int(by[o[k]]), 2)


def trachea_axis(mask, which, masks):
    """Tracheal axis point: mid-width of the trachea at 25% ('upper') of its height or 15% of
    its height above the carina ('lower'); mid-width is robust where the skeleton bends into a
    main bronchus."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y_top, y_bot = ys.min(), ys.max()
    bif = masks.get("tracheal bifurcation")
    if bif is not None and bif.any():
        y_bot = min(y_bot, int(np.where(bif)[0].min()))
    span = y_bot - y_top
    y = int(y_top + 0.25 * span) if which == "upper" else int(y_bot - 0.15 * span)
    rows = np.unique(ys)
    y = int(rows[np.argmin(np.abs(rows - y))])
    r = np.sort(xs[ys == y])
    runs = np.split(r, np.where(np.diff(r) > 1)[0] + 1)
    run = max(runs, key=len)                               # the tracheal lumen, not a stray piece
    return (int((run[0] + run[-1]) / 2), y, 2)


def tb_angle_right(mask, bif, ctx):
    """Right tracheobronchial angle (azygos-arch level): deepest inward point of the
    right outer contour of trachea+bifurcation between mid-trachea and the right
    main bronchus."""
    u = mask | bif if bif is not None else mask
    prof = _row_profile(u, lambda xs: -_anat_left(xs, ctx))
    ys = np.where(mask)[0]
    if len(ys) == 0:
        return None
    y0 = int(ys.min() + 0.5 * (ys.max() - ys.min()))
    if bif is not None and bif.any():
        by = np.where(bif)[0]
        y1 = int(by.min() + 0.6 * (by.max() - by.min()))
    else:
        y1 = int(ys.max())
    return _chord_knee(prof, y0, y1)


def trachea_at_clavicles(mask, masks, ctx):
    """Tracheal midpoint at the level of the medial clavicle ends."""
    yy = []
    for nm in ("clavicle left", "clavicle right"):
        m = masks.get(nm)
        if m is not None and m.any():
            cy, cx = np.where(m)
            j = int(np.argmin(np.abs(_anat_left(cx, ctx))))     # medial end (labels may be flipped)
            if cy[j] > 0:
                yy.append(int(cy[j]))
    ys, xs = np.where(mask)
    if len(xs) == 0 or not yy:
        return None
    y = int(np.mean(yy))
    rows = np.unique(ys)
    y = int(rows[np.argmin(np.abs(rows - y))])
    r = xs[ys == y]
    return (int((r.min() + r.max()) / 2), y, 2)


def fundus_under_diaphragm(mask, hemi, ctx, masks=None):
    """Gastric fundus apex: highest stomach point BELOW the left hemidiaphragm ridge."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if hemi is not None and hemi.any():
        bx, by = superior_boundary(hemi)
        top = dict(zip(bx.tolist(), by.tolist()))
        keep = np.array([y >= top.get(x, 1 << 20) for x, y in zip(xs, ys)]) & (_anat_left(xs, ctx) > 0)
        eso = (masks or {}).get("esophagus")
        if eso is not None and eso.any():                 # the fundus, not the cardia next to the GEJ
            g = centerline_point(eso, 1.0)
            if g is not None:
                lat_gej = _anat_left([g[0]], ctx)[0] + 1.5 * ctx["px_per_cm"]
                k2 = keep & (_anat_left(xs, ctx) >= lat_gej)
                if k2.sum() >= 10:
                    keep = k2
        if keep.sum() >= 10:
            xs, ys = xs[keep], ys[keep]
    j = int(np.argmin(ys))
    return (int(xs[j]), int(ys[j]), 2)


def ridge_mid(mask):
    """Midpoint (by column) of the superior border, e.g. the minor fissure on the
    middle lobe between chest wall and hilum."""
    bx, by = superior_boundary(mask)
    if bx is None:
        return None
    j = len(bx) // 2
    return (int(bx[j]), int(by[j]), 2)


def retrosternal(which, mask, sternum, ctx, masks=None):
    """Retrosternal space 3 cm below the sternal angle (lateral view): posterior sternal surface
    ('sternal') and anterior ascending-aortic wall ('aortic') on ONE common row -- the target row
    clamped into the ascending aorta's vertical extent, so the pair stays horizontal."""
    if sternum is None or not sternum.any():
        return None
    sa = centerline_curvature(sternum)
    y = int(sa[1] + 3.0 * ctx["px_per_cm"])
    ao = mask if which == "aortic" else (masks or {}).get("ascending aorta")
    if ao is not None and ao.any():
        ay = np.where(ao)[0]
        y = int(np.clip(y, ay.min() + 2, ay.max() - 2))
    m = sternum if which == "sternal" else mask
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    rows = np.unique(ys)
    y = int(rows[np.argmin(np.abs(rows - y))])
    r = xs[ys == y]
    ap = ctx["ant_sign"] * (r - ctx["mid_ap"])
    x = r[np.argmin(ap)] if which == "sternal" else r[np.argmax(ap)]
    return (int(x), y, 2)


def hoffman_rigler(mask, ivc, ctx):
    """Hoffman-Rigler point (lateral view): posterior heart (LV) border 2 cm above
    the IVC crossing."""
    if ivc is None or not ivc.any():
        return None
    y = int(np.where(ivc)[0].min() - 2.0 * ctx["px_per_cm"])
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    rows = np.unique(ys)
    y = int(rows[np.argmin(np.abs(rows - y))])
    r = xs[ys == y]
    x = r[np.argmin(ctx["ant_sign"] * (r - ctx["mid_ap"]))]
    return (int(x), y, 2)


def endplate_corner(mask, token, ctx):
    """Vertebral body corner = lateral (or anterior/posterior) end of the superior /
    inferior endplate: maximise the normalised diagonal score inside the body, so the
    point sits on the corner and not mid-way along the side wall or at the notch."""
    lateral = any(c in token for c in "AP")
    body = vertebral_body(mask, lateral, ctx)
    ys, xs = np.where(body)
    if len(xs) == 0:
        return None
    v = (ys - ys.min()) / max(1, ys.max() - ys.min())
    v = v if "I" in token else -v
    hor = next(c for c in token if c in "LRAP")
    h = _anat_left(xs, ctx) if hor in "LR" else ctx["ant_sign"] * (xs - ctx["mid_ap"])
    u = (h - h.min()) / max(1e-6, h.max() - h.min())
    u = u if hor in "LA" else -u
    j = int(np.argmax(v + u))
    return (int(xs[j]), int(ys[j]), 2)


# ---------------- per-image context ---------------- #
def make_ctx(masks, W, H):
    def cx(name):
        m = masks.get(name)
        if m is not None and m.any():
            return float(np.where(m)[1].mean())
        return None
    mid_x = cx("spine") or cx("thoracic spine") or cx("heart") or W / 2.0
    # anatomical-left image sign: where do known left structures sit vs midline?
    left_sign = 1
    for ref in ("left lung", "left hemidiaphragm", "scapula left", "heart ventricle left"):
        c = cx(ref)
        if c is not None:
            left_sign = 1 if c >= mid_x else -1
            break
    # AP axis (lateral view): anterior = toward sternum, posterior = toward spine
    stx = cx("sternum")
    spx = cx("spine") or cx("thoracic spine")
    if stx is not None and spx is not None:
        ant_sign = 1 if stx >= spx else -1
        mid_ap = (stx + spx) / 2.0
    else:
        ant_sign, mid_ap = 1, W / 2.0
    # pixel scale (no spacing metadata): a thoracic vertebral level (body + disc)
    # spans ~2.5 cm, so px/cm = median centroid pitch of consecutive T-levels / 2.5.
    cs = []
    for n in range(1, 13):
        m = masks.get(f"vertebrae T{n}")
        cs.append(np.argwhere(m).mean(0) if m is not None and m.any() else None)
    pitch = [float(np.hypot(*(b - a))) for a, b in zip(cs, cs[1:]) if a is not None and b is not None]
    px_per_cm = float(np.median(pitch)) / 2.5 if pitch else 14.0
    return {"mid_x": mid_x, "left_sign": left_sign, "ant_sign": ant_sign, "mid_ap": mid_ap,
            "px_per_cm": px_per_cm}


def _side_from(name, cls_side):
    """Return (side, from_name). from_name=True if the side is encoded in the
    keypoint name (structure follows the anatomical L/R convention); False if it
    only comes from the class (may be flipped, e.g. the diaphragm)."""
    toks = name.split("_")
    if "left" in toks:
        return "left", True
    if "right" in toks:
        return "right", True
    if cls_side in ("left", "right"):
        return cls_side, False
    return None, False


def compute_kp(kp, mask, cls, ctx, masks, view="frontal"):
    m = kp.method
    if m == "body_centroid":
        return centroid(vertebral_body(mask, view == "lateral", ctx))
    if m.startswith("extreme:"):
        side, from_name = _side_from(kp.name, cls.side)
        return directed_extreme(mask, m.split(":", 1)[1], ctx, side, from_name)
    if m.startswith("ridge:"):
        return ridge_extreme(mask, m.split(":", 1)[1], ctx, _side_from(kp.name, cls.side)[0])
    if m == "centroid":
        side, from_name = _side_from(kp.name, cls.side)
        return centroid(restrict_half(mask, side, ctx) if from_name else mask)
    if m == "overlap_centroid":
        return overlap_centroid(mask, masks.get(kp.via[0]) if kp.via else None)
    if m == "nearest":
        side, from_name = _side_from(kp.name, cls.side)
        mk = restrict_half(mask, side, ctx) if from_name else mask
        return nearest_point(mk, masks.get(kp.via[0]) if kp.via else None)
    if m.startswith("corner:"):
        return rect_corner(mask, m.split(":", 1)[1], ctx)
    if m.startswith("skel:"):
        return skeleton_point(mask, m.split(":", 1)[1])
    if m.startswith("centerline:"):
        return centerline_point(mask, float(m.split(":", 1)[1]))
    if m == "curv:centerline":
        return centerline_curvature(mask)
    if m == "curv:max":
        return curvature_max(mask, ctx)
    if m.startswith("contact:"):
        other = masks.get(kp.via[0]) if kp.via else None
        return contact_point(mask, other, m.split(":", 1)[1], ctx)
    if m == "rib_costal":
        return rib_costal(mask, [masks.get(v) for v in kp.via], ctx)
    if m == "index:ctr_thorax":
        return ctr_thorax(mask, masks.get(kp.via[0]) if kp.via else None, ctx)
    if m.startswith("bronchus:"):
        return bronchus_point(mask, m.split(":", 1)[1], ctx)
    via = [masks.get(v) for v in kp.via]
    v0 = via[0] if via else None
    if m.startswith("visible_corner:"):
        d = m.split(":", 1)[1]
        fb = {"infero_lateral": lambda: directed_extreme(mask, "infero_lateral", ctx, None, False),
              "antero_inferior": lambda: ridge_extreme(mask, "anterior", ctx),
              "postero_inferior": lambda: ridge_extreme(mask, "posterior", ctx)}[d]
        if view == "frontal":
            vis = visible_lung(_side_of_mask(mask, ctx), masks, ctx, mask)
            return directed_extreme(vis, d, ctx, None, False) if vis is not None else fb()
        return visible_corner(_union(masks, ("less obstructed lung", "lung")), d, ctx, masks, fb)
    if m.startswith("sulcus_vertex:"):
        return sulcus_vertex(mask, m.split(":", 1)[1], ctx, masks)
    if m.startswith("ridge_under_lung:"):
        return ridge_under_lung(mask, masks.get("less obstructed lung"), m.split(":", 1)[1], ctx)
    if m.startswith("cp_limb:"):
        return costophrenic_limb(mask, m.split(":")[1], ctx, masks)
    if m.startswith("sulcus_limb:"):
        _, sul, which = m.split(":")
        return sulcus_limb(mask, sul, which, ctx, masks)
    if m.startswith("cph_limb:"):
        _, side, which = m.split(":")
        return cardiophrenic_limb(mask, side, which, ctx, masks)
    if m == "apex_lateral":
        return apex_lateral(mask, ctx, masks)
    if m == "plateau:top":
        return plateau_top(mask)
    if m == "hilar_point":
        return hilar_point(mask, v0, ctx, masks)
    if m == "lateral_at_hilum":
        h = hilar_point(mask, v0, ctx, masks)
        return lateral_at_row(mask, h[1] if h else None, ctx)
    if m == "index:ctr_thorax_rib":
        return ctr_thorax_rib(mask, masks, ctx)
    if m == "dome_lateral":
        return dome_lateral(mask, v0, ctx)
    if m == "heart_apex":
        return heart_apex(mask, hemi_on("left", masks, ctx), ctx)
    if m == "cardiophrenic_ivc":
        return cardiophrenic_ivc(mask, v0, hemi_on("right", masks, ctx), ctx)
    if m.startswith("lmc:"):
        return left_mediastinal(m.split(":", 1)[1], masks, ctx)
    if m == "knob_silhouette":
        return knob_on_silhouette(mask, ctx, masks)
    if m == "anat:right":
        return anat_right_extreme(mask, ctx)
    if m == "summit_distal":
        return summit_distal(mask, ctx)
    if m == "mw_right":
        return mw_right(mask, v0, ctx, masks)
    if m == "rib_slope_limb":
        return rib_slope_limb(mask, ctx)
    if m.startswith("manubriosternal_limb:"):
        return manubriosternal_limb(mask, m.split(":", 1)[1], ctx)
    if m.startswith("sternodiaphragmatic:"):
        return sternodiaphragmatic(mask, m.split(":", 1)[1], ctx, masks)
    if m.startswith("trachea_axis:"):
        return trachea_axis(mask, m.split(":", 1)[1], masks)
    if m.startswith("subcarinal_limb:"):
        return subcarinal_limb(mask, m.split(":", 1)[1], ctx)
    if m == "carina_notch":
        return carina_notch(mask) or directed_extreme(mask, "top", ctx, None, False)
    if m.startswith("bronchus_mid:"):
        return bronchus_mid(mask, m.split(":", 1)[1], ctx)
    if m == "tb_angle:right":
        return tb_angle_right(mask, v0, ctx)
    if m == "trachea_at_clavicles":
        return trachea_at_clavicles(mask, masks, ctx)
    if m == "fundus_under_diaphragm":
        return fundus_under_diaphragm(mask, hemi_on("left", masks, ctx), ctx, masks)
    if m == "ridge_mid":
        return ridge_mid(mask)
    if m.startswith("retrosternal:"):
        return retrosternal(m.split(":", 1)[1], mask, v0, ctx, masks)
    if m == "hoffman_rigler":
        return hoffman_rigler(mask, v0, ctx)
    if m.startswith("endplate:"):
        return endplate_corner(mask, m.split(":", 1)[1], ctx)
    return None                            # todo -> v=0


def extract_for_class(cname, mask, schema, ctx, masks):
    cls = schema[cname]
    view = "lateral" if schema is KS.SCHEMA_LATERAL else "frontal"
    out = {}
    H, W = mask.shape
    for kp in cls.keypoints:
        pt = compute_kp(kp, mask, cls, ctx, masks, view)
        # a point on the image border is a field-of-view cut, not the landmark
        if pt and not (1 < pt[0] < W - 2 and 1 < pt[1] < H - 2):
            pt = None
        out[kp.name] = pt if pt else (0, 0, 0)
    return out


# ---------------- overlay validation ---------------- #
def _draw(img, pt, color, label):
    if pt is None or pt[2] == 0:
        return
    x, y = pt[0], pt[1]
    cv2.circle(img, (x, y), 5, color, -1)
    cv2.circle(img, (x, y), 5, (255, 255, 255), 1)
    cv2.putText(img, label, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def run_demo(image_ids, coco, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    name2id = {v: k for k, v in cats.items()}
    imgs = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    want = ["left lung", "right lung", "left hemidiaphragm", "right hemidiaphragm", "heart",
            "spine", "left pulmonary artery ", "right pulmonary artery ", "scapula left"]
    CY, YE, MA = (255, 255, 0), (0, 255, 255), (255, 0, 255)   # lung, diaphragm, heart (BGR)
    GRP = {"costophrenic_left": (0, 255, 0), "costophrenic_right": (0, 200, 0),
           "cardiophrenic_left": (0, 165, 255), "cardiophrenic_right": (0, 120, 255)}

    for iid in image_ids:
        fn = imgs[iid]["file_name"]
        img = cv2.imread(os.path.join(IMGDIR, fn))
        if img is None:
            print("  missing image", fn); continue
        H, W = img.shape[:2]
        masks = {}
        for a in by_img[iid]:
            nm = cats[a["category_id"]]
            if nm in want:
                mk = decode_rle(a["segmentation"])
                if mk.any():
                    masks[nm] = clean_mask(mk)
        ctx = make_ctx(masks, W, H)

        kpts = {}
        for cname, col in (("left lung", CY), ("right lung", CY),
                           ("left hemidiaphragm", YE), ("right hemidiaphragm", YE),
                           ("heart", MA)):
            if cname not in masks:
                continue
            kpts[cname] = extract_for_class(cname, masks[cname], KS.SCHEMA_FRONTAL, ctx, masks)

        # draw points (abbreviated labels)
        abbr = {"apex": "apex", "lateral_margin": "LM", "costophrenic_pulmonary": "CP",
                "cardiophrenic_pulmonary": "CPh", "hilum": "hil", "dome": "dome",
                "costophrenic_diaphragmatic": "CP", "cardiophrenic_diaphragmatic": "CPh",
                "right_cardiac_border": "Rb", "left_cardiac_border": "Lb", "waist_cardiac": "waist",
                "cardiophrenic_cardiac_left": "CPh-L", "cardiophrenic_cardiac_right": "CPh-R"}
        colmap = {"left lung": CY, "right lung": CY, "left hemidiaphragm": YE,
                  "right hemidiaphragm": YE, "heart": MA}
        for cname, kps in kpts.items():
            for nm, pt in kps.items():
                _draw(img, pt, colmap[cname], abbr.get(nm, nm[:4]))

        # group links across instances (the decomposed angle)
        grouped = {}
        for cname, kps in kpts.items():
            cls = KS.SCHEMA_FRONTAL[cname]
            for kp in cls.keypoints:
                if kp.group in GRP and kps[kp.name][2] > 0:
                    grouped.setdefault(kp.group, []).append(kps[kp.name][:2])
        for g, pts in grouped.items():
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    cv2.line(img, pts[i], pts[j], GRP[g], 1, cv2.LINE_AA)

        cv2.line(img, (int(ctx["mid_x"]), 0), (int(ctx["mid_x"]), H), (120, 120, 120), 1)
        outp = os.path.join(out_dir, fn.replace(".png", "_kp.png"))
        cv2.imwrite(outp, img)
        print("  wrote", outp,
              "| pts:", {c: sum(1 for p in k.values() if p[2] > 0) for c, k in kpts.items()})


# ---------------- generic overlay (any view / class list) ---------------- #
CTX_REFS = ["spine", "thoracic spine", "sternum", "left lung", "left hemidiaphragm",
            "scapula left", "heart ventricle left"] + [f"vertebrae T{n}" for n in range(1, 13)]
PALETTE = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 165, 255),
           (255, 128, 0), (128, 255, 0), (255, 0, 128), (0, 215, 255)]


def _load_masks(iid, by_img, cats, names):
    masks = {}
    for a in by_img[iid]:
        nm = cats[a["category_id"]]
        if nm in names:
            mk = decode_rle(a["segmentation"])
            if mk.any():
                masks[nm] = clean_mask(mk)
    return masks


def overlay(iid, idx, schema, classes, out_dir, suffix):
    cats, by_img, imgs = idx
    fn = imgs[iid]["file_name"]
    img = cv2.imread(os.path.join(IMGDIR, fn))
    if img is None:
        print("  missing image", fn); return
    H, W = img.shape[:2]
    need = set(classes) | set(CTX_REFS)
    for c in classes:
        for kp in schema[c].keypoints:
            need |= set(kp.via)
    masks = _load_masks(iid, by_img, cats, need)
    ctx = make_ctx(masks, W, H)
    cv2.line(img, (int(ctx["mid_x"]), 0), (int(ctx["mid_x"]), H), (90, 90, 90), 1)
    stats = {}
    for ci, c in enumerate(classes):
        if c not in masks:
            continue
        col = PALETTE[ci % len(PALETTE)]
        kps = extract_for_class(c, masks[c], schema, ctx, masks)
        for a, b in schema[c].skeleton:
            pa, pb = kps.get(a), kps.get(b)
            if pa and pb and pa[2] > 0 and pb[2] > 0:
                cv2.line(img, tuple(pa[:2]), tuple(pb[:2]), col, 1, cv2.LINE_AA)
        for nm, p in kps.items():
            _draw(img, p, col, nm[:7])
        stats[c] = sum(1 for p in kps.values() if p[2] > 0)
    cv2.imwrite(os.path.join(out_dir, fn.replace(".png", f"_{suffix}.png")), img)
    print("  wrote", fn, suffix, stats)


if __name__ == "__main__":
    # python -m cxrlandmarks.rules.extract --paxray-root DIR --out-dir DIR
    import argparse
    _ap = argparse.ArgumentParser(description="Overlay demo of the landmark rules on PAXRay++ test images.")
    _ap.add_argument("--paxray-root", required=True)
    _ap.add_argument("--out-dir", required=True)
    _args = _ap.parse_args()
    DATA = _args.paxray_root
    IMGDIR = os.path.join(DATA, "images_patlas")
    OUT = _args.out_dir
    os.makedirs(OUT, exist_ok=True)
    print("loading test.json ...")
    coco = json.load(open(os.path.join(DATA, "paxray_test.json")))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    idx = (cats, by_img, imgs)

    def area(a):
        c = dict(a["segmentation"])
        if isinstance(c["counts"], str):
            c["counts"] = c["counts"].encode()
        return int(M.area(c))

    print("[a] frontal costophrenic/cardiophrenic groups:")
    run_demo([51, 134, 144, 154], coco, OUT)

    print("[2] frontal corner/skel/curv (vertebrae, descending aorta, trachea):")
    for iid in [51, 134, 144]:
        overlay(iid, idx, KS.SCHEMA_FRONTAL,
                ["vertebrae T5", "vertebrae T6", "vertebrae T7", "vertebrae T8",
                 "descending aorta", "trachea"], OUT, "cornerskel")

    print("[1] lateral diaphragm/heart/sternum/vertebrae:")
    lat_ids = []
    for i in by_img:
        if "frontal" in imgs[i]["file_name"]:
            continue
        have = {cats[a["category_id"]] for a in by_img[i] if area(a) > 0}
        if {"diaphragm", "heart", "sternum", "vertebrae T7"} <= have \
                and os.path.exists(os.path.join(IMGDIR, imgs[i]["file_name"])):
            lat_ids.append(i)
        if len(lat_ids) >= 3:
            break
    for iid in lat_ids:
        overlay(iid, idx, KS.SCHEMA_LATERAL,
                ["diaphragm", "heart", "sternum", "vertebrae T6", "vertebrae T7",
                 "vertebrae T8", "spine", "posterior 7th rib"], OUT, "lateral")
    print("done")
