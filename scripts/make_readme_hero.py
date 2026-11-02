"""README hero image: all predicted landmarks of a frontal and a lateral CXR, coloured by region, over the
predicted mask outlines, with callouts for a few named landmarks.

    python scripts/make_readme_hero.py --weights weights/joint_dense_v1 \
        --frontal chestxdet/test/41460.png --lateral paxraypp/images_patlas/1010a_lateral.png \
        --out assets/hero.png --font-dir <folder with Carlito .ttf>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_readme_figure import GROUP_REGION, group_of  # noqa: E402
from make_readme_figure import REGION_COLOR as PAPER_COLOR  # noqa: E402

from cxrlandmarks.predict import LandmarkPredictor, load_image  # noqa: E402

BG, CARD, INK, MUTED = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
# paper colours; the dark slate of "Pleura & diaphragm" is lightened to stay visible on the dark card
REGION_COLOR = {**PAPER_COLOR, "Pleura & diaphragm": "#6b97a8"}
DARK_FIX = {PAPER_COLOR["Pleura & diaphragm"]: REGION_COLOR["Pleura & diaphragm"]}
HIDDEN_GROUPS = {"G20", "G21", "G22", "G25", "G26", "G27"}   # ribs, lung zones, fissures, helpers: shown in the gallery
OUTLINES = {"frontal": ["right lung", "left lung", "heart", "clavicle left", "clavicle right", "trachea",
                        "aortic arch", "right hemidiaphragm", "left hemidiaphragm"],
            "lateral": ["sternum", "diaphragm", "trachea", "heart"]}
CALLOUTS = {"frontal": [("tracheal bifurcation", "carina", "Carina"),
                        ("aortic arch", "knob", "Aortic knob"),
                        ("heart", "right_cardiac_border", "R cardiac border"),
                        ("heart", "left_cardiac_border", "L cardiac border"),
                        ("right lung", "costophrenic_pulmonary", "R costophrenic angle"),
                        ("left lung", "costophrenic_pulmonary", "L costophrenic angle"),
                        ("right hemidiaphragm", "dome", "R hemidiaphragm dome"),
                        ("left hemidiaphragm", "dome", "L hemidiaphragm dome"),
                        ("clavicle right", "sternoclavicular_clavicular", "R sternoclavicular joint"),
                        ("clavicle left", "acromioclavicular_clavicular", "L acromioclavicular joint")],
            "lateral": [("sternum", "jugular_notch", "Jugular notch"),
                        ("sternum", "sternal_angle", "Sternal angle"),
                        ("diaphragm", "anterior_costophrenic", "Anterior sulcus")]}
REGIONS = ["Heart & mediastinum", "Vessels", "Airway", "Lungs", "Pleura & diaphragm", "Bones", "Devices / lines"]


def spread(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Push sorted label positions apart by at least `gap`, inside [lo, hi]."""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + gap) if out else max(y, lo))
    over = out[-1] - hi if out else 0
    if over > 0:
        out = [y - over for y in out]
    return out


def panel(fig, rect, img, out, view, gmap, cls_color, left_x, right_x):
    h, w = img.shape[:2]
    ax = fig.add_axes(rect)
    ax.set_facecolor(CARD)
    ax.imshow(img, cmap="gray", extent=(0, w, h, 0))
    masks = {st["class"]: st["mask_rle"] for st in out["structures"]}
    for cls in OUTLINES[view]:
        if cls in masks:
            m = mask_util.decode(masks[cls]).astype(float)
            col = DARK_FIX.get(cls_color.get(cls, MUTED), cls_color.get(cls, MUTED))
            ax.contourf(m, levels=[0.5, 1.5], colors=[col], alpha=0.16)
            ax.contour(m, levels=[0.5], colors=[col], linewidths=1.1, alpha=0.95)
    pos = {}
    for lm in out["landmarks"]:
        gid = gmap.get((view, lm["class"], lm["landmark"]))
        if gid is None or gid in HIDDEN_GROUPS:
            continue
        pos[(lm["class"], lm["landmark"])] = (lm["x"], lm["y"])
        small = "limb" in lm["landmark"]
        ax.plot(lm["x"], lm["y"], "o", ms=3.2 if small else 5.6, mfc=REGION_COLOR[GROUP_REGION[gid]],
                mec="white", mew=0.7, zorder=4)
    # callouts in the margins, left/right by image side
    sides = {"L": [], "R": []}
    for cls, kp, text in CALLOUTS[view]:
        if (cls, kp) in pos:
            x, y = pos[(cls, kp)]
            gid = gmap[(view, cls, kp)]
            sides["L" if x < w / 2 and left_x is not None else "R"].append((y, x, text, REGION_COLOR[GROUP_REGION[gid]]))
    for side, items in sides.items():
        if not items:
            continue
        items.sort()
        ys = spread([it[0] for it in items], gap=0.085 * h, lo=0.05 * h, hi=0.95 * h)
        lx = (left_x if side == "L" else right_x) * w
        for (y, x, text, col), ty in zip(items, ys):
            ex = -0.03 * w if side == "L" else 1.03 * w
            tx = lx - 0.02 * w if side == "L" else lx + 0.02 * w
            ax.plot([x, ex, tx], [y, ty, ty], color=col, lw=0.9,
                    alpha=0.9, zorder=3, clip_on=False, solid_capstyle="round")
            ax.plot(x, y, "o", ms=8, mfc="none", mec=col, mew=1.4, zorder=5)
            ax.text(lx, ty, text, color=INK, fontsize=13, va="center", ha="right" if side == "L" else "left",
                    clip_on=False)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--lateral", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--font-dir", type=Path)
    a = ap.parse_args()
    if a.font_dir:
        for f in a.font_dir.glob("*.ttf"):
            fm.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Carlito"
    pred = LandmarkPredictor.from_pretrained(a.weights, device="cpu")
    gmap, cls_color = group_of()
    W, H = 17.0, 8.0
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W)
    bg.set_ylim(0, H)
    bg.axis("off")
    bg.add_patch(FancyBboxPatch((0.2, 0.2), W - 0.4, H - 0.4, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=CARD, ec="#30363d", lw=1.0))
    # (view, image, x0, size, label anchors left/right as fractions of the image width)
    layout = [("frontal", a.frontal, 2.75, 5.7, -0.04, 1.04, "Frontal  ·  NIH ChestX-ray14 (unseen)"),
              ("lateral", a.lateral, 11.0, 4.4, None, 1.05, "Lateral  ·  PAXRay++ test (CT projection)")]
    top = 7.05
    for view, path, x0, side, lft, rgt, title in layout:
        img = load_image(path)
        out = pred.predict(path, view=view)
        y0 = top - side - (5.7 - side) / 2
        panel(fig, [x0 / W, y0 / H, side / W, side / H], img, out, view, gmap, cls_color, lft, rgt)
        bg.text(x0 + side / 2, y0 + side + 0.18, title, color=MUTED, fontsize=14,
                ha="center", va="bottom")
    renderer = fig.canvas.get_renderer()
    texts = [bg.text(0, 0.75, r, color=INK, fontsize=13, va="center") for r in REGIONS]
    to_in = W / fig.bbox.width
    widths = [0.17 + t.get_window_extent(renderer).width * to_in for t in texts]
    x = (W - sum(widths) - 0.55 * (len(REGIONS) - 1)) / 2
    for r, t, wd in zip(REGIONS, texts, widths):
        bg.plot(x, 0.75, "o", ms=10, mfc=REGION_COLOR[r], mec="white", mew=0.8)
        t.set_x(x + 0.17)
        x += wd + 0.55
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, transparent=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
