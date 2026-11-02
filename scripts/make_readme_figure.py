"""README figure: predicted named landmarks of selected clinical groups, one panel per group.

    python scripts/make_readme_figure.py --weights weights/joint_dense_v1 \
        --frontal chestxdet/test/41460.png --lateral paxraypp/images_patlas/1010a_lateral.png \
        --out assets/examples.png --font-dir <folder with Carlito .ttf>
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from cxrlandmarks.predict import LandmarkPredictor, load_image  # noqa: E402

VOCAB = Path(__file__).resolve().parents[1] / "src" / "cxrlandmarks" / "vocab"
PANELS = {
    "frontal": [("G01", "Cardiothoracic ratio"), ("G02", "Costophrenic angles"), ("G12", "Carina & main bronchi"),
                ("G09", "Aortic knob & arch"), ("G04", "Diaphragm domes"), ("G18", "Clavicles (rotation)")],
    "lateral": [("G17", "Sternum"), ("G19", "Vertebral corners"), ("G02", "Costophrenic sulci")],
}
SHORT = {"acromioclavicular_clavicular": "AC joint", "sternoclavicular_clavicular": "SC joint",
         "costophrenic_pulmonary": "CP angle", "ctr_thorax": "thorax (CTR)", "posterior_costophrenic": "posterior sulcus",
         "anterior_costophrenic": "anterior sulcus", "sternodiaphragmatic_vertex": "sterno-diaphragmatic",
         "left_main_bronchus": "L main bronchus", "right_main_bronchus": "R main bronchus"}
# Region colours of the paper's Fig. 1 and Fig. 2: a landmark takes the region of its group, a structure the
# region most of its landmarks belong to.
REGION_COLOR = {"Heart & mediastinum": "#e76f51", "Vessels": "#b5179e", "Airway": "#e9c46a",
                "Devices / lines": "#f4a261", "Lungs": "#2a9d8f", "Pleura & diaphragm": "#264653",
                "Bones": "#7b8cad", "Soft tissue": "#c9a0a6", "Abdomen / other": "#9aa0a6", "Meta": "#d5d8dc"}
GROUP_REGION = {"G01": "Heart & mediastinum", "G02": "Pleura & diaphragm", "G03": "Heart & mediastinum",
                "G04": "Pleura & diaphragm", "G05": "Lungs", "G06": "Lungs", "G07": "Heart & mediastinum",
                "G08": "Heart & mediastinum", "G09": "Vessels", "G10": "Vessels", "G11": "Vessels", "G12": "Airway",
                "G13": "Airway", "G14": "Devices / lines", "G15": "Devices / lines", "G16": "Heart & mediastinum",
                "G17": "Bones", "G18": "Bones", "G19": "Bones", "G20": "Bones", "G21": "Lungs", "G22": "Lungs",
                "G23": "Soft tissue", "G24": "Bones", "G25": "Vessels", "G26": "Abdomen / other", "G27": "Meta"}


def group_of() -> tuple[dict[tuple[str, str, str], str], dict[str, str]]:
    g, regions = {}, {}
    for r in csv.DictReader(open(VOCAB / "landmarks.csv")):
        for view in ("frontal", "lateral"):
            for c in filter(None, (s.strip() for s in r[f"classes_{view}"].split(";"))):
                g[(view, c, r["landmark"])] = r["group"]
                regions.setdefault(c, Counter())[GROUP_REGION[r["group"]]] += 1
    return g, {c: REGION_COLOR[cnt.most_common(1)[0][0]] for c, cnt in regions.items()}


def label(cls: str, lm: str) -> str | None:
    if "limb" in lm:
        return None
    if cls.startswith("vertebrae"):
        return cls.split()[-1] if lm in ("antero_superior", "superior_left") else None
    if lm in SHORT and lm.endswith("bronchus"):
        return SHORT[lm]
    words = cls.split()
    side = "R " if "right" in words else "L " if "left" in words else ""
    return side + SHORT.get(lm, lm.replace("_", " "))


def crop(pts: list[dict], h: int, w: int, frac: float = 0.45) -> tuple[float, float, float, float]:
    xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.4
    side = min(max(side, frac * min(h, w)), min(h, w))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x0 = min(max(cx - side / 2, 0), w - side)
    y0 = min(max(cy - side / 2, 0), h - side)
    return x0, x0 + side, y0 + side, y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frontal", required=True)
    ap.add_argument("--lateral", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--font-dir", type=Path, help="optional .ttf folder (the paper figures use Carlito)")
    a = ap.parse_args()
    if a.font_dir:
        for f in a.font_dir.glob("*.ttf"):
            fm.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Carlito"
    pred = LandmarkPredictor.from_pretrained(a.weights, device="cpu")
    gmap, cls_color = group_of()
    rows = [("frontal", a.frontal), ("lateral", a.lateral)]
    panels = [(view, path, g) for view, path in rows for g in PANELS[view]]
    ncol = 3
    fig, axes = plt.subplots(-(-len(panels) // ncol), ncol, figsize=(4.2 * ncol, 4.0 * -(-len(panels) // ncol)))
    cache = {}
    for ax, (view, path, (gid, title)) in zip(axes.flat, panels):
        if path not in cache:
            cache[path] = load_image(path), pred.predict(path, view=view)
        img, out = cache[path]
        ax.axis("off")
        ax.imshow(img, cmap="gray")
        pts = [lm for lm in out["landmarks"] if gmap.get((view, lm["class"], lm["landmark"])) == gid]
        x0, x1, y1, y0 = crop(pts, *img.shape[:2])
        masks = {st["class"]: st["mask_rle"] for st in out["structures"]}
        decoded = [(cls, mask_util.decode(masks[cls]).astype(float))
                   for cls in sorted({lm["class"] for lm in pts} & masks.keys())]
        for cls, m in sorted(decoded, key=lambda t: -t[1].sum()):  # small structures on top
            col = cls_color.get(cls, "#9aa0a6")
            ax.contourf(m, levels=[0.5, 1.5], colors=[col], alpha=0.18)
            ax.contour(m, levels=[0.5], colors=[col], linewidths=1.0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        placed = []
        if True:
            for lm in pts:
                col = REGION_COLOR[GROUP_REGION[gid]]
                ax.plot(lm["x"], lm["y"], "o", ms=4 if "limb" in lm["landmark"] else 7, mfc=col, mec="white", mew=1.0, zorder=4)
                text = label(lm["class"], lm["landmark"])
                if text is None:
                    continue
                left = text.startswith("R ") or (view == "lateral" and "anterior" in lm["landmark"])
                side = x1 - x0
                ty = lm["y"] + 0.012 * side
                while any(abs(ty - py) < 0.04 * side and abs(lm["x"] - px) < 0.25 * side and pl == left
                          for px, py, pl in placed):
                    ty += 0.04 * side
                placed.append((lm["x"], ty, left))
                ax.annotate(text, (lm["x"], lm["y"]), xytext=(lm["x"] + (-0.02 if left else 0.02) * side, ty),
                            ha="right" if left else "left", va="center",
                            textcoords="data", fontsize=10, color="#202124",
                            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
            ax.set_title(f"{title} ({view})", fontsize=13, fontweight="bold", color="#202124")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
