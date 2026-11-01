import csv
import json

import numpy as np

from cxrlandmarks.eval import gold as G
from cxrlandmarks.eval.cli import evaluate

DICOMS = [f"d{i:02d}" for i in range(8)]
POINTS = {"carina": (250.0, 200.0), "left_main_bronchus": (280.0, 230.0)}


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _release(tmp_path):
    (tmp_path / "readers").mkdir()
    _write(tmp_path / "images.csv", [dict(image_id=i + 1, dicom_id=d, view="frontal", width=512, height=512)
                                     for i, d in enumerate(DICOMS)], ["image_id", "dicom_id", "view", "width", "height"])
    _write(tmp_path / "scale.csv", [dict(dicom_id=d, **{"class": "tracheal bifurcation"}, area=10000.0, bbox_x=200.0, bbox_y=150.0,
                                         bbox_w=100.0, bbox_h=100.0) for d in DICOMS],
           ["dicom_id", "class", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    rng = np.random.default_rng(0)
    for r in G.READERS:
        rows = []
        for i, d in enumerate(DICOMS):
            for kp, (x, y) in POINTS.items():
                hidden = r == "R2" and i == 0 and kp == "carina"           # one visibility disagreement
                dx, dy = rng.normal(0, 3, 2)
                rows.append(dict(dicom_id=d, view="frontal", **{"class": "tracheal bifurcation"}, landmark=kp,
                                 status="not_visible" if hidden else "moved",
                                 x="" if hidden else x + dx, y="" if hidden else y + dy))
        _write(tmp_path / "readers" / f"{r}.csv", rows, ["dicom_id", "view", "class", "landmark", "status", "x", "y"])
    return tmp_path


def test_build_gold_consensus_and_sigma(tmp_path):
    gold, table = G.build(_release(tmp_path))
    assert len(gold["annotations"]) == len(DICOMS)
    first = next(a for a in gold["annotations"] if a["image_id"] == 1)
    cat = next(c for c in gold["categories"] if c["id"] == first["category_id"])
    j = cat["keypoints"].index("carina")
    assert first["keypoints"][3 * j + 2] == 0                              # disagreement -> not scored
    assert first["num_keypoints"] == 1
    sig = {t["landmark"]: t for t in table}
    assert sig["carina"]["n_pairs"] == len(DICOMS) - 1
    assert 0.01 < sig["carina"]["sigma"] < 0.06                          # 3 px reader noise / s=100


def test_gold_evaluation_perfect_prediction(tmp_path):
    gold, _ = G.build(_release(tmp_path))
    gt = tmp_path / "gold_kp.json"
    json.dump(gold, open(gt, "w"))
    preds = [dict(image_id=a["image_id"], category_id=a["category_id"], keypoints=a["keypoints"], score=1.0)
             for a in gold["annotations"]]
    m = evaluate(str(gt), preds, B=50, gold=True, min_anns=1)
    assert abs(m["oks_ap"] - 100) < 1e-6 and abs(m["pck"] - 100) < 1e-6 and m["mre"] < 1e-9
    assert m["dropped_classes"] == []
