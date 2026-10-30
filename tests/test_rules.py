"""Fast checks of the packaged landmark rules and vocabulary (no PAXRay++ data needed)."""
from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from cxrlandmarks.rules import extract as E
from cxrlandmarks.rules import schema as KS

VOCAB = resources.files("cxrlandmarks") / "vocab"


def _vocab() -> dict:
    return json.loads((VOCAB / "final_vocab_v1.json").read_text())


def _csv(name: str) -> list[dict]:
    return list(csv.DictReader((VOCAB / name).read_text().splitlines()))


def test_schema_uses_packaged_vocab() -> None:
    p = Path(KS.FINAL_VOCAB_PATH)
    assert p.exists()
    assert p.resolve() == Path(str(VOCAB / "final_vocab_v1.json")).resolve()
    v = _vocab()["vocab"]
    for view, sch in (("frontal", KS.SCHEMA_FRONTAL), ("lateral", KS.SCHEMA_LATERAL)):
        # slot order follows the rule schema, not the vocab file
        assert {c: set(s.names()) for c, s in sch.items() if s.names()} == {c: set(n) for c, n in v[view].items() if n}
    # the pruned schema is a strict subset of the full rule set
    assert sum(len(s.keypoints) for s in KS.SCHEMA_FRONTAL_FULL.values()) > 203


def test_vocab_counts_match_paper() -> None:
    v = _vocab()
    pairs = {view: sum(len(n) for n in v["vocab"][view].values()) for view in ("frontal", "lateral")}
    assert pairs == {"frontal": 203, "lateral": 108}
    assert sum(pairs.values()) == 311
    assert len(v["provenance"]) == 311
    assert len({p["source"] for p in v["provenance"]}) == 132
    # K_max: per class the union of frontal and lateral slots (one COCO category serves both views)
    fr, la = v["vocab"]["frontal"], v["vocab"]["lateral"]
    kmax = max(len(set(fr.get(c, [])) | set(la.get(c, []))) for c in set(fr) | set(la))
    assert kmax == 13


def test_landmark_table() -> None:
    rows = _csv("landmarks.csv")
    assert len(rows) == 132
    assert sum(int(r["n_pairs"]) for r in rows) == 311
    groups = _csv("groups.csv")
    assert len(groups) == 27  # clinical groups of the rating taxonomy (25 still covered after review)
    assert sum(int(g["n_definitions_final"]) > 0 for g in groups) == 25
    assert {r["group"] for r in rows} <= {g["group"] for g in groups}


def test_categories_cover_paxray() -> None:
    names = [n for _, n in KS.CATEGORIES]
    assert len(names) == 177
    assert set(KS.SCHEMA_FRONTAL) == set(KS.SCHEMA_LATERAL) == set(names)


def _ellipse(H: int, W: int, cy: int, cx: int, ry: int, rx: int) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def test_centroid_rule_on_synthetic_mask() -> None:
    H = W = 256
    mask = E.clean_mask(_ellipse(H, W, 150, 100, 30, 50))
    masks = {"liver": mask}
    ctx = E.make_ctx(masks, W, H)
    kp = E.extract_for_class("liver", mask, KS.SCHEMA_FRONTAL, ctx, masks)
    assert set(kp) == {"center"}
    x, y, v = kp["center"]
    assert v > 0 and abs(x - 100) <= 1 and abs(y - 150) <= 1


def test_extract_points_inside_image() -> None:
    H = W = 256
    heart = _ellipse(H, W, 160, 140, 40, 55)
    masks = {"heart": heart}
    ctx = E.make_ctx(masks, W, H)
    kp = E.extract_for_class("heart", heart, KS.SCHEMA_FRONTAL, ctx, masks)
    assert list(kp) == KS.SCHEMA_FRONTAL["heart"].names()
    found = [p for p in kp.values() if p[2] > 0]
    assert found, "no heart landmark derived from a plain ellipse"
    for x, y, _ in found:
        assert 0 <= x < W and 0 <= y < H


@pytest.mark.parametrize("view", ["frontal", "lateral"])
def test_skeleton_edges_reference_kept_names(view: str) -> None:
    sch = KS.SCHEMA_FRONTAL if view == "frontal" else KS.SCHEMA_LATERAL
    for s in sch.values():
        names = set(s.names())
        assert all(a in names and b in names for a, b in s.skeleton)
