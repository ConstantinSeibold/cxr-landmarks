import os
import sys

import pytest
import torch

from cxrlandmarks.decode import decode_dark
from cxrlandmarks.landmarks import Channels, view_of
from cxrlandmarks.model import JointDenseModel, pad_batch

KP_PREP_BASELINES = os.path.join(os.environ.get("KP_PREP_ROOT", "kp_prep"), "baselines")   # paper code (not public)


def test_channels():
    ch = Channels()
    assert ch.n == 311 and ch.n_classes == 177 and ch.k_max == 13
    assert int(ch.view_mask["frontal"].sum()) == 203 and int(ch.view_mask["lateral"].sum()) == 108
    for v in ("frontal", "lateral"):
        assert sorted(ch.table[v].values()) == sorted(int(i) for i in torch.nonzero(ch.view_mask[v]).flatten())


def test_view_of():
    assert view_of("images/RSNAPE_x_lateral.png") == "lateral"
    assert view_of("images/RSNAPE_x_frontal.png") == "frontal"


def test_decode_dark_subpixel():
    h, w = 64, 48
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    centers = torch.tensor([[10.3, 20.7], [30.0, 5.45], [1.2, 60.8]])            # (x, y), one near the edge
    hm = torch.stack([torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 2.0 ** 2)) for cx, cy in centers])
    xy, peak = decode_dark(hm)
    assert torch.allclose(xy, centers, atol=0.1), xy
    assert (peak > 0.8).all()


def test_forward_shapes():
    ch = Channels()
    m = JointDenseModel(ch.n, ch.n_classes, pretrained_encoder=False).eval()
    x = pad_batch([torch.zeros(3, 100, 70, dtype=torch.uint8)])
    assert x.shape == (1, 3, 128, 96)
    with torch.no_grad():
        hm, sl = m(x)
    assert hm.shape == (1, 311, 32, 24) and sl.shape == (1, 177, 32, 24)


@pytest.mark.skipif(not os.path.isdir(KP_PREP_BASELINES), reason="paper training code not available")
def test_param_groups_match_paper():
    sys.path.insert(0, KP_PREP_BASELINES)
    import flat_heatmap as FH                                                 # noqa: E402
    ch = Channels()
    ours = JointDenseModel(ch.n, ch.n_classes, pretrained_encoder=False)
    ref = FH.FlatModel("raddino", ch.n, "wide")

    def assign(model, groups):
        ids = {id(p): n for n, p in model.named_parameters()}
        return {ids[id(p)]: (g["lr_scale"], g["weight_decay"]) for g in groups for p in g["params"]}

    a = assign(ours, ours.param_groups(5e-4, 0.1, 0.75))
    b = assign(ref, ref.param_groups(5e-4, 0.1, 0.75))
    shared = {k: v for k, v in a.items() if not k.startswith("seg.")}
    assert shared == b
