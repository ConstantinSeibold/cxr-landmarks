"""Train the joint dense model as in the paper.

    cxrlm-train --paxray-root /data/paxraypp --out runs/joint_dense

Recipe (paper defaults): RAD-DINO ViT-B/14 fine-tuned end-to-end, AdamW (lr 5e-4, weight decay 0.1,
layer-wise lr decay 0.75), 500 warm-up iterations then cosine decay, batch 8, 18 epochs, bf16, grad
clip 1.0, EMA 0.9995 (the EMA weights are evaluated and saved). Landmark loss: MSE to Gaussian
heatmaps (sigma 2 heatmap px, x1000) on the channels of the image's view; mask loss: BCE over all class
maps + Dice over the classes present, at stride 2. The checkpoint is selected on validation OKS-AP
(validation after epoch 1 and every epoch from 10 on). Data: the paper's augmentation pipeline and
sample order (`cxrlandmarks.data`), CutMix on the collated batch.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PaperOrderSampler, PaxRayLandmarkDataset, apply_cutmix, collate_list, paper_config, worker_init_fn
from .eval.cli import evaluate
from .landmarks import Channels
from .model import STRIDE, JointDenseModel, pad_batch
from .predict import LandmarkPredictor


def heatmap_targets(batch: list[dict], ch: Channels, hm_hw: tuple[int, int], sigma: float, device):
    """Gaussian targets (B, C, h, w) and per-channel weights (B, C): the image's view channels get
    weight 1 (absent landmarks keep a zero map), the other view's channels get no loss."""
    B, (h, w) = len(batch), hm_hw
    tgt = torch.zeros((B, ch.n, h, w), device=device)
    wt = torch.zeros((B, ch.n), device=device)
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    for b, d in enumerate(batch):
        v = d["view"]
        wt[b] = ch.view_mask[v].to(device).float()
        kp = d["keypoints"]                                              # (N, K, 3) input px
        if kp.numel() == 0:
            continue
        lut = ch.lut[v][d["classes"]][:, :kp.shape[1]]                   # (N, K) channel or -1
        ok = (lut >= 0) & (kp[..., 2] > 0)
        if not ok.any():
            continue
        idx = lut[ok].to(device)
        p = kp[..., :2][ok].to(device)
        u = (p[:, 0] - (STRIDE - 1) / 2) / STRIDE
        q = (p[:, 1] - (STRIDE - 1) / 2) / STRIDE
        g = torch.exp(-((xs[None] - u.view(-1, 1, 1)) ** 2 + (ys[None] - q.view(-1, 1, 1)) ** 2) / (2 * sigma ** 2))
        tgt[b].index_reduce_(0, idx, g, "amax")
    return tgt, wt


def heatmap_loss(pred, tgt, wt):
    """mmpose KeypointMSELoss(use_target_weight): mean over pixels, then over weighted channels."""
    per = ((pred.float() - tgt) ** 2).mean(dim=(2, 3))
    return (per * wt).sum() / wt.sum().clamp(min=1.0)


def mask_targets(batch: list[dict], n_cls: int, hw: tuple[int, int], device):
    """(B, C, h, w) binary class maps at stride 2 and (B, C) presence."""
    B = len(batch)
    tgt = torch.zeros((B, n_cls) + tuple(hw), device=device)
    pres = torch.zeros((B, n_cls), device=device)
    for b, d in enumerate(batch):
        m = d["masks"]
        if m.shape[0] == 0:
            continue
        m = m.to(device).float()
        Hp, Wp = hw[0] * 2, hw[1] * 2
        m = F.pad(m, (0, Wp - m.shape[-1], 0, Hp - m.shape[-2]))
        m = (F.interpolate(m[None], size=tuple(hw), mode="area")[0] > 0.5).float()
        c = d["classes"].to(device)
        tgt[b].index_reduce_(0, c, m, "amax")
        pres[b, c] = 1.0
    return tgt, pres


def mask_loss(logits, tgt, pres):
    """BCE over all class maps + Dice over the classes present in the image."""
    logits = F.interpolate(logits.float(), size=tgt.shape[-2:], mode="bilinear", align_corners=False)
    bce = F.binary_cross_entropy_with_logits(logits, tgt)
    p = logits.sigmoid().flatten(2)
    t = tgt.flatten(2)
    dice = 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)
    return bce, (dice * pres).sum() / pres.sum().clamp(min=1)


def run_eval(ema: JointDenseModel, ch: Channels, gt_json: str, image_root: str, out_dir: Path, batch: int = 8,
             limit: int = 0) -> dict:
    pred = LandmarkPredictor(ema, ch, next(ema.parameters()).device)
    ims = sorted(json.load(open(gt_json))["images"], key=lambda im: im["id"])
    if limit:                                              # smoke tests only: metrics over a subset
        ims = ims[:limit]
        gt = json.load(open(gt_json))
        keep = {im["id"] for im in ims}
        gt["images"] = ims
        gt["annotations"] = [x for x in gt["annotations"] if x["image_id"] in keep]
        out_dir.mkdir(parents=True, exist_ok=True)
        gt_json = str(out_dir / "gt_subset.json")
        json.dump(gt, open(gt_json, "w"))
    kps, segs = [], []
    for s in range(0, len(ims), batch):
        chunk = ims[s:s + batch]
        views = ["lateral" if "lateral" in Path(im["file_name"]).name.lower() else "frontal" for im in chunk]
        for im, r in zip(chunk, pred.predict_batch([str(Path(image_root) / im["file_name"]) for im in chunk], views)):
            k, g = pred.coco_results(r, im["id"])
            kps += k
            segs += g
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(kps, open(out_dir / "keypoints.json", "w"))
    json.dump(segs, open(out_dir / "segm.json", "w"))
    m = evaluate(gt_json, kps, segs)
    json.dump(m, open(out_dir / "metrics.json", "w"), indent=1)
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paxray-root", required=True, help="PAXRay++ root with annotations/{train,val,test}_kp_v1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--eval-from", type=int, default=10, help="validate every epoch from this one (plus epoch 1)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--sigma", type=float, default=2.0, help="heatmap target std (heatmap px)")
    ap.add_argument("--loss-scale", type=float, default=1000.0)
    ap.add_argument("--mask-weight", type=float, default=1.0, help="0 = landmark-only model")
    ap.add_argument("--landmark-weight", type=float, default=1.0, help="0 = segmentation-only model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test: stop after this many iterations")
    ap.add_argument("--eval-limit", type=int, default=0, help="smoke test: evaluate on the first N images only")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = paper_config(a.paxray_root)
    cfg.batch_size, cfg.num_workers, cfg.seed = a.batch, a.workers, a.seed
    ds = PaxRayLandmarkDataset(cfg, "train")
    ch = Channels()
    assert ds.categories.k_max == ch.k_max
    device = torch.device(a.device)
    amp = device.type == "cuda"
    model = JointDenseModel(ch.n, ch.n_classes).to(device)
    model.train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.wd, a.layer_decay), betas=(0.9, 0.999))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch, sampler=PaperOrderSampler(len(ds), seed=a.seed, batch_size=a.batch,
                                                          num_workers=a.workers),
        num_workers=a.workers, collate_fn=collate_list, worker_init_fn=worker_init_fn, persistent_workers=a.workers > 0)
    n_images = len(json.load(open(cfg.json_train))["images"])
    iters_per_epoch = max(1, n_images // a.batch)
    max_iter = iters_per_epoch * a.epochs

    def lr_at(it: int) -> float:
        if it < a.warmup:
            return 0.001 + (1 - 0.001) * it / a.warmup
        return 0.5 * (1 + math.cos(math.pi * (it - a.warmup) / max(1, max_iter - a.warmup)))

    (out / "args.json").write_text(json.dumps({**vars(a), "iters_per_epoch": iters_per_epoch,
                                               "data": cfg.to_dict()}, indent=2))
    log = open(out / "metrics.jsonl", "a")
    best, best_ep, it, t0 = -1.0, -1, 0, time.time()
    data_iter = iter(loader)
    for ep in range(a.epochs):
        for _ in range(iters_per_epoch):
            batch = next(data_iter)
            if cfg.aug.cutmix:
                apply_cutmix(batch, enable=True, prob=cfg.aug.cutmix_prob, alpha=cfg.aug.cutmix_alpha)
            for g in opt.param_groups:
                g["lr"] = a.lr * g["lr_scale"] * lr_at(it)
            x = pad_batch([d["image"] for d in batch]).to(device, non_blocking=True)
            tgt, wt = heatmap_targets(batch, ch, (x.shape[2] // STRIDE, x.shape[3] // STRIDE), a.sigma, device)
            stgt, pres = mask_targets(batch, ch.n_classes, (x.shape[2] // 2, x.shape[3] // 2), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                hm, sl = model(x)
            bce, dice = mask_loss(sl, stgt, pres)
            losses = {"loss_landmark": heatmap_loss(hm, tgt, wt) * a.loss_scale * a.landmark_weight,
                      "loss_mask_bce": bce * a.mask_weight, "loss_mask_dice": dice * a.mask_weight}
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at iteration {it}: { {k: float(v) for k, v in losses.items()} }")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.grad_clip)
            opt.step()
            with torch.no_grad():
                dec = min(a.ema, (1 + it) / (10 + it))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(dec).add_(pm.detach(), alpha=1 - dec)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            it += 1
            if it % 20 == 0:
                el = time.time() - t0
                ls = " ".join(f"{k} {float(v):.4f}" for k, v in losses.items())
                print(f"ep {ep} it {it}/{max_iter} {ls} gn {gn.item():.3f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{el / it:.3f}s/it eta {el / it * (max_iter - it) / 3600:.2f}h", flush=True)
                log.write(json.dumps({"iter": it, **{k: float(v) for k, v in losses.items()}}) + "\n")
            if a.max_iters and it >= a.max_iters:
                break
        torch.save({"model": ema.state_dict(), "epoch": ep + 1}, out / "model_final.pth")
        if (ep + 1) >= a.eval_from or ep == 0 or (ep + 1) == a.epochs:
            m = run_eval(ema, ch, cfg.json_val, cfg.image_root, out / "val_epoch", limit=a.eval_limit)
            print(f"epoch {ep + 1} val OKS-AP {m['oks_ap']:.2f} PCK {m['pck']:.2f} mask AP {m['segm_ap']:.2f}", flush=True)
            log.write(json.dumps({"epoch": ep + 1, "iter": it, **{f"val/{k}": v for k, v in m.items()}}) + "\n")
            log.flush()
            if best_ep < 0 or m["oks_ap"] > best:
                best, best_ep = m["oks_ap"], ep + 1
                torch.save({"model": ema.state_dict(), "epoch": ep + 1}, out / "model_best.pth")
        if a.max_iters and it >= a.max_iters:
            break

    ema.load_state_dict(torch.load(out / "model_best.pth", map_location=device)["model"])
    summary = {"best_epoch": best_ep, "val/oks_ap": best, "train_hours": (time.time() - t0) / 3600}
    for split, gt in (("val", cfg.json_val), ("test", cfg.json_test)):
        m = run_eval(ema, ch, gt, cfg.image_root, out / split, limit=a.eval_limit)
        summary.update({f"{split}/{k}": v for k, v in m.items()})
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("done", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
