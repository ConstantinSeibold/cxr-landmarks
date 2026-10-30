"""Augmentation ops of the paper's training pipeline (ported from qseg
``qseg/data/augmentations.py`` @ the paper commit; only the ops active in the paper
config are kept: lateral-gated flip, the RandAugment wrapper, and the 16 pool members
rotation, shear, perspective, aspect-resize, gaussian noise, salt&pepper, gaussian blur,
motion blur, gamma, CLAHE, poisson, intensity remap, fourier amplitude, elastic,
inversion, random erasing).

The bodies are kept line-for-line identical to qseg so that, under identical python
``random`` / ``numpy`` RNG state, every op consumes the same random numbers in the same
order and produces bit-identical outputs (proved by ``tests/test_data_equivalence.py``).
Every pool member is constructed with ``prob=1.0`` under RandAugment, but the
``np.random.random() > prob`` coin is still drawn (one numpy draw) -- keep it.
"""
from __future__ import annotations

import random

import numpy as np

from .transforms_base import (
    Augmentation, HFlipTransform, NoOpTransform, ResizeTransform, RotationTransform,
    Transform, TransformList,
)

class LateralGatedFlip(Augmentation):
    """Horizontal flip applied ONLY to lateral CXRs.

    The mapper sets ``.active`` per image (from the file_name) before running the
    aug list. Laterals are left/right mirror acquisitions with intrinsic anatomical
    labels (e.g. ``heart atrium left`` stays the left atrium under a mirror), so a
    plain horizontal flip yields a valid in-distribution sample — no left/right
    class or keypoint-slot swap is needed; masks/boxes/keypoints follow via
    ``apply_coords``. Frontals carry real L/R asymmetry (cardiac silhouette, aortic
    arch, gastric bubble) and must never be flipped — the mapper leaves ``.active``
    False for them, so this returns a NoOp.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = float(prob)
        self.active = False

    def get_transform(self, image):
        if self.active and random.random() < self.prob:
            return HFlipTransform(image.shape[1])
        return NoOpTransform()


class RandAugment(Augmentation):
    """Pick between `num_ops_min` and `num_ops_max` augmentations at random from
    `pool` (uniform over k, then `random.sample(pool, k)`) and apply in sequence.

    The variable-k draw means different images in a batch see different aug
    *counts* as well as different *subsets* — the original RandAugment idea
    pushed further. Set `num_ops_min == num_ops_max` to recover the original
    fixed-k behavior.

    Each picked aug's strength is uniformly sampled inside its own configured range
    (its `__call__` already does this). `magnitude` in [0, 1] scales each aug's
    max-strength parameter before sampling — magnitude=1.0 keeps the configured
    range; magnitude=0.5 halves it; magnitude=0 effectively disables that op.

    If `magnitude_min < magnitude`, the magnitude is itself sampled per image,
    uniformly in `[magnitude_min, magnitude]` (TrivialAugment-style): each image
    gets its own strength scale, applied to all of its selected augs. With
    `magnitude_min == magnitude` (default) the behavior is the original fixed scale.

    Augs in the pool are applied SEQUENTIALLY via d2's `Augmentation.__call__`
    pattern, so each subsequent aug sees the post-previous-transform image.
    """

    def __init__(self, pool, num_ops_min: int, num_ops_max: int, magnitude: float,
                 magnitude_min: float | None = None):
        super().__init__()
        self.pool = list(pool)
        self.num_ops_min = max(0, int(num_ops_min))
        self.num_ops_max = max(self.num_ops_min, int(num_ops_max))
        self.magnitude = float(magnitude)
        self.magnitude_min = float(magnitude_min) if magnitude_min is not None else float(magnitude)

    def __call__(self, aug_input):  # type: ignore[override]
        if not self.pool or self.num_ops_max == 0:
            return TransformList([])
        lo = min(self.num_ops_min, len(self.pool))
        hi = min(self.num_ops_max, len(self.pool))
        k = random.randint(lo, hi) if hi > lo else hi
        if k == 0:
            return TransformList([])
        mag = self.magnitude
        if self.magnitude_min < self.magnitude:
            mag = random.uniform(self.magnitude_min, self.magnitude)
        chosen = random.sample(self.pool, k)
        transforms: list = []
        for aug in chosen:
            scaled = _scale_aug_magnitude(aug, mag)
            t = scaled(aug_input)
            if isinstance(t, TransformList):
                transforms.extend(t.transforms)
            else:
                transforms.append(t)
        return TransformList(transforms)

    def get_transform(self, image):  # pragma: no cover — d2 prefers __call__ path
        return NoOpTransform()


def _scale_aug_magnitude(aug: Augmentation, magnitude: float) -> Augmentation:
    """Return a copy of `aug` with its max-strength parameter scaled by `magnitude`.

    For augs whose strength is controlled by a single scalar attribute we scale
    that scalar; for augs that don't have a recognised strength attribute we
    return the original aug unchanged.
    """
    if magnitude >= 1.0:
        return aug
    for attr in ("max_deg", "max_shear_deg", "scale", "std_max",
                 "density_max", "sigma_max", "max_kernel",
                 "strength_max", "clip_limit_max", "alpha_max",
                 "area_ratio_max"):
        if hasattr(aug, attr):
            current = getattr(aug, attr)
            scaled = type(current)(current * magnitude) if not isinstance(current, int) \
                else max(0, int(round(current * magnitude)))
            new = aug.__class__.__new__(aug.__class__)
            new.__dict__.update(aug.__dict__)
            setattr(new, attr, scaled)
            return new
    return aug


class _AffineTransform(Transform):
    """Apply a 2x3 affine matrix `M` to image, masks, segmentation, and coords.

    `M` maps source pixels to destination pixels (cv2 convention).
    """

    def __init__(self, M: np.ndarray, height: int, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: int | None = None) -> np.ndarray:
        import cv2
        flags = cv2.INTER_LINEAR if (interp is None or img.dtype != np.uint8) else cv2.INTER_LINEAR
        return cv2.warpAffine(
            img, self.M, (self.width, self.height),
            flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        ones = np.ones((coords.shape[0], 1), dtype=np.float64)
        homog = np.concatenate([coords, ones], axis=1)        # (N, 3)
        out = homog @ self.M.T                                # (N, 2)
        return out.astype(coords.dtype)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.warpAffine(
            segmentation, self.M, (self.width, self.height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )


class _PerspectiveTransform(Transform):
    """Apply a 3x3 projective matrix `M` to image, masks, segmentation, and coords."""

    def __init__(self, M: np.ndarray, height: int, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: int | None = None) -> np.ndarray:
        import cv2
        return cv2.warpPerspective(
            img, self.M, (self.width, self.height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        ones = np.ones((coords.shape[0], 1), dtype=np.float64)
        homog = np.concatenate([coords, ones], axis=1)        # (N, 3)
        out = homog @ self.M.T                                # (N, 3)
        out = out[:, :2] / np.clip(out[:, 2:3], 1e-6, None)
        return out.astype(coords.dtype)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.warpPerspective(
            segmentation, self.M, (self.width, self.height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )


class RandomRotationQSeg(Augmentation):
    """Symmetric random rotation in degrees. `max_deg=0` is a no-op."""

    def __init__(self, max_deg: float):
        super().__init__()
        self.max_deg = float(max_deg)

    def get_transform(self, image):
        if self.max_deg <= 0:
            return NoOpTransform()
        angle = float(np.random.uniform(-self.max_deg, self.max_deg))
        h, w = image.shape[:2]
        return RotationTransform(h, w, angle, expand=False, center=None)


class RandomAspectResize(Augmentation):
    """Slight ANISOTROPIC scale jitter: scale height and width by INDEPENDENT factors
    sampled in ``[1 - strength_max, 1 + strength_max]``.

    Unlike ``ResizeShortestEdge`` (isotropic, aspect-ratio-preserving) this mildly
    stretches/squashes the aspect ratio, training some invariance to per-axis scale.
    Image + masks + boxes + keypoints are transformed consistently by d2's
    ``ResizeTransform`` (its ``apply_coords`` scales x and y independently). The
    resulting image is slightly non-square; the model's size-divisibility padding
    absorbs it.

    Keep ``strength_max`` small for anatomy — large values distort true proportions
    (cardiothoracic ratio, rib spacing). ``strength_max`` is the per-axis bound and is
    scaled by RandAugment's magnitude knob (via ``_scale_aug_magnitude``).
    """

    def __init__(self, strength_max: float = 0.1, prob: float = 0.5):
        super().__init__()
        self.strength_max = float(strength_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.strength_max <= 0 or random.random() >= self.prob:
            return NoOpTransform()
        h, w = image.shape[:2]
        s = self.strength_max
        sy = float(np.random.uniform(1.0 - s, 1.0 + s))
        sx = float(np.random.uniform(1.0 - s, 1.0 + s))
        new_h = max(1, int(round(h * sy)))
        new_w = max(1, int(round(w * sx)))
        if new_h == h and new_w == w:
            return NoOpTransform()
        return ResizeTransform(h, w, new_h, new_w)


class RandomShear(Augmentation):
    """Random shear in degrees, applied as a 2x3 affine matrix about the image
    centre.

    Shear is applied along **one axis per call** (50/50 x vs y), not both
    simultaneously. Applying both x and y shear with the same sign yields a
    matrix that visually mimics rotation (e.g., 30° both ways → ``[[1, 0.577],
    [0.577, 1]]``, det 0.67), losing the parallelogram signature that makes
    shear distinctive. Single-axis shear produces an unmistakable parallelogram
    while remaining different from rotation.

    ``max_shear_deg`` is the per-axis range bound; the actual angle is sampled
    uniformly in [-max, max].
    """

    def __init__(self, max_shear_deg: float):
        super().__init__()
        self.max_shear_deg = float(max_shear_deg)

    def get_transform(self, image):
        if self.max_shear_deg <= 0:
            return NoOpTransform()
        h, w = image.shape[:2]
        deg = float(np.random.uniform(-self.max_shear_deg, self.max_shear_deg))
        tan = np.tan(np.deg2rad(deg))
        cx, cy = w / 2.0, h / 2.0
        if np.random.random() < 0.5:
            # X-shear: each row slides horizontally by tan * (y - cy).
            M = np.array(
                [[1.0, tan, -tan * cy],
                 [0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
        else:
            # Y-shear: each column slides vertically by tan * (x - cx).
            M = np.array(
                [[1.0, 0.0, 0.0],
                 [tan, 1.0, -tan * cx]],
                dtype=np.float64,
            )
        return _AffineTransform(M, h, w)


class RandomPerspective(Augmentation):
    """Random projective warp: jitter each image corner by `scale * (W, H)` pixels."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def get_transform(self, image):
        if self.scale <= 0:
            return NoOpTransform()
        import cv2
        h, w = image.shape[:2]
        jx = self.scale * w
        jy = self.scale * h
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.random.uniform(-1.0, 1.0, size=src.shape).astype(np.float32) * np.float32([jx, jy])
        M = cv2.getPerspectiveTransform(src, dst)
        return _PerspectiveTransform(M, h, w)


class _ImageOnlyTransform(Transform):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def apply_image(self, img, interp=None):
        return self.fn(img)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation


class RandomGaussianNoise(Augmentation):
    """Additive Gaussian noise. `std` is in pixel units (assumes uint8 image)."""

    def __init__(self, std_max: float, prob: float = 0.5):
        super().__init__()
        self.std_max = float(std_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.std_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        std = float(np.random.uniform(0.0, self.std_max))

        def _apply(img):
            noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
            out = img.astype(np.float32) + noise
            return np.clip(out, 0, 255).astype(img.dtype)

        return _ImageOnlyTransform(_apply)


class RandomSaltPepperNoise(Augmentation):
    """Salt-and-pepper noise. `density` = fraction of pixels affected (split 50/50)."""

    def __init__(self, density_max: float, prob: float = 0.3):
        super().__init__()
        self.density_max = float(density_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.density_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        density = float(np.random.uniform(0.0, self.density_max))

        def _apply(img):
            out = img.copy()
            h, w = out.shape[:2]
            n = int(density * h * w)
            if n <= 0:
                return out
            # salt
            ys = np.random.randint(0, h, size=n)
            xs = np.random.randint(0, w, size=n)
            out[ys, xs] = 255
            # pepper
            ys = np.random.randint(0, h, size=n)
            xs = np.random.randint(0, w, size=n)
            out[ys, xs] = 0
            return out

        return _ImageOnlyTransform(_apply)


class RandomGaussianBlur(Augmentation):
    """Gaussian blur with a random sigma in [0, sigma_max]."""

    def __init__(self, sigma_max: float, prob: float = 0.3):
        super().__init__()
        self.sigma_max = float(sigma_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.sigma_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        sigma = float(np.random.uniform(0.1, self.sigma_max))

        def _apply(img):
            import cv2
            return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

        return _ImageOnlyTransform(_apply)


class RandomMotionBlur(Augmentation):
    """Linear motion blur with a random kernel size and angle."""

    def __init__(self, max_kernel: int, prob: float = 0.2):
        super().__init__()
        self.max_kernel = int(max_kernel)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.max_kernel < 3 or np.random.random() > self.prob:
            return NoOpTransform()
        # Force odd kernel size between 3 and max_kernel.
        k = int(np.random.randint(3, self.max_kernel + 1))
        if k % 2 == 0:
            k += 1
        angle = float(np.random.uniform(0, 180))

        def _apply(img):
            import cv2
            kern = np.zeros((k, k), dtype=np.float32)
            kern[k // 2, :] = 1.0 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
            kern = cv2.warpAffine(kern, M, (k, k))
            kern = kern / max(kern.sum(), 1e-6)
            return cv2.filter2D(img, -1, kern)

        return _ImageOnlyTransform(_apply)


class RandomGammaJitter(Augmentation):
    """Random gamma correction. img' = (img/255) ^ γ * 255.

    γ is sampled log-uniform in [1/(1+max), 1+max] so the brightening and
    darkening directions are symmetric on the log scale. Targets display-
    window / brightness mismatch between synthetic and real X-rays.
    """

    def __init__(self, strength_max: float = 0.5, prob: float = 0.5):
        super().__init__()
        self.strength_max = float(strength_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.strength_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        lg = np.log(1.0 + self.strength_max)
        gamma = float(np.exp(np.random.uniform(-lg, lg)))

        def _apply(img):
            x = img.astype(np.float32) / 255.0
            x = np.power(np.clip(x, 0.0, 1.0), gamma)
            return (x * 255.0).clip(0, 255).astype(img.dtype)

        return _ImageOnlyTransform(_apply)


class RandomCLAHE(Augmentation):
    """Contrast-Limited Adaptive Histogram Equalization with random clip-limit
    and random tile size. Trains the model to be invariant to contrast-
    normalization choices (which differ wildly between MIMIC stations).
    """

    def __init__(self, clip_limit_max: float = 4.0, prob: float = 0.3):
        super().__init__()
        self.clip_limit_max = float(clip_limit_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.clip_limit_max <= 1.0 or np.random.random() > self.prob:
            return NoOpTransform()
        clip = float(np.random.uniform(1.0, self.clip_limit_max))
        grid = int(np.random.choice([4, 8, 16]))

        def _apply(img):
            import cv2
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            if img.ndim == 2:
                return clahe.apply(img)
            # Apply CLAHE on the L channel of LAB so colour is preserved.
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return _ImageOnlyTransform(_apply)


class RandomPoissonNoise(Augmentation):
    """Photon-count Poisson noise simulating X-ray shot noise.

    The image is rescaled to a "photon count" intensity (controlled by
    ``strength_max``, where higher = noisier), Poisson-sampled, and rescaled
    back to 0-255. Physics-grounded: real X-rays carry Poisson noise from
    the photon-counting detector, and the synthetic synthfb composites don't.
    """

    def __init__(self, strength_max: float = 0.7, prob: float = 0.3):
        super().__init__()
        # strength_max in [0, 1]: 0 = clean, 1 = scale_min ≈ 10 photons/pixel.
        self.strength_max = float(strength_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.strength_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        s = float(np.random.uniform(0.0, self.strength_max))
        # Map strength → photon-count scale. s=0 → 256 (no noise);
        # s=1 → 10 (very noisy).
        scale = 256.0 * (1.0 - s) + 10.0 * s

        def _apply(img):
            x = (img.astype(np.float32) / 255.0) * scale
            x = np.random.poisson(np.clip(x, 0.0, None)).astype(np.float32)
            x = (x / scale) * 255.0
            return np.clip(x, 0, 255).astype(img.dtype)

        return _ImageOnlyTransform(_apply)


class RandomIntensityRemap(Augmentation):
    """Random smooth monotonic intensity remap.

    Builds a monotonic piecewise-cubic curve through ``n`` control points
    near the identity, with per-point jitter scaled by ``strength_max``,
    then applies it as a LUT. More expressive than gamma alone — covers
    arbitrary smooth non-linear remappings that differ between detector
    types. Endpoints (0 and 255) are pinned.
    """

    def __init__(self, strength_max: float = 0.15, prob: float = 0.3):
        super().__init__()
        self.strength_max = float(strength_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.strength_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        s = float(np.random.uniform(0.0, self.strength_max))
        n = 7
        xs = np.linspace(0.0, 1.0, n)
        jitter = np.random.uniform(-s, s, n)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        ys = np.clip(xs + jitter, 0.0, 1.0)
        # Force strictly monotonic with a tiny epsilon.
        for i in range(1, n):
            if ys[i] <= ys[i - 1]:
                ys[i] = ys[i - 1] + 1e-4
        ys[-1] = 1.0
        lut_x = np.linspace(0.0, 1.0, 256)
        lut_y = np.interp(lut_x, xs, ys)
        lut = (lut_y * 255.0).clip(0, 255).astype(np.uint8)

        def _apply(img):
            import cv2
            return cv2.LUT(img, lut)

        return _ImageOnlyTransform(_apply)


class _ElasticTransform(Transform):
    """Pixel-accurate elastic deformation via smooth random displacement fields.

    Convention: ``dx, dy`` are **push displacements** in pixels — a point
    originally at ``(x, y)`` ends up at ``(x + dx[y, x], y + dy[y, x])`` in the
    output. For the image we call cv2.remap in pull mode, which is the inverse:
    ``output[v, u] = input[v - dy[v, u], u - dx[v, u]]``. For small smooth
    displacements (a few pixels with σ≈40) the inverse-vs-forward distinction
    is sub-pixel and the small-displacement approximation suffices for keypoints
    too.
    """

    def __init__(self, dx: np.ndarray, dy: np.ndarray, height: int, width: int):
        super().__init__()
        self._set_attributes(locals())
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        self.map_x = (xs - dx).astype(np.float32)
        self.map_y = (ys - dy).astype(np.float32)

    def apply_image(self, img: np.ndarray, interp: int | None = None) -> np.ndarray:
        import cv2
        return cv2.remap(
            img, self.map_x, self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        # The image is PULLED (output u samples input u - d(u)), so content at p lands at
        # the u solving u = p + d(u). The one-step p + d(p) drifts keypoints off the image
        # content as alpha grows (measured at alpha_max=60: mean 2 px, p95 9 px, max ~50 px),
        # so iterate the fixed point with bilinear sampling (contraction: |grad d| << 1).
        from scipy.ndimage import map_coordinates
        coords = np.asarray(coords, dtype=np.float64)
        if coords.size == 0:
            return coords.copy()
        u = coords.copy()
        for _ in range(20):
            yx = [np.clip(u[:, 1], 0, self.height - 1), np.clip(u[:, 0], 0, self.width - 1)]
            nxt = coords + np.stack([map_coordinates(self.dx, yx, order=1, mode="nearest"),
                                     map_coordinates(self.dy, yx, order=1, mode="nearest")], 1)
            done = np.abs(nxt - u).max() < 1e-3
            u = nxt
            if done:
                break
        return u

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.remap(
            segmentation, self.map_x, self.map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )


class RandomElasticTransform(Augmentation):
    """Smooth elastic deformation.

    Generates two random displacement fields, smooths them with a Gaussian of
    width ``sigma``, normalises them to unit max-absolute and scales by an
    amplitude sampled in ``[0, alpha_max]``. ``alpha_max`` is in pixels — the
    maximum displacement any point can experience.

    Defaults target chest-X-ray-style anatomical deformation: σ=40 px (smooth
    global warp, not noise), α≤12 px (clearly visible bend, not pathological).
    Cost is roughly 5-15 ms per 512² image — about the same as motion blur.
    """

    def __init__(self, alpha_max: float = 12.0, sigma: float = 40.0,
                 prob: float = 0.3):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.sigma = float(sigma)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.alpha_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        h, w = image.shape[:2]
        alpha = float(np.random.uniform(0.0, self.alpha_max))
        if alpha <= 0:
            return NoOpTransform()
        import cv2
        dx = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        dy = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        max_abs = float(max(np.abs(dx).max(), np.abs(dy).max(), 1e-6))
        dx = (dx / max_abs) * alpha
        dy = (dy / max_abs) * alpha
        return _ElasticTransform(dx, dy, h, w)


class RandomInversion(Augmentation):
    """Intensity inversion (photographic negative): ``img' = 255 - img``.

    Clinically valid for CXR — some PACS systems display images inverted
    (bone dark / lungs bright) by default, and some radiologists prefer the
    inverted view for spotting pneumothorax, free air, or subtle pleural
    effusions. Including this in training teaches polarity invariance.

    Binary on/off; no scalable strength parameter. RandAugment's k-of-N
    sampling controls how often it fires.
    """

    def __init__(self, prob: float = 0.3):
        super().__init__()
        self.prob = float(prob)

    def get_transform(self, image):
        if np.random.random() > self.prob:
            return NoOpTransform()

        def _apply(img):
            if img.dtype == np.uint8:
                return (255 - img.astype(np.int16)).clip(0, 255).astype(np.uint8)
            return (255.0 - img.astype(np.float32)).clip(0, 255).astype(img.dtype)

        return _ImageOnlyTransform(_apply)


class _ErasingTransform(Transform):
    """Variant-B erasing: image gets a random fill, mask is zeroed in the
    erased region, and keypoints that land inside the rectangle are pushed
    to (-1, -1) so d2's transform_keypoint_annotations zero-visibility
    logic catches them.

    Bbox is left to the mapper's ``recompute_boxes=True`` path — once the
    mask is zeroed in the erased region, the recomputed bbox shrinks (or
    becomes empty) automatically.
    """

    def __init__(self, top: int, left: int, erase_h: int, erase_w: int,
                 fill_block: np.ndarray, height: int, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: int | None = None) -> np.ndarray:
        out = img.copy()
        out[self.top:self.top + self.erase_h, self.left:self.left + self.erase_w] = self.fill_block
        return out

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        out = segmentation.copy()
        out[self.top:self.top + self.erase_h, self.left:self.left + self.erase_w] = 0
        return out

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64).copy()
        inside = (
            (coords[:, 0] >= self.left)
            & (coords[:, 0] < self.left + self.erase_w)
            & (coords[:, 1] >= self.top)
            & (coords[:, 1] < self.top + self.erase_h)
        )
        # Push erased points out of the image; d2 then zeros their visibility.
        # bbox corners may also be touched here, but our mapper uses
        # ``recompute_boxes=True`` so the bbox is rederived from the modified
        # mask and this apply_coords value for corners is unused.
        coords[inside, 0] = -1
        coords[inside, 1] = -1
        return coords


class RandomErasing(Augmentation):
    """Variant-B random erasing for instance segmentation.

    Picks a rectangle of area in ``[area_ratio_min, area_ratio_max]`` of the
    image, aspect ratio in ``[aspect_ratio_min, aspect_ratio_max]``, and fills
    with random noise / mean / constant. **Mask + keypoint visibility are
    updated**: pixels in the erased region get masked out, keypoints inside
    are pushed out of image (d2 then zeros their visibility), bbox is
    recomputed from the modified mask by the mapper.

    Semantics: "this part of the object is gone from the image" — the model
    is supervised only on the remaining visible pixels. Useful for the
    synthfb→MIMIC gap because it adds random sharp-edge rectangles to the
    training distribution, defusing the ``cutpaste`` feature.
    """

    def __init__(
        self,
        area_ratio_max: float = 0.2,
        area_ratio_min: float = 0.02,
        aspect_ratio_min: float = 0.3,
        aspect_ratio_max: float = 3.3,
        fill: str = "random",          # "random" | "mean" | "constant"
        prob: float = 0.3,
    ):
        super().__init__()
        self.area_ratio_max = float(area_ratio_max)
        self.area_ratio_min = float(area_ratio_min)
        self.aspect_ratio_min = float(aspect_ratio_min)
        self.aspect_ratio_max = float(aspect_ratio_max)
        self.fill = str(fill)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.area_ratio_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        h, w = image.shape[:2]
        for _ in range(20):
            target_area = float(np.random.uniform(
                self.area_ratio_min, self.area_ratio_max,
            )) * (h * w)
            aspect = float(np.random.uniform(
                self.aspect_ratio_min, self.aspect_ratio_max,
            ))
            erase_h = int(round((target_area * aspect) ** 0.5))
            erase_w = int(round((target_area / aspect) ** 0.5))
            if 1 <= erase_h < h and 1 <= erase_w < w:
                top = int(np.random.randint(0, h - erase_h))
                left = int(np.random.randint(0, w - erase_w))
                if self.fill == "random":
                    fill_block = (
                        np.random.randint(0, 256, size=(erase_h, erase_w) + image.shape[2:], dtype=np.uint8)
                        if image.dtype == np.uint8
                        else np.random.uniform(0, 255, size=(erase_h, erase_w) + image.shape[2:]).astype(image.dtype)
                    )
                elif self.fill == "mean":
                    if image.ndim == 3:
                        m = image.mean(axis=(0, 1)).astype(image.dtype)
                        fill_block = np.broadcast_to(m, (erase_h, erase_w, image.shape[2])).copy()
                    else:
                        m = float(image.mean())
                        fill_block = np.full((erase_h, erase_w), m, dtype=image.dtype)
                else:
                    fill_block = np.zeros((erase_h, erase_w) + image.shape[2:], dtype=image.dtype)
                return _ErasingTransform(
                    top=top, left=left, erase_h=erase_h, erase_w=erase_w,
                    fill_block=fill_block, height=h, width=w,
                )
        return NoOpTransform()


class RandomFourierAmplitude(Augmentation):
    """Perturb the high-frequency amplitude spectrum while preserving phase.

    Sim-to-real differences (sharp synthetic paste edges vs smooth real X-ray
    diffusion) live primarily in the high-frequency amplitude. Multiplying the
    high-frequency band by a random per-frequency factor in
    ``[1 - s, 1 + s]`` (mask-weighted by frequency radius) trains the model
    to ignore that band as a class cue. Phase is left untouched so geometry
    is preserved.
    """

    def __init__(self, strength_max: float = 0.3, prob: float = 0.3):
        super().__init__()
        self.strength_max = float(strength_max)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.strength_max <= 0 or np.random.random() > self.prob:
            return NoOpTransform()
        s = float(np.random.uniform(0.0, self.strength_max))

        def _apply(img):
            channels = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[-1])]
            h, w = channels[0].shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.float32)
            r_norm = r / max(r.max(), 1e-6)
            # Sigmoid high-pass mask centred at r_norm = 0.3.
            mask = 1.0 / (1.0 + np.exp(-10.0 * (r_norm - 0.3)))
            outs: list[np.ndarray] = []
            for ch in channels:
                f = np.fft.fftshift(np.fft.fft2(ch.astype(np.float32)))
                amp = np.abs(f)
                phase = np.angle(f)
                pert = 1.0 + s * mask * np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
                f_new = (amp * pert) * np.exp(1j * phase)
                outs.append(np.real(np.fft.ifft2(np.fft.ifftshift(f_new))))
            out = outs[0] if img.ndim == 2 else np.stack(outs, axis=-1)
            return np.clip(out, 0, 255).astype(img.dtype)

        return _ImageOnlyTransform(_apply)
