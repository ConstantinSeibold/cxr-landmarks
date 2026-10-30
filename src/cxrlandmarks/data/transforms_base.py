"""Minimal vendored transform framework (replaces the fvcore / detectron2 classes the
paper's data pipeline used).

Ported verbatim in behaviour from fvcore 0.1.5 ``fvcore.transforms.transform`` and
detectron2 0.6 ``detectron2.data.transforms`` (Apache-2.0):

  Transform, TransformList, NoOpTransform, HFlipTransform   (fvcore)
  ResizeTransform, RotationTransform                         (detectron2 transform.py)
  Augmentation, AugmentationList, AugInput, ResizeShortestEdge (detectron2 augmentation*.py)

Only the methods the pipeline calls are kept. RNG consumption is identical to the
originals (``ResizeShortestEdge`` draws ``np.random.choice`` even for a single size),
which the equivalence test relies on.
"""
from __future__ import annotations

import inspect
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image


class Transform:
    """Deterministic transform of image / coords / segmentation / boxes / polygons."""

    def _set_attributes(self, params: dict[str, Any] | None = None) -> None:
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        # corners -> apply_coords -> axis-aligned min/max (fvcore default).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        return np.concatenate((minxy, maxxy), axis=1)

    def apply_polygons(self, polygons: list) -> list:
        return [self.apply_coords(p) for p in polygons]


class TransformList(Transform):
    """Sequence of transforms (flattened); each ``apply_*`` chains through all of them."""

    def __init__(self, transforms: list[Transform]):
        flat: list[Transform] = []
        for t in transforms:
            assert isinstance(t, Transform), f"TransformList requires Transforms, got {type(t)}"
            if isinstance(t, TransformList):
                flat.extend(t.transforms)
            else:
                flat.append(t)
        self.transforms = flat

    def _apply(self, x, meth: str):
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def apply_image(self, x):
        return self._apply(x, "apply_image")

    def apply_coords(self, x):
        return self._apply(x, "apply_coords")

    def apply_segmentation(self, x):
        return self._apply(x, "apply_segmentation")

    def apply_box(self, x):
        return self._apply(x, "apply_box")

    def apply_polygons(self, x):
        return self._apply(x, "apply_polygons")

    def __len__(self) -> int:
        return len(self.transforms)


class NoOpTransform(Transform):
    def apply_image(self, img):
        return img

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_box(self, box):
        return box

    def apply_polygons(self, polygons):
        return polygons


class HFlipTransform(Transform):
    """Horizontal flip; coords flip as ``W - x`` (continuous coordinates)."""

    def __init__(self, width: int):
        self.width = width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if img.ndim <= 3:
            return np.flip(img, axis=1)
        return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] = self.width - coords[:, 0]
        return coords


class ResizeTransform(Transform):
    """PIL resize (bilinear for images, nearest for segmentations); uint8 only here."""

    def __init__(self, h, w, new_h, new_w, interp=None):
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert img.dtype == np.uint8, "only uint8 images/masks occur in this pipeline"
        interp_method = interp if interp is not None else self.interp
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation, interp=Image.NEAREST)


class RotationTransform(Transform):
    """Rotation about the image centre (cv2), counter-clockwise ``angle`` degrees."""

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h
        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # https://github.com/opencv/opencv/issues/11784 (pixel-centre offset for images)
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation, interp=cv2.INTER_NEAREST)

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm


# --------------------------------------------------------------------------- augmentations
class AugInput:
    """Holds the image being augmented; ``transform`` applies a Transform in place."""

    def __init__(self, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        assert image.ndim in (2, 3)
        self.image = image

    def transform(self, tfm: Transform) -> None:
        self.image = tfm.apply_image(self.image)


class Augmentation:
    """Random policy -> deterministic Transform. ``__call__`` samples the transform from the
    CURRENT image and applies it to ``aug_input`` immediately (so augs compose sequentially
    and any RNG drawn inside ``apply_image`` is consumed right after ``get_transform``)."""

    input_args: tuple[str, ...] | None = None

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def get_transform(self, *args) -> Transform:
        raise NotImplementedError

    def __call__(self, aug_input: AugInput) -> Transform:
        if self.input_args is None:
            prms = list(inspect.signature(self.get_transform).parameters)
            self.input_args = ("image",) if len(prms) == 1 else tuple(prms)
        args = [getattr(aug_input, f) for f in self.input_args]
        tfm = self.get_transform(*args)
        assert isinstance(tfm, Transform), f"{type(self)}.get_transform must return a Transform"
        aug_input.transform(tfm)
        return tfm


class AugmentationList(Augmentation):
    def __init__(self, augs: list[Augmentation]):
        self.augs = list(augs)

    def __call__(self, aug_input: AugInput) -> TransformList:
        return TransformList([x(aug_input) for x in self.augs])


class ResizeShortestEdge(Augmentation):
    """Resize the short edge to ``short_edge_length`` (capped by ``max_size`` on the long edge).

    ``sample_style="choice"`` draws ``np.random.choice`` over the (duplicated) size tuple even
    for a single size -- one numpy RNG draw per image, kept for RNG-stream equivalence.
    """

    def __init__(self, short_edge_length, max_size=sys.maxsize, sample_style="range",
                 interp=Image.BILINEAR):
        assert sample_style in ("range", "choice"), sample_style
        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()
        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int) -> tuple[int, int]:
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        return (int(newh + 0.5), int(neww + 0.5))
