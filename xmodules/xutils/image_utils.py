#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import math
from typing import Literal
from typing import Sequence, Union, List, Optional, Tuple, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from .lib_utils import import_available
from . import array_utils as xp

skt = None
mark_boundaries = None
if HAS_SKIMAGE := import_available('skimage'):
    import skimage.transform as skt
    from skimage.segmentation import mark_boundaries

cp = None
cpt = None
if HAS_CUPY := import_available('cupy'):
    import cupy as cp
    import cupy.typing as cpt

cu_skt = None
if HAS_CUCIM := import_available('cucim'):
    import cucim.skimage.transform as cu_skt
    from cucim.skimage.segmentation import mark_boundaries as cu_mark_boundaries

NPGeneric: TypeAlias = xp.NPGeneric
_NPDType = TypeVar('_NPDType', bound=NPGeneric)
TypeArrayLike: TypeAlias = xp.TypeArrayLike[_NPDType]

class ImageUtils:
    @classmethod
    def resize[T: NPGeneric](
            cls,
            image: TypeArrayLike[T],
            output_shape: Tuple[int, int],
            order: int,
            mode='reflect',
            cval=0,
            clip=True,
            preserve_range=True,
            anti_aliasing=True,
            anti_aliasing_sigma: Optional[
                Union[float, Tuple[float, float]]
            ] = None,
            batched=False,
    ) -> TypeArrayLike[T]:
        if batched:
            return cls.batch_resize(
                batch_image=image,
                output_shape=output_shape,
                order=order,
                mode=mode,
                cval=cval,
                clip=clip,
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma,
            )
        else:
            return cls.single_resize(
                image=image,
                output_shape=output_shape,
                order=order,
                mode=mode,
                cval=cval,
                clip=clip,
            )


    @staticmethod
    def single_resize[T: NPGeneric](
            image: TypeArrayLike[T],
            output_shape: Tuple[int, int],
            order: int,
            mode='reflect',
            cval=0,
            clip=True,
            preserve_range=True,
            anti_aliasing=True,
            anti_aliasing_sigma: Optional[
                Union[float, Tuple[float, float]]
            ] = None
    ) -> TypeArrayLike[T]:
        if isinstance(image, np.ndarray):
            if not HAS_SKIMAGE:
                raise RuntimeError(
                    "skimage is required for resize() with numpy array input.")
            skt_lib = skt
        elif HAS_CUPY and isinstance(image, cp.ndarray):
            if not HAS_CUCIM:
                raise RuntimeError(
                    "cucim is required for resize() with cupy array input.")
            skt_lib = cu_skt
        else:
            raise TypeError(f"Unsupported image type: {type(image)}.")

        # Validate and normalize shapes
        ndim = image.ndim
        if ndim == 2:
            H, W = image.shape
            C = None
        elif ndim == 3:
            H, W, C = image.shape
        else:
            raise ValueError(f"Unsupported shape: {image.shape}. Expect (H,W) or (H,W,C).")

        # Early exit if all images already match target size
        if (H, W) == output_shape:
            return image.copy()  # keep function semantics (new array)

        if order == 0:
            anti_aliasing = False
        return skt_lib.resize(
            image,
            output_shape=output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma,
        )

    @staticmethod
    def batch_resize[T: NPGeneric](
            batch_image: TypeArrayLike[T],
            output_shape: Tuple[int, int],
            order: int,
            mode='reflect',
            cval=0,
            clip=True,
            preserve_range=True,
            anti_aliasing=True,
            anti_aliasing_sigma: Optional[
                Union[float, Tuple[float, float]]
            ] = None
    ) -> TypeArrayLike[T]:
        if isinstance(batch_image, np.ndarray):
            if not HAS_SKIMAGE:
                raise RuntimeError(
                    "skimage is required for resize() with numpy array input.")
            skt_lib = skt
            np_lib = np
        elif HAS_CUPY and  isinstance(batch_image, cp.ndarray):
            if not HAS_CUCIM:
                raise RuntimeError(
                    "cucim is required for resize() with cupy array input.")
            skt_lib = cu_skt
            np_lib = cp
        else:
            raise TypeError(f"Unsupported image type: {type(batch_image)}.")

        # Validate and normalize shapes
        ndim = batch_image.ndim
        if ndim == 3:
            B, H, W = batch_image.shape
            C = None
            out_shape_full = (B, *output_shape)
        elif ndim == 4:
            B, H, W, C = batch_image.shape
            out_shape_full = (B, *output_shape, C)
        else:
            raise ValueError(f"Unsupported shape: {batch_image.shape}. Expect (B,H,W) or (B,H,W,C).")

        # Early exit if all images already match target size
        if (H, W) == output_shape:
            return batch_image.copy()  # keep function semantics (new array)

        out = np_lib.empty(out_shape_full, dtype=batch_image.dtype)

        if order == 0:
            anti_aliasing = False
        for i in range(B):
            image = batch_image[i]
            out[i] = skt_lib.resize(
                image,
                output_shape=output_shape,
                order=order,
                mode=mode,
                cval=cval,
                clip=clip,
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma,
            )
        return out

    @staticmethod
    def calc_center_padding(
            image_size: Tuple[int, int],
            w_to_h_ratio: float,
    ) -> npt.NDArray[np.int64]:
        """
        Return padding ((top, bottom), (left, right)) to center-pad an image of
        size (H, W) to the target aspect ratio w_to_h_ratio (W/H).
        """
        H, W = image_size
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid image_size {image_size}")
        if w_to_h_ratio <= 0:
            raise ValueError(f"w_to_h_ratio must be > 0, got {w_to_h_ratio}")

        # Target sizes that achieve the desired aspect ratio while covering the image
        target_W = int(math.ceil(H * w_to_h_ratio))
        target_H = int(math.ceil(W / w_to_h_ratio))

        pad_top = pad_bottom = pad_left = pad_right = 0

        if W < target_W:  # need horizontal padding
            pad_w = target_W - W
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        elif H < target_H:  # need vertical padding
            pad_h = target_H - H
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

        return np.array(
            [[pad_top, pad_bottom], [pad_left, pad_right]],
            dtype=np.int64,
        )

    @staticmethod
    def _normalize_pad_width(
            pad_width: Union[int, npt.NDArray[np.int64]],
            ndim: int,
    ) -> npt.NDArray[np.int64]:
        """
        Normalize pad_width to shape (ndim, 2).
        - If scalar: same padding on all dims.
        - If array of shape (k, 2) with k <= ndim: pad missing trailing dims with zeros.
        - If array of shape (2,) : broadcast to all dims.
        """
        if ndim <= 0:
            raise ValueError(f"Invalid ndim={ndim}")

        if np.isscalar(pad_width):
            val = int(pad_width)  # raises if non-int convertible
            return np.full((ndim, 2), val, dtype=np.int64)

        pw = np.asarray(pad_width, dtype=np.int64)

        if pw.ndim == 1:
            # e.g., (2,) → broadcast to all dims
            if pw.shape == (2,):
                return np.broadcast_to(pw, (ndim, 2)).copy()
            raise ValueError(f"pad_width 1D shape must be (2,), got {pw.shape}")

        if pw.ndim == 2 and pw.shape[1] == 2:
            k = pw.shape[0]
            if k > ndim:
                raise ValueError(f"pad_width has {k} dims > image ndim {ndim}")
            if k == ndim:
                return pw
            # pad trailing dims with zeros without concatenate
            out = np.zeros((ndim, 2), dtype=np.int64)
            out[:k] = pw
            return out

        raise ValueError(
            f"pad_width must be scalar, (2,), or (k,2); got shape {pw.shape}")

    @classmethod
    def padding[T: NPGeneric](
            cls,
            image: TypeArrayLike[T],
            pad_width: Union[int, npt.NDArray[np.int64]],
            mode: Literal["constant"] = "constant",
            constant_values: Union[float, Sequence[float]] = 0.0,
            batched: bool = False,
    ) -> TypeArrayLike[T]:
        if batched:
            if image.ndim == 3:
                # (N, H, W)
                pw = cls._normalize_pad_width(pad_width, image.ndim - 1)
                pw = np.concatenate([np.zeros((1, 2), dtype=pw.dtype), pw])
            elif image.ndim == 4:
                # (N, H, W, C)
                pw = cls._normalize_pad_width(pad_width, image.ndim - 2)
                pw = np.concatenate([
                    np.zeros((1, 2), dtype=pw.dtype),
                    pw,
                    np.zeros((1, 2), dtype=pw.dtype)]
                )
            else:
                raise ValueError(f"Unsupported image ndim {image.ndim}, expect 3 or 4.")
        else:
            if image.ndim == 2:
                # (H, W)
                pw = cls._normalize_pad_width(pad_width, image.ndim)
            elif image.ndim == 3:
                # (H, W, C)
                pw = cls._normalize_pad_width(pad_width, image.ndim - 1)
                pw = np.concatenate([pw, np.zeros((1, 2), dtype=pw.dtype)])
            else:
                raise ValueError(f"Unsupported image ndim {image.ndim}, expect 2 or 3.")
        np_lib = np
        if HAS_CUPY:
            if isinstance(image, cp.ndarray):
                np_lib = cp
        return np_lib.pad(image, pw, mode, constant_values=constant_values)

    @classmethod
    def center_padding[T: NPGeneric](
            cls,
            image: TypeArrayLike[T],
            w_to_h_ratio: float,
            mode: Literal["constant"] = "constant",
            constant_values: Union[float, Sequence[float]] = 0.,
    ) -> TypeArrayLike[T]:
        ndim = image.ndim
        if ndim not in (2, 3):
            raise ValueError(f"Unsupported image ndim {ndim}, expect 2 or 3.")
        H, W = image.shape[: 2]
        pad_width = cls.calc_center_padding(
            (H, W), w_to_h_ratio=w_to_h_ratio)
        return cls.padding(
            image=image,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )

    @classmethod
    def unpadding[T: NPGeneric](
            cls,
            image: TypeArrayLike[T],
            pad_width: Union[int, npt.NDArray[np.int64]],
            batched: bool = False,
    ) -> TypeArrayLike[T]:
        if batched:
            pad_width = cls._normalize_pad_width(
                pad_width=pad_width, ndim=image.ndim - 1)
            pad_width = np.concatenate([
                np.zeros((1, 2), dtype=pad_width.dtype),
                pad_width,
            ])
        else:
            pad_width = cls._normalize_pad_width(
                pad_width=pad_width, ndim=image.ndim)
        slices: List[slice] = []
        for c in pad_width.tolist():
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return image[tuple(slices)]

    @staticmethod
    def calc_center_cropping(
            image_size: Tuple[int, int],
            w_to_h_ratio: float,
    ) -> Tuple[int, int, int, int]:
        """
        Return (x0, y0, x1, y1) that center-crops an image of size (H, W)
        to the target aspect ratio (W/H = w_to_h_ratio). If already at ratio,
        returns the full image.
        """
        H, W = image_size
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid image_size {image_size}")
        if w_to_h_ratio <= 0:
            raise ValueError(f"w_to_h_ratio must be > 0, got {w_to_h_ratio}")

        # Compare without dividing by H to avoid precision issues
        if W < H * w_to_h_ratio:  # too tall → reduce height
            target_h = int(math.floor(W / w_to_h_ratio))
            y0 = (H - target_h) // 2
            y1 = y0 + target_h
            return 0, y0, W, y1
        elif W > H * w_to_h_ratio:  # too wide → reduce width
            target_w = int(math.floor(H * w_to_h_ratio))
            x0 = (W - target_w) // 2
            x1 = x0 + target_w
            return x0, 0, x1, H
        else:
            # Already at desired ratio
            return 0, 0, W, H

    @staticmethod
    def cropping[T: NPGeneric](
            image: TypeArrayLike[T],
            cropping_box: Tuple[int, int, int, int]
    ) -> TypeArrayLike[T]:
        cx, cy, cw, ch = cropping_box
        return image[cy: cy + ch, cx: cx + cw]

    @classmethod
    def center_cropping[T: NPGeneric](
            cls,
            image: TypeArrayLike[T],
            w_to_h_ratio: float,
    ) -> TypeArrayLike[T]:
        ndim = image.ndim
        if ndim not in (2, 3):
            raise ValueError(f"Unsupported image ndim {ndim}, expect 2 or 3.")
        H, W = image.shape[: 2]
        box = cls.calc_center_cropping(
            image_size=(H, W), w_to_h_ratio=w_to_h_ratio)
        return cls.cropping(image=image, cropping_box=box)

    @staticmethod
    def labelmap_on_image(
            image: TypeArrayLike[np.floating],
            labelmap: Union[TypeArrayLike[np.int64], list[TypeArrayLike[np.int64]]],
            color_table: Union[TypeArrayLike[np.floating], list[TypeArrayLike[np.floating]]],
            alpha: float = 0.5,
            dtype: np.generic | np.dtype = np.float32,
    ) -> TypeArrayLike[np.floating]:
        """

        :param image: (H, W) or (H, W, 3)
        :param labelmap: (H, W) or (N, H, W) or list of (H, W)
        :param color_table: (C, 3) or (N, C, 3) or list of (C, 3)
        :param alpha:
        :param dtype:
        :return:
        """

        nplib = np
        to_array_fn = xp.to_numpy
        if HAS_CUPY:
            def _is_array(x):
                return isinstance(x, (np.ndarray, cp.ndarray))

            if isinstance(image, cp.ndarray):
                nplib = cp
                to_array_fn = xp.to_cupy
        else:
            def _is_array(x):
                return isinstance(x, np.ndarray)

        if image.ndim == 2:
            image = image[..., np.newaxis]  # (H, W) -> (H, W, 1)
        assert image.ndim == 3, \
            f'Image must be 2D or 3D, got {image.ndim}D'
        if image.shape[-1] == 1:
            image = nplib.repeat(image, 3, axis=-1)
        image = image.astype(dtype, copy=False)  # Ensure float type

        assert image.shape[-1] == 3, \
            f'Image must have 3 channels, got {image.shape[-1]} channels'
        H, W, _ = image.shape

        if isinstance(labelmap, (list, tuple)):
            labelmap = nplib.stack([to_array_fn(lm) for lm in labelmap], axis=0)
        elif _is_array(labelmap):
            if labelmap.ndim == 2:
                labelmap = labelmap[np.newaxis, ...]
            assert labelmap.ndim == 3, \
                f'Labelmap must be 2D or 3D, got {labelmap.ndim}D'
            labelmap = to_array_fn(labelmap)
        else:
            raise ValueError(
                f'Labelmap must be a list or numpy/cupy array, got {type(labelmap)}')
        B = labelmap.shape[0]
        assert labelmap.shape[1:] == (H, W), \
            f'Labelmap shape {labelmap.shape[1:]} must match image shape {image.shape[:2]}'
        labelmap = labelmap.astype(np.int64, copy=False)  # Ensure int type

        if isinstance(color_table, (list, tuple)):
            if len(color_table) != B:
                raise ValueError(
                    f'Color table length {len(color_table)} must match labelmap batch size {B}')
            ct0 = color_table[0]
            for ct in color_table:
                if ct.shape != ct0.shape:
                    raise ValueError(
                        f'All color tables must have the same shape, got {ct.shape} and {ct0.shape}')
            color_table = [to_array_fn(ct) for ct in color_table]
        elif _is_array(color_table):
            color_table = to_array_fn(color_table)
            if color_table.ndim == 2:  # (C, 3)
                color_table = nplib.repeat(color_table[np.newaxis, ...], B, axis=0)  # (B, C, 3)
            elif color_table.ndim == 3:  # (N, C, 3)
                if color_table.shape[0] != B:
                    raise ValueError(
                        f'Color table shape {color_table.shape[0]} must match labelmap batch size {B}')
            else:
                raise ValueError(
                    f'Color table must be 2D or 3D, got {color_table.ndim}D')
        else:
            raise ValueError(
                f'Color table must be a numpy array or a list, got {type(color_table)}')

        # image (H, W, 3)
        # labelmap (B, H, W)
        # color_table (B, C, 3) or list of (C, 3)

        colors = nplib.empty((B, H, W, 3), dtype=dtype)
        for b in range(B):
            colors[b] = color_table[b][labelmap[b]]  # (H, W, 3)

        mask = (labelmap > 0)[..., np.newaxis]  # (B, H, W, 1)

        weight = 1. / mask.astype(dtype).sum(axis=0).clip(min=1e-6, max=None)
        blended = (colors * weight).sum(axis=0)  # (H, W, 3)
        active = mask.any(axis=0).repeat(3, axis=-1)  # (H, W, 3)
        alpha = float(alpha)
        out = nplib.where(
            active,
            blended * alpha + image * (1 - alpha),
            image
        )
        return out

    @staticmethod
    def labelmap_contour_on_image[T: NPGeneric](
            image: TypeArrayLike[T],
            labelmap: TypeArrayLike[np.int64],
            color_table: TypeArrayLike[T],
    ) -> TypeArrayLike[T]:
        """
        :param image:
        :param labelmap:
        :param color_table:
        :return:
        """

        if isinstance(image, np.ndarray):
            if not HAS_SKIMAGE:
                raise RuntimeError(
                    "skimage is required for contour-based labelmap on image")
            nplib = np
            mark_bound_fn = mark_boundaries
        elif HAS_CUPY and isinstance(image, cp.ndarray):
            nplib = cp
            mark_bound_fn = cu_mark_boundaries
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        labeled_image = image.copy()
        classes = nplib.unique(labelmap)
        for c in classes:
            if c == 0:
                continue
            mask: TypeArrayLike = labelmap == c
            if not nplib.any(mask):
                continue
            labeled_image = mark_bound_fn(
                labeled_image,
                mask.astype(int),
                color=color_table[c],
                outline_color=None,
            )
        return labeled_image


resize = ImageUtils.resize
calc_center_padding = ImageUtils.calc_center_padding
padding = ImageUtils.padding
center_padding = ImageUtils.center_padding
unpadding = ImageUtils.unpadding
cropping = ImageUtils.cropping
center_cropping = ImageUtils.center_cropping
labelmap_on_image = ImageUtils.labelmap_on_image
labelmap_contour_on_image = ImageUtils.labelmap_contour_on_image