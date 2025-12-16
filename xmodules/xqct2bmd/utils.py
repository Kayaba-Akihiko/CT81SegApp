#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import TypeAlias, Optional, Union, Sequence, Tuple, Literal
import functools

import numpy as np
import numpy.typing as npt
from scipy.ndimage import (
    label as label_component,
    generate_binary_structure,
)

from xmodules.xutils import array_utils as xp

cp = None
cpt = None
cu_label_component = None
cu_generate_binary_structure = None
if xp.HAS_CUPY:
    import cupy as cp
    import cupy.typing as cpt
    from cupyx.scipy.ndimage import (
        label as cu_label_component,
        generate_binary_structure as cu_generate_binary_structure,
    )

torch = None
if xp.HAS_TORCH:
    import torch


NPIntOrFloat: TypeAlias = xp.NPIntOrFloat
TypeBBoxND: TypeAlias = Union[
    Tuple[Union[int, float], ...],
    npt.NDArray[NPIntOrFloat],
    'cpt.NDArray[NPIntOrFloat]',
    'torch.Tensor',
]

def keep_largest_connected_component_np(
        labelmap: xp.TypeArrayLike[np.integer],
        target_class: int | Sequence[int] | None = None,
        structure: Optional[xp.TypeArrayLike[np.bool_]] = None,
        n_components: int = 1,
        exclusion_volume_threshold: Optional[Union[int, float]] = None,
) -> xp.TypeArrayLike[np.integer]:
    assert n_components >= 1, \
        f'Expected n_components to be at least 1, got {n_components}.'

    if isinstance(labelmap, np.ndarray):
        _nplib = np
        _as_array_fn = xp.to_numpy
        _generate_structure_fn = generate_binary_structure
        _label_comp_fn = label_component
    elif xp.HAS_CUPY and isinstance(labelmap, cp.ndarray):
        _nplib = cp
        _as_array_fn = xp.to_cupy
        _generate_structure_fn = cu_generate_binary_structure
        _label_comp_fn = cu_label_component
    else:
        raise TypeError(f"{type(labelmap)=} must be numpy/cupy ndarray")

    if exclusion_volume_threshold is not None:
        if exclusion_volume_threshold <= 0:
            raise ValueError(f"exclusion_volume_threshold must be positive, got {exclusion_volume_threshold}")
        if exclusion_volume_threshold >= 1:
            raise ValueError(f"exclusion_volume_threshold must be < 1, got {exclusion_volume_threshold}")

    # --- Build connectivity structure if not provided
    if structure is None:
        # Full connectivity in N-D: connectivity = N (same as scipy's generate_binary_structure(N, N))
        ndim = labelmap.ndim
        structure = _generate_structure_fn(ndim, ndim)
    else:
        structure = _as_array_fn(structure)

    # --- Normalize target classes (exclude background=0 if None)
    if target_class is None:
        uniq = _nplib.unique(labelmap)
        # remove background 0
        target_class = uniq[uniq != _as_array_fn(0)]
        # convert to plain list for iteration
        target_class = target_class.tolist()

    if not isinstance(target_class, (list, tuple)):
        target_class = [target_class]
    target_class = list(set(target_class))

    labelmap_ = _as_array_fn(labelmap)
    restore_shape = labelmap_.shape
    results = _nplib.zeros_like(labelmap_, dtype=labelmap_.dtype)

    for cls in target_class:
        # Binary mask for this class
        binary = (labelmap_ == cls)
        if _nplib.count_nonzero(binary) == 0:
            continue

        # Tight crop to speed up labeling
        bbox = get_bbox_from_labelmap(binary, valid_class=[1])  # nonzero in `binary`
        binary_c = crop_image_bbox(binary, bbox)

        # Connected components on the cropped mask
        labeled_mask, n_found = _label_comp_fn(binary_c, structure=structure)
        if n_found == 0:
            continue

        # Component sizes (ignore background id 0)
        # Ensure bincount has enough length even for highest label id:
        max_id = int(labeled_mask.max())
        counts = _nplib.bincount(labeled_mask.ravel(), minlength=max_id + 1)
        vols = counts[1:]  # component ids are 1..n_found

        # Select top-k largest components
        if vols.size == 0:
            continue

        order = _nplib.argsort(vols)[::-1].astype(np.int64)
        keep_ids = order[:n_components] + 1  # shift back to component ids

        if exclusion_volume_threshold is not None:
            total_vol = vols.sum()
            thresh = total_vol * exclusion_volume_threshold

            # mark components (in sorted order) that are too small
            sorted_vols = vols[order]  # vols aligned with descending order
            keep_mask = sorted_vols[:n_components] >= thresh
            keep_ids = keep_ids[keep_mask]


        if keep_ids.size == 0:
            continue

        # Build class result in the cropped region
        # (vectorized: set kept components to class id, others 0)
        if _nplib is np:
            mask_keep = _nplib.isin(labeled_mask, keep_ids)
        else:  # cupy
            mask_keep = _nplib.isin(
                labeled_mask, _nplib.asarray(keep_ids, dtype=np.int64))
        class_cropped = _nplib.where(
            mask_keep, cls, 0).astype(labelmap_.dtype)

        # Restore to full canvas and merge
        class_full = restore_image_bbox(
            class_cropped, bbox, restore_shape=restore_shape, padding_value=0)
        results = _nplib.where(class_full != 0, class_full, results)

    return results



def get_bbox_from_labelmap(
        labelmap: xp.TypeArrayLike[np.integer],
        valid_class: Optional[Union[int, Sequence[int]]] = None
) -> TypeBBoxND:
    # --- Pick backend + helpers
    if isinstance(labelmap, np.ndarray):
        _nplib = np
        _as_array_fn = xp.to_numpy
        # reducers for (K, N) coords array
        _min_reduce = lambda a: a.min(axis=0)
        _max_reduce = lambda a: a.max(axis=0)
        _to_int = lambda a: a.astype(np.int64)
    elif xp.HAS_CUPY and isinstance(labelmap, cp.ndarray):
        _nplib = cp
        _as_array_fn = xp.to_cupy
        _min_reduce = lambda a: a.min(axis=0)
        _max_reduce = lambda a: a.max(axis=0)
        _to_int = lambda a: a.astype(cp.int64)
    elif xp.HAS_TORCH and isinstance(labelmap, torch.Tensor):
        _nplib = torch
        _as_array_fn = functools.partial(xp.to_torch, device=labelmap.device)
        _min_reduce = lambda a: a.min(dim=0).values
        _max_reduce = lambda a: a.max(dim=0).values
        _to_int = lambda a: a.to(torch.int64)
    else:
        raise TypeError(f"{type(labelmap)=} must be numpy/cupy ndarray or torch.Tensor")

    # --- Select foreground classes
    if valid_class is None:
        u = _nplib.unique(labelmap)
        # Exclude 0 as background if present
        valid = u[u != 0]
        valid_classes_seq = valid
    else:
        # normalize to sequence/array on same backend for isin()
        if isinstance(valid_class, int):
            valid_classes_seq = _as_array_fn([valid_class])
        else:
            valid_classes_seq = _as_array_fn(valid_class)

    # --- Build mask and gather coordinates
    mask = _nplib.isin(labelmap, valid_classes_seq)
    coords = _nplib.argwhere(mask)  # shape: (K, N) where N=labelmap.ndim

    if coords.numel() == 0 if _nplib is torch else coords.size == 0:
        raise RuntimeError(f"No valid classes found in the label map: {valid_class=}.")

    # --- Per-axis min/max (make max exclusive with +1)
    mins = _to_int(_min_reduce(coords))
    maxs = _to_int(_max_reduce(coords)) + 1

    # --- Return a plain Python tuple of ints (backend-agnostic)
    mins_np = np.asarray(xp.to_numpy(mins)).astype(np.int64)
    maxs_np = np.asarray(xp.to_numpy(maxs)).astype(np.int64)
    return tuple(mins_np.tolist() + maxs_np.tolist())


def crop_image_bbox[T: NPIntOrFloat](
        image: xp.TypeArrayLike[T],
        bbox: TypeBBoxND,
) -> xp.TypeArrayLike[T]:
    # Always safe to convert this small vector to NumPy for int extraction:
    bbox_np = xp.to_numpy(bbox)
    if bbox_np.size % 2 != 0:
        raise ValueError(f"Invalid bbox length {bbox_np.size}: must be even (2N).")
    n = bbox_np.size // 2
    if n > image.ndim:
        raise ValueError(f"bbox dims ({n}) exceed image.ndim ({image.ndim}).")

    mins = bbox_np[:n].astype(int).tolist()
    maxs = bbox_np[n:].astype(int).tolist()

    # Validate per-axis
    for d, (lo, hi, dim) in enumerate(zip(mins, maxs, image.shape)):
        if lo < 0:
            raise ValueError(f"Invalid bbox on axis {d}: min ({lo}) < 0.")
        if hi > dim:
            raise ValueError(f"Invalid bbox on axis {d}: max ({hi}) > dim ({dim}).")
        if lo >= hi:
            raise ValueError(f"Invalid bbox on axis {d}: min ({lo}) >= max ({hi}); empty region.")

    sl = tuple(slice(lo, hi) for lo, hi in zip(mins, maxs))
    return image[sl]


def restore_image_bbox[T: NPIntOrFloat](
        image: xp.TypeArrayLike[T],
        bbox: TypeBBoxND,
        restore_shape: Tuple[int, ...],
        padding_mode: Literal['constant'] = 'constant',
        padding_value: Union[int, float] = 0,
) -> xp.TypeArrayLike[T]:

    restore_shape = xp.to_numpy(restore_shape)
    bbox = xp.to_numpy(bbox)

    # ---- Parse & validate bbox / shape
    bbox = np.asarray(bbox, dtype=int)
    if bbox.size % 2 != 0:
        raise ValueError(f"bbox length must be even (2N), got {bbox.size}")
    n = bbox.size // 2
    mins = bbox[:n]
    maxs = bbox[n:]

    if len(restore_shape) != n:
        raise ValueError(
            f"bbox must have same length as restore_shape, got {bbox} and {restore_shape}")
    if image.ndim != n:
        raise ValueError(
            f"image must have same length as bbox, got {image.ndim}D and {bbox}")

    if np.any(mins >= maxs):
        raise ValueError(f"bbox must be valid (min <= max), got {bbox}")

    pad_lefts = mins
    pad_rights = restore_shape - maxs
    pad_spec = tuple(zip(pad_lefts, pad_rights))
    return xp.pad(image, pad_spec, mode=padding_mode, constant_values=padding_value)