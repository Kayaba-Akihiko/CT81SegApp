#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Literal, TypeAlias, TypeVar, Union, Tuple, Optional

import numpy as np
import numpy.typing as npt

from .lib_utils import import_available
from .array_utils import to_numpy, to_cupy, torch, to_torch

cp = None
cpt = None
if HAS_CUPY := import_available('cupy'):
    import cupy as cp
    import cupy.typing as cpt

torch = None
if HAS_TORCH := import_available('torch'):
    import torch

NPGeneric: TypeAlias = np.generic
T = TypeVar("T", bound=NPGeneric)
NDArray: TypeAlias = Union[npt.NDArray[T], 'cpt.NDArray[T]', 'torch.Tensor']


def calc_statistics(
        array: NDArray[NPGeneric],
        quantile_low: Optional[float] = None,
        quantile_high: Optional[float] = None,
        channel_first=False,
        batched=False,
) -> NDArray[np.float64]:
    if quantile_low is not None:
        assert 0 <= quantile_low <= 1, f'{quantile_low}'
    if quantile_high is not None:
        assert 0 <= quantile_high <= 1, f'{quantile_high}'
    if quantile_low is not None and quantile_high is not None:
        assert quantile_low < quantile_high, f'{quantile_low} < {quantile_high}'

    if batched:
        assert array.ndim > 2, f'{array.shape=}'
        B = len(array)
        if channel_first:
            # (B, C, ...)
            channel_axis = 1
            split_axis = -1
            C = array.shape[channel_axis]
            # (B, C, D)
            array = array.reshape(B, C, -1)
        else:
            # (B, ..., C)
            channel_axis = -1
            split_axis = 1
            C = array.shape[channel_axis]
            array = array.reshape(B, -1, C)
    else:
        assert array.ndim > 1, f'{array.shape=}'
        if channel_first:
            # (C, ...)
            channel_axis = 0
            split_axis = -1
            C = array.shape[channel_axis]
            array = array.reshape(C, -1)
        else:
            # (..., C)
            channel_axis = -1
            split_axis = 0
            C = array.shape[channel_axis]
            array = array.reshape(-1, C)

    if HAS_TORCH and isinstance(array, torch.Tensor):
        if quantile_low is not None or quantile_high is not None:
            array = array.clone()
        # (B, 1, C) or (B, C, 1) or (1, C) or (C, 1)
        if quantile_low is not None:
            min_val = torch.quantile(
                array, quantile_low, dim=split_axis, keepdim=True)
            array.clamp_(min=min_val)
        else:
            min_val = torch.min(array, dim=split_axis, keepdim=True).values
        if quantile_high is not None:
            max_val = torch.quantile(
                array, quantile_high, dim=split_axis, keepdim=True)
            array.clamp_(max=max_val)
        else:
            max_val = torch.max(array, dim=split_axis, keepdim=True).values
        std_val, mean_val = torch.std_mean(
            array, dim=split_axis, keepdim=True)

        # (B, 4, C) or (B, C, 4) or (4, C) or (C, 4)
        statistics = torch.cat(
            (min_val, max_val, mean_val, std_val),
            dim=split_axis,
        )
        return statistics

    if HAS_CUPY and isinstance(array, cp.ndarray):
        nplib = cp
    else:
        nplib = np

    if quantile_low is not None or quantile_high is not None:
        array = array.copy()
    # (B, 1, C) or (B, C, 1) or (1, C) or (C, 1)
    if quantile_low is not None:
        min_val = nplib.quantile(
            array, quantile_low, axis=split_axis, keepdims=True)
        array.clip(min=min_val, out=array)
    else:
        min_val = nplib.min(array, axis=split_axis, keepdims=True)
    if quantile_high is not None:
        max_val = nplib.quantile(
            array, quantile_high, axis=split_axis, keepdims=True)
        array.clip(max=max_val, out=array)
    else:
        max_val = nplib.max(array, axis=split_axis, keepdims=True)

    mean_val = nplib.mean(array, axis=split_axis, keepdims=True)
    std_val = nplib.std(
        array, axis=split_axis, keepdims=True, ddof=1)
    # (B, 4, C) or (B, C, 4) or (4, C) or (C, 4)
    statistics = nplib.concatenate(
        (min_val, max_val, mean_val, std_val),
        axis=split_axis,
    ).astype(np.float64, copy=False)
    return statistics


def match_ndims[T: NPGeneric](
        source: NDArray[T],
        target: NDArray[NPGeneric],
        channel_first=False,
        batched=False,
) -> NDArray[T]:
    if target.ndim == source.ndim:
        return source
    n_pads = target.ndim - source.ndim

    new_axis_tag = np.newaxis
    if HAS_TORCH and isinstance(target, torch.Tensor):
        new_axis_tag = None

    if batched:
        assert source.ndim > 2, f'{source.shape=}'
        assert target.ndim > 2, f'{target.shape=}'
        if channel_first:
            # (B, C, ...)
            expand_fn = lambda x: x[..., new_axis_tag]
            squeeze_fn = lambda x: x.squeeze(axis=-1)
        else:
            # (B, ..., C)
            expand_fn = lambda x: x[:, new_axis_tag, ...]
            squeeze_fn = lambda x: x.squeeze(axis=1)
    else:
        assert source.ndim > 1, f'{source.shape=}'
        assert target.ndim > 1, f'{target.shape=}'
        if channel_first:
            # (C, ...)
            expand_fn = lambda x: x[..., new_axis_tag]
            squeeze_fn = lambda x: x.squeeze(axis=-1)
        else:
            # (..., C)
            expand_fn = lambda x: x[new_axis_tag, ...]
            squeeze_fn = lambda x: x.squeeze(axis=0)

    while n_pads != 0:
        if n_pads > 0:
            source = expand_fn(source)
        elif n_pads < 0:
            assert source.shape[0] == 1, f'{source.shape}'
            source = squeeze_fn(source)
        n_pads = target.ndim - source.ndim
    return source

def _norm_base[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        method: Literal['z', 'minmax'],
        mode: Literal['norm', 'denorm'],
        channel_first=False,
        batched=False,
) -> NDArray[T]:



    if batched:
        assert array.ndim > 3, f'{array.shape=}'
        # (B, 4, C) or (B, C, 4)
        assert statistics.ndim == 3, f'{statistics.shape}'
        if channel_first:
            # (B, C, 4)
            split_axis = -1
        else:
            # (B, 4, C)
            split_axis = 1
    else:
        assert array.ndim > 2
        assert statistics.ndim == 2, f'{statistics.shape}'
        if channel_first:
            # (C, 4)
            split_axis = -1
        else:
            # (4, C)
            split_axis = 0

    torch_backend = False
    if HAS_TORCH and isinstance(array, torch.Tensor):
        nplib = torch
        to_array_fn = to_torch
        torch_backend = True
    elif HAS_CUPY and isinstance(array, cp.ndarray):
        nplib = cp
        to_array_fn = to_cupy
    else:
        nplib = np
        to_array_fn = to_numpy

    statistics = to_array_fn(statistics)

    if torch_backend:
        min_val, max_val, mean_val, std_val = torch.chunk(
            statistics,
            dim=split_axis,
            chunks=4,
        )
    else:
        min_val, max_val, mean_val, std_val = nplib.split(
            statistics,
            axis=split_axis,
            indices_or_sections=4,
        )
    if method == 'z':
        mean_val = match_ndims(
            mean_val, array, channel_first=channel_first, batched=batched)
        std_val = match_ndims(
            std_val, array, channel_first=channel_first, batched=batched)
        bias = mean_val
        weight = std_val
    elif method == 'minmax':
        min_val = match_ndims(
            min_val, array, channel_first=channel_first, batched=batched)
        max_val = match_ndims(
            max_val, array, channel_first=channel_first, batched=batched)
        bias = min_val
        weight = max_val - min_val
    else:
        raise NotImplementedError(f'Unknown mode: {method}.')
    del min_val, max_val, mean_val, std_val

    if mode == 'denorm':
        return nplib.multiply(array, weight) + bias

    if mode != 'norm':
        raise NotImplementedError(f'Unknown mode: {mode}.')

    # (B, ..., C) or (B, C, ...) or (..., C) or (C, ...)
    numerator = array - bias
    denominator = weight
    valid = denominator > 0

    if torch_backend:
        if torch.all(valid):
            return torch.div(numerator, denominator)
        denominator: torch.Tensor
        denominator = denominator.expand_as(array)
        valid = valid.expand_as(array)

        res = torch.zeros_like(array)
        res[valid] = torch.div(
            numerator[valid], denominator[valid])
        return res

    if nplib.all(valid):
        return nplib.divide(numerator, denominator)
    denominator = nplib.broadcast_to(denominator, array.shape)
    valid = nplib.broadcast_to(valid, array.shape)

    res = nplib.zeros_like(array)
    res[valid] = nplib.divide(
        numerator[valid], denominator[valid])

    return res

def norm[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        method: Literal['z', 'minmax'],
        channel_first=False,
        batched=False,
)-> NDArray[T]:
    return _norm_base(
        array=array,
        statistics=statistics,
        method=method,
        mode='norm',
        channel_first=channel_first,
        batched=batched,
    )

def z_norm[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        channel_first=False,
        batched=False,
) -> NDArray[T]:
    return norm(
        array,
        statistics,
        method='z',
        channel_first=channel_first,
        batched=batched,
    )

def minmax_norm[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        channel_first=False,
        batched=False,
) -> NDArray[T]:
    return norm(
        array,
        statistics,
        method='minmax',

        channel_first=channel_first,
        batched=batched,
    )

def self_norm[T: NPGeneric](
        array: NDArray[T],
        method: Literal['z', 'minmax'],
        quantile_low: Optional[float] = None,
        quantile_high: Optional[float] = None,
        channel_first=False,
        batched=False,
        return_statistics=False,
) -> Union[NDArray[T], Tuple[NDArray[T], NDArray[np.float64]]]:
    statistics = calc_statistics(
        array,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        channel_first=channel_first,
        batched=batched)
    norm_array = norm(
        array, statistics, method,
        channel_first=channel_first, batched=batched,
    )
    if not return_statistics:
        return norm_array
    return norm_array, statistics

def self_z_norm[T: NPGeneric](
        array: NDArray[T],
        quantile_low: Optional[float] = None,
        quantile_high: Optional[float] = None,
        channel_first=False,
        batched=False,
        return_statistics=False,
) -> Union[NDArray[T], Tuple[NDArray[T], NDArray[np.float64]]]:
    return self_norm(
        array,
        'z',
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        channel_first=channel_first,
        batched=batched,
        return_statistics=return_statistics,
    )

def self_minmax_norm[T: NPGeneric](
        array: NDArray[T],
        quantile_low: Optional[float] = None,
        quantile_high: Optional[float] = None,
        channel_first=False,
        batched=False,
        return_statistics=False,
) -> Union[NDArray[T], Tuple[NDArray[T], NDArray[np.float64]]]:
    return self_norm(
        array,
        'minmax',
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        channel_first=channel_first,
        batched=batched,
        return_statistics=return_statistics,
    )


def denorm[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        method: Literal['z', 'minmax'],
        channel_first=False,
        batched=False,
) -> NDArray[T]:
    return _norm_base(
        array=array,
        statistics=statistics,
        method=method,
        mode='denorm',
        channel_first=channel_first,
        batched=batched,
    )

def z_denorm[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        channel_first=False,
        batched=False,
) -> NDArray[T]:
    return denorm(
        array, statistics,
        method='z',
        channel_first=channel_first,
        batched=batched,
    )

def minmax_denorm[T: NPGeneric](
        array: NDArray[T],
        statistics: NDArray[NPGeneric],
        channel_first=False,
        batched=False,
) -> NDArray[T]:
    return denorm(
        array, statistics, method='minmax',
        channel_first=channel_first,
        batched=batched,
    )