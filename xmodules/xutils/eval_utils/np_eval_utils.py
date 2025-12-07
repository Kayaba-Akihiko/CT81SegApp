#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import (
    Union, Optional, Tuple, TypeAlias, TypeVar, Literal, Sequence, Set,
    Dict, Type
)
import math

import numpy as np
import numpy.typing as npt
from scipy.stats import norm  # for z-score
from scipy.stats import rankdata  # Needed for ranking
from scipy.special import logsumexp

from ..lib_utils import import_available
from ..array_utils import to_numpy, to_cupy

if HAS_CUPY := import_available('cupy'):
    import cupy as cp
    import cupy.typing as cpt
    from cupyx.scipy.special import logsumexp as cu_logsumexp, softmax as cu_softmax

NPGeneric: TypeAlias = np.generic
NPFloating: TypeAlias = np.floating
T = TypeVar("T", bound=NPGeneric)
NDArray: TypeAlias = Union[npt.NDArray[T], 'cpt.NDArray[T]']
StatMetric: TypeAlias = Literal[
    'dice', 'jaccard', 'accuracy', 'precision', 'recall']

def pearson_correlation[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
        eps=1e-12,
) -> NDArray[T]:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')

    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if isinstance(x, cp.ndarray) or isinstance(y, cp.ndarray):
            nplib = cp
            to_array_fn = to_cupy

    x, y = to_array_fn(x), to_array_fn(y)
    x_mean = nplib.mean(x, axis=axis, keepdims=True)
    y_mean = nplib.mean(y, axis=axis, keepdims=True)
    x_z_mean = x - x_mean
    y_z_mean = y - y_mean
    x_z_mean_norm = nplib.linalg.norm(
        x_z_mean, axis=axis, keepdims=True)
    y_z_mean_norm = nplib.linalg.norm(
        y_z_mean, axis=axis, keepdims=True)
    denominator = np.multiply(x_z_mean_norm, y_z_mean_norm)
    corr = nplib.sum(
        nplib.divide(
            nplib.multiply(x_z_mean, y_z_mean),
            denominator + eps,
        ),
        axis=axis, keepdims=keepdims,
    )
    return corr


def spearman_correlation[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
        eps=1e-12,
) -> NDArray[T]:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')

    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if isinstance(x, cp.ndarray) or isinstance(y, cp.ndarray):
            nplib = cp
            to_array_fn = to_cupy

    x, y = to_numpy(x), to_numpy(y)
    if axis is None:
        x_rank = rankdata(x.ravel()).reshape(x.shape)
        y_rank = rankdata(y.ravel()).reshape(y.shape)
    elif isinstance(axis, int):
        x_rank = np.apply_along_axis(rankdata, axis, x)
        y_rank = np.apply_along_axis(rankdata, axis, y)
    else:
        raise NotImplementedError("Spearman correlation does not support multi-axis ranking.")
    x_rank, y_rank = to_array_fn(x_rank), to_array_fn(y_rank)

    # Now compute Pearson correlation on the ranks
    x_mean = nplib.mean(x_rank, axis=axis, keepdims=True)
    y_mean = nplib.mean(y_rank, axis=axis, keepdims=True)
    x_z_mean = x_rank - x_mean
    y_z_mean = y_rank - y_mean
    x_z_mean_norm = nplib.linalg.norm(x_z_mean, axis=axis, keepdims=True)
    y_z_mean_norm = nplib.linalg.norm(y_z_mean, axis=axis, keepdims=True)
    denominator = nplib.multiply(x_z_mean_norm, y_z_mean_norm)
    corr = nplib.sum(
        nplib.divide(
            nplib.multiply(x_z_mean, y_z_mean),
            denominator + eps,
        ),
        axis=axis, keepdims=keepdims,
    )
    return corr


def intra_class_correlation[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
        eps=1e-12,
) -> NDArray[T]:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')

    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if isinstance(x, cp.ndarray) or isinstance(y, cp.ndarray):
            nplib = cp
            to_array_fn = to_cupy

    x, y = to_array_fn(x), to_array_fn(y)
    x_mean = nplib.mean(x, axis=axis, keepdims=True)
    y_mean = nplib.mean(y, axis=axis, keepdims=True)
    global_mean = (x_mean + y_mean) / 2.
    x_zero_mean = x - global_mean
    y_zero_mean = y - global_mean

    denom_t1 = nplib.sum(
        nplib.square(x_zero_mean), axis=axis, keepdims=True)
    denom_t2 = nplib.sum(
        nplib.square(y_zero_mean), axis=axis, keepdims=True)
    denorm = denom_t1 + denom_t2
    corr = nplib.sum(
        nplib.divide(
            nplib.multiply(x_zero_mean, y_zero_mean),
            denorm + eps,
        ),
        axis=axis, keepdims=keepdims
    ) * 2.
    return corr


def intersection_and_sum[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
) -> Tuple[
    NDArray[T],
    NDArray[T],
    NDArray[T],
    Union[Type[np], Type['cp']],
]:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')
    if x.min() < 0 or x.max() > 1:
        raise RuntimeError(
            f'Only support binary values but got {x.min()} {x.max()}.')
    if y.min() < 0 or y.min() > 1:
        raise RuntimeError(
            f'Only support binary values but got {y.min()} {y.max()}.')

    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if isinstance(x, cp.ndarray) or isinstance(y, cp.ndarray):
            nplib = cp
            to_array_fn = to_cupy

    x, y = to_array_fn(x), to_array_fn(y)
    intersection = nplib.sum(
        nplib.multiply(x, y), axis=axis, keepdims=keepdims)
    sum_ = (
            nplib.sum(x, axis=axis, keepdims=keepdims)
            + nplib.sum(y, axis=axis, keepdims=keepdims)
    )
    mask = (sum_ > 0.)
    return intersection, sum_, mask, nplib


def dice_score[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
) -> NDArray[T]:
    intersection, sum_, mask, nplib = intersection_and_sum(
        x=x, y=y, axis=axis, keepdims=keepdims)

    if nplib.all(mask):
        return nplib.divide(intersection * 2, sum_)

    dice = nplib.ones_like(intersection)
    dice[mask] = nplib.divide(intersection[mask] * 2., sum_[mask])
    return dice


def jaccard_index[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
) -> NDArray[T]:
    # x: (B, C, H*W)
    # Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    intersection, sum_, mask, nplib = intersection_and_sum(
        x=x, y=y, axis=axis, keepdims=keepdims)

    if nplib.all(mask):
        return nplib.divide(intersection, sum_ - intersection)

    jac = np.ones_like(intersection)
    valid_intersection = intersection[mask]
    jac[mask] = np.divide(
        valid_intersection, sum_[mask] - valid_intersection)
    return jac


def signal_to_noise_ratio[T: NPFloating](
        x: NDArray[T], y: NDArray[T],
        r=255.,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
        eps=1e-12,
) -> NDArray[T]:
    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if isinstance(x, cp.ndarray) or isinstance(y, cp.ndarray):
            nplib = cp
            to_array_fn = to_cupy

    x, y = to_array_fn(x), to_array_fn(y)
    mse = nplib.mean(
        nplib.square(x / r - y / r), axis=axis, keepdims=keepdims)
    return 10 * nplib.log10(1. / (mse + eps))


def gaussian_nll[T: NPFloating](
        x: NDArray[T], y: NDArray[T], var: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
        full: bool = False,
        eps=1e-12,
):
    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if (
                isinstance(x, cp.ndarray)
                or isinstance(y, cp.ndarray)
                or isinstance(var, cp.ndarray)
        ):
            nplib = cp
            to_array_fn = to_cupy
    x, y, var = map(to_array_fn, (x, y, var))

    if var.shape != x.shape:
        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if x.shape[:-1] == var.shape:
            var = nplib.expand_dims(var, -1)
            # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
            # This is also a homoscedastic case.
            # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif (
                x.shape[:-1] == var.shape[:-1] and var.shape[-1] == 1
        ):  # Heteroscedastic case
            pass
        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Entries of var must be non-negative
    if nplib.any(var < 0):
        raise ValueError("var has negative entry/entries")

    var = var.copy()
    nplib.clip(var, a_min=eps, a_max=None, out=var)

    nll = 0.5 * (np.log(var) + (x - y) ** 2 / var)
    if full:
        nll += 0.5 * math.log(2 * math.pi)
    nll = nplib.mean(nll, axis=axis, keepdims=keepdims)
    return nll


def prediction_interval_coverage_probability[T: NPFloating](
        x: NDArray[T], y: NDArray[T], var: NDArray[T],
        alpha: float, # confidence level, e.g., 0.95 for 95%
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
        eps=1e-12,
) -> NDArray[T]:
    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY:
        if (
                isinstance(x, cp.ndarray)
                or isinstance(y, cp.ndarray)
                or isinstance(var, cp.ndarray)
        ):
            nplib = cp
            to_array_fn = to_cupy
    x, y, var = map(to_array_fn, (x, y, var))

    if var.shape != x.shape:
        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if x.shape[:-1] == var.shape:
            var = nplib.expand_dims(var, -1)
            # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
            # This is also a homoscedastic case.
            # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif (
                x.size[:-1] == var.size[:-1] and var.size[-1] == 1
        ):  # Heteroscedastic case
            pass
        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Entries of var must be non-negative
    if nplib.any(var < 0.):
        raise ValueError("var has negative entry/entries")

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")

    var = var.copy()
    nplib.clip(var, a_min=eps, a_max=None, out=var)

    std = nplib.sqrt(var)

    z = norm.ppf(0.5 + alpha / 2)
    # Compute prediction intervals
    lower = x - z * std
    upper = x + z * std
    within_interval = (y >= lower) & (y <= upper)
    picp = nplib.mean(within_interval, axis=axis, keepdims=keepdims)
    return picp


def entropy_with_logits[T: NPFloating](
    logits: NDArray[T],
    probs: Optional[NDArray[T]] = None,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> NDArray[T]:
    """
    If probs is None: returns H(softmax(logits)).
    If probs is given: returns cross-entropy H(probs, softmax(logits)) without forming log_softmax.
    Works with NumPy or CuPy (if HAS_CUPY and inputs are cupy arrays).
    """

    # --- backend dispatch (avoid needless copies) ---
    nplib = np
    to_array_fn = to_numpy
    logsumexp_fn = logsumexp
    if HAS_CUPY and (isinstance(logits, cp.ndarray) or (probs is not None and isinstance(probs, cp.ndarray))):
        nplib = cp
        to_array_fn = to_cupy
        logsumexp_fn = cu_logsumexp

    # Only convert if needed (prevents extra allocations)
    if not isinstance(logits, nplib.ndarray):
        logits = to_array_fn(logits)
    if probs is not None and not isinstance(probs, nplib.ndarray):
        probs = to_array_fn(probs)

    if probs is not None:
        assert probs.shape == logits.shape, f'{probs.shape} and {logits.shape} do not have the same size.'

    # --- stable log-sum-exp ---
    # lse = logsumexp(logits, axis, keepdims=True) computed stably via shift-by-max
    max_logits = nplib.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_logits
    # logsumexp(shifted) is safe because shifted <= 0
    lse_shifted = logsumexp_fn(shifted, axis=axis, keepdims=True)
    lse = max_logits + lse_shifted  # same shape as max_logits

    if probs is None:
        # H(softmax(logits)) = lse - sum(p * logits)
        # Compute sum(p*logits) without materializing p:
        #   p ∝ exp(shifted); denom = sum(exp(shifted))
        #   sum(p*logits) = sum(exp(shifted)*logits)/denom
        w = nplib.exp(shifted)
        denom = nplib.sum(w, axis=axis, keepdims=True)
        num = nplib.sum(w * logits, axis=axis, keepdims=True)
        expected_logits = num / denom
        out = (lse - expected_logits)
    else:
        # Cross-entropy H(probs, softmax(logits)) = sum(probs)*(lse) - sum(probs*logits)
        # (no need to form log_softmax)
        sum_p = nplib.sum(probs, axis=axis, keepdims=True)
        dot_p_logit = nplib.sum(probs * logits, axis=axis, keepdims=True)
        out = sum_p * lse - dot_p_logit

    # Final shape
    if not keepdims and axis is not None:
        out = nplib.squeeze(out, axis=axis)

    return out


def entropy[T: NPFloating](
        probs: NDArray[T],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
) -> NDArray[T]:
    valid = (probs > 0)

    nplib = np
    if HAS_CUPY and isinstance(probs, cp.ndarray):
        nplib = cp

    if nplib.all(valid):
        h = -nplib.multiply(probs, nplib.log(probs))
    else:
        h = nplib.zeros_like(probs)
        valid_probs = probs[valid]
        h[valid] = - nplib.multiply(valid_probs, nplib.log(valid_probs))
    return nplib.sum(h, axis=axis, keepdims=keepdims)

def binary_entropy_with_logits[T: NPFloating](
    logits: NDArray[T],
    probs: Optional[NDArray[T]] = None,
) -> NDArray[T]:
    """
    If probs is None: returns H(σ(logits))  (entropy of Bernoulli with logit = logits)
    If probs is given: returns cross-entropy H(probs, σ(logits))
    Optional:
      - pos_weight: scales the positive-class contribution; accepts scalar/array (broadcastable)
      - weight: elementwise weight; accepts scalar/array (broadcastable)
    Works with NumPy or CuPy transparently.
    """
    # --- backend dispatch (avoid needless copies) ---
    nplib = np
    to_array_fn = to_numpy
    if HAS_CUPY and (
            isinstance(logits, cp.ndarray)
            or (probs is not None and isinstance(probs, cp.ndarray))
    ):
        nplib = cp
        to_array_fn = to_cupy

    # Convert only if needed
    if not isinstance(logits, nplib.ndarray):
        logits = to_array_fn(logits)
    if probs is not None and not isinstance(probs, nplib.ndarray):
        probs = to_array_fn(probs)

    # Validate shapes if targets provided
    if probs is not None:
        assert probs.shape == logits.shape, f'{probs.shape} and {logits.shape} do not have the same size.'

    # --- numerically stable softplus(x) = relu(x) + log1p(exp(-|x|)) ---
    absx = nplib.abs(logits)
    relu_x = nplib.maximum(logits, 0)
    softplus_x = relu_x + nplib.log1p(nplib.exp(-absx))

    # --- y term (targets): provided or σ(x) computed stably ---
    if probs is None:
        # Stable sigmoid: avoid overflow for large |x|
        # σ(x) = 1 / (1 + exp(-x)) for x>=0; = exp(x) / (1 + exp(x)) for x<0
        exp_neg = nplib.exp(-logits)
        exp_pos = nplib.exp(logits)
        y = nplib.where(logits >= 0, 1.0 / (1.0 + exp_neg), exp_pos / (1.0 + exp_pos))
    else:
        y = probs

    # Base BCE / entropy: softplus(x) - x*y
    loss = softplus_x - logits * y

    return loss

def binary_entropy[T: NPFloating](probs: NDArray[T]) -> NDArray[T]:
    valid = (probs > 0) & (probs < 1)

    nplib = np
    if HAS_CUPY and isinstance(probs, cp.ndarray):
        nplib = cp

    if nplib.all(valid):
        return (
                -nplib.multiply(probs, nplib.log(probs))
                - nplib.multiply(1 - probs, nplib.log(1 - probs))
        )

    h = nplib.zeros_like(probs)
    valid_probs = probs[valid]
    h[valid] = (
            - nplib.multiply(valid_probs, nplib.log(valid_probs))
            - nplib.multiply(1 - valid_probs, nplib.log(1 - valid_probs))
    )
    return h


def calc_metrics_from_stat_scores(
        stat_scores: NDArray[NPGeneric],
        metric: Union[StatMetric, Sequence[StatMetric]],
        ignore_class: Optional[Union[int, Sequence[int], Set[int]]] = None,
        batched: bool = False,
) -> Tuple[
    Union[Dict[int, NDArray[NPGeneric]], Dict[StatMetric, Dict[int, NDArray[NPGeneric]]]],
    Union[NDArray[NPGeneric], Dict[StatMetric, NDArray[NPGeneric]]],
    Union[NDArray[NPGeneric], Dict[StatMetric, NDArray[NPGeneric]]],
    Union[NDArray[NPGeneric], Dict[StatMetric, NDArray[NPGeneric]]],
]:
    if batched:
        # state_scores: (N, C, 5)
        assert stat_scores.ndim == 3, f'{stat_scores.shape=}'
    else:
        # state_scores: (C, 5)
        assert stat_scores.ndim == 2, f'{stat_scores.shape=}'

    assert stat_scores.shape[-1] == 5, f'{stat_scores.shape=}'

    if isinstance(metric, str):
        metric = {metric, }
    assert len(metric) > 0
    metric = set(metric)

    n_classes, _ = stat_scores.shape
    if ignore_class is None:
        include_classes = list(range(stat_scores.shape[0]))
    else:
        if isinstance(ignore_class, int):
            ignore_class = {ignore_class, }
        elif isinstance(ignore_class, Sequence):
            ignore_class = set(ignore_class)
        else:
            assert isinstance(ignore_class, set), \
                f'{ignore_class} {type(ignore_class)}'
        include_classes = sorted(list(set(range(n_classes)) - ignore_class))
        assert len(include_classes) > 0, f'{n_classes=} {ignore_class=}'
        stat_scores = stat_scores[include_classes]

    nplib = np
    if HAS_CUPY and isinstance(metric, cp.ndarray):
        nplib = cp

    # (C, 5) -> (C + 1, 5) or (N, C, 5) -> (N, C + 1, 5)
    stat_scores = nplib.concatenate((
        stat_scores, nplib.sum(stat_scores, axis=-2, keepdims=True)),
        axis=-2
    )

    # (C + 1) or (N, C + 1)
    tp, fp, tn, fn, sup = (
        x.squeeze(-1)
        for x in nplib.split(stat_scores, axis=-1, indices_or_sections=5,)
    )
    total = tp + fp + tn + fn
    all_tn = total == tn
    no_tp = tp == 0
    require_calc = nplib.logical_not(
        nplib.logical_or(all_tn, no_tp))

    def _calc_a_metric(numerator, denominator):
        if nplib.all(require_calc):
            return nplib.divide(numerator, denominator)
        m = nplib.empty_like(denominator, dtype=np.float64)
        m[all_tn] = 1
        m[no_tp] = 0
        m[require_calc] = nplib.divide(
            numerator[require_calc].astype(np.float64),
            denominator[require_calc].astype(np.float64)
        )
        return m

    individual_res = {}
    micro_res = {}
    macro_res = {}
    weighted_avg_res = {}

    metric_res = {}
    while len(metric) > 0:
        tag = metric.pop()
        if tag == 'dice':
            vals = _calc_a_metric(2 * tp, 2 * tp + fp + fn)
        elif tag == 'jaccard':
            vals = _calc_a_metric(tp, tp + fp + fn)
        elif tag == 'precision':
            vals = _calc_a_metric( tp, tp + fp)
        elif tag == 'recall':
            vals = _calc_a_metric(
                tp, tp + fp)
        elif tag == 'accuracy':
            vals = nplib.divide(tp + tn, total)
        else:
            raise NotImplementedError(f'{tag=}')

        metric_res[tag] = vals

    weights = nplib.divide(
        sup[..., :-1],
        nplib.sum(sup[..., :-1], axis=-1, keepdims=True),
    )

    tag = None
    for tag, vals in metric_res.items():
        # {class_id: (1,) or (N,)}
        individual_res[tag] = {
            c: vals[..., i] for i, c in enumerate(include_classes)
        }
        # (1, ) or (N,)
        micro_res[tag] = vals[..., -1]
        # (1, ) or (N,)
        macro_res[tag] = nplib.mean(
            vals[..., :-1], axis=-1, keepdims=False)
        # (1, ) or (N,)
        weighted_avg_res[tag] = nplib.sum(
            nplib.multiply(vals[..., :-1], weights),
            axis=-1, keepdims=False,
        )
    if len(macro_res) == 1:
        assert tag is not None
        individual_res = individual_res[tag]
        micro_res = micro_res[tag]
        macro_res = macro_res[tag]
        weighted_avg_res = weighted_avg_res[tag]
    return individual_res, micro_res, macro_res, weighted_avg_res