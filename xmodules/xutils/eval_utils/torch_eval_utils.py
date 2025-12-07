#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import (
    Union, Optional, Tuple, TypeAlias, TypeVar, Literal, Sequence, Set, Dict
)
import math

from scipy.stats import norm  # for z-score
import torch
from torch import Tensor
import torch.nn.functional as F

StatMetric: TypeAlias = Literal[
    'dice', 'jaccard', 'accuracy', 'precision', 'recall', 'specificity']

def pearson_correlation(
        x: Tensor, y: Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
        eps: float = 1e-12
) -> Tensor:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')

    x_mean = torch.mean(x, dim=dim, keepdim=True)
    y_mean = torch.mean(y, dim=dim, keepdim=True)
    x_z_mean = x - x_mean
    y_z_mean = y - y_mean
    x_z_mean_norm = torch.linalg.norm(x_z_mean, dim=dim, keepdim=True)
    y_z_mean_norm = torch.linalg.norm(y_z_mean, dim=dim, keepdim=True)
    denominator = torch.mul(x_z_mean_norm, y_z_mean_norm)
    corr = torch.sum(
        torch.div(
            torch.mul(x_z_mean, y_z_mean),
            denominator + eps,
        ),
        dim=dim, keepdim=keepdim
    )
    return corr


def spearman_correlation(
        x: Tensor, y: Tensor,
        dim: Optional[int] = None,
        keepdim: bool = False,
        eps: float = 1e-12
) -> Tensor:
    raise NotImplementedError


def intra_class_correlation(
        x: Tensor, y: Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
        eps: float = 1e-12
) -> Tensor:
    if x.shape != y.shape:
        raise RuntimeError(
            f'{x.shape} and {y.shape} do not have the same size.')

    x_mean = torch.mean(x, dim=dim, keepdim=True)
    y_mean = torch.mean(y, dim=dim, keepdim=True)
    global_mean = (x_mean + y_mean) / 2.
    x_zero_mean = x - global_mean
    y_zero_mean = y - global_mean

    denom_t1 = torch.sum(
        torch.square(x_zero_mean), dim=dim, keepdim=True)
    denom_t2 = torch.sum(
        torch.square(y_zero_mean), dim=dim, keepdim=True)
    denorm = denom_t1 + denom_t2
    corr = torch.sum(
        torch.divide(
            torch.multiply(x_zero_mean, y_zero_mean),
            denorm + eps,
        ),
        dim=dim, keepdim=keepdim
    ) * 2.
    return corr


def intersection_and_sum(
        x: Tensor, y: Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim=False,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
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

    intersection = torch.sum(
        torch.mul(x, y), dim=dim, keepdim=keepdim)
    sum_ = (
            torch.sum(x, dim=dim, keepdim=keepdim)
            + torch.sum(y, dim=dim, keepdim=keepdim)
    )
    mask = (sum_ > 0.)
    return intersection, sum_, mask


def dice_score(
        x: Tensor, y: Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim=False,
) -> Tensor:
    intersection, sum_, mask = intersection_and_sum(
        x=x, y=y, dim=dim, keepdim=keepdim)

    if torch.all(mask):
        return torch.div(intersection * 2, sum_)

    dice = torch.ones_like(intersection)
    dice[mask] = torch.div(intersection[mask] * 2., sum_[mask])
    return dice

def signal_to_noise_ratio(
        x: Tensor, y: Tensor,
        r=255.,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim=False,
        eps=1e-12,
) -> Tensor:
    mse = torch.mean(
        torch.square(x / r - y / r), dim=dim, keepdim=keepdim)
    return 10 * torch.log10(1. / (mse + eps))


def gaussian_nll(
        x: Tensor, y: Tensor, var: Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim=False,
        full: bool = False,
        eps=1e-12,
) -> Tensor:
    if var.shape != x.shape:
        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if x.shape[:-1] == var.shape:
            var = torch.unsqueeze(var, -1)
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
    if torch.any(var < 0.):
        raise ValueError("var has negative entry/entries")

    var = var.detach().clone()
    torch.clamp_(var, min=eps, max=None)

    nll = 0.5 * (torch.log(var) + (x - y) ** 2 / var)
    if full:
        nll += 0.5 * math.log(2 * math.pi)
    nll = torch.mean(nll, dim=dim, keepdim=keepdim)
    return nll

def prediction_interval_coverage_probability(
        x: Tensor, y: Tensor, var: Tensor,
        alpha: float, # confidence level, e.g., 0.95 for 95%
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim=False,
        eps=1e-12,
) -> Tensor:
    if var.shape != x.shape:
        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if x.shape[:-1] == var.shape:
            var = torch.unsqueeze(var, -1)
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
    if torch.any(var < 0.):
        raise ValueError("var has negative entry/entries")

    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")

    var = var.detach().clone()
    torch.clamp_(var, min=eps, max=None)

    std = torch.sqrt(var)

    z = norm.ppf(0.5 + alpha / 2)
    # Compute prediction intervals
    lower = x - z * std
    upper = x + z * std
    within_interval = (y >= lower) & (y <= upper)
    picp = torch.mean(within_interval, dim=dim, keepdim=keepdim)
    return picp


def entropy_with_logits(
    logits: Tensor,
    probs: Optional[Tensor] = None,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Tensor:

    if probs is not None:
        assert probs.shape == logits.shape, f'{probs.shape} and {logits.shape} do not have the same size.'

    # --- stable log-sum-exp ---
    # lse = logsumexp(logits, axis, keepdims=True) computed stably via shift-by-max
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    shifted = logits - max_logits
    # logsumexp(shifted) is safe because shifted <= 0
    lse_shifted = torch.logsumexp(shifted, dim=dim, keepdim=True)
    lse = max_logits + lse_shifted  # same shape as max_logits

    if probs is None:
        # H(softmax(logits)) = lse - sum(p * logits)
        # Compute sum(p*logits) without materializing p:
        #   p ∝ exp(shifted); denom = sum(exp(shifted))
        #   sum(p*logits) = sum(exp(shifted)*logits)/denom
        w = torch.exp(shifted)
        denom = torch.sum(w, dim=dim, keepdim=True)
        num = torch.sum(w * logits, dim=dim, keepdim=True)
        expected_logits = num / denom
        out = (lse - expected_logits)
    else:
        # Cross-entropy H(probs, softmax(logits)) = sum(probs)*(lse) - sum(probs*logits)
        # (no need to form log_softmax)
        sum_p = torch.sum(probs, dim=dim, keepdim=True)
        dot_p_logit = torch.sum(probs * logits, dim=dim, keepdim=True)
        out = sum_p * lse - dot_p_logit

    # Final shape
    if not keepdim and dim is not None:
        out = torch.squeeze(out, dim=dim)

    return out


def entropy(
        probs: Tensor,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
) -> Tensor:
    valid = (probs > 0)

    if torch.all(valid):
        h = -torch.mul(probs, torch.log(probs))
    else:
        h = torch.zeros_like(probs)
        valid_probs = probs[valid]
        h[valid] = - torch.mul(valid_probs, torch.log(valid_probs))
    return torch.sum(h, dim=dim, keepdim=keepdim)


def binary_entropy_with_logits(
        logits: Tensor,
        probs: Optional[Tensor] = None,
) -> Tensor:
    if probs is None:
        probs = torch.sigmoid(logits)

    return F.binary_cross_entropy_with_logits(
        logits, probs, reduction='none')


def binary_entropy(probs: Tensor) -> Tensor:
    return F.binary_cross_entropy(
        probs, probs, reduction='none')


def calc_metrics_from_stat_scores(
        stat_scores: Tensor,
        metric: Union[StatMetric, Sequence[StatMetric]],
        ignore_class: Optional[Union[int, Sequence[int], Set[int]]] = None,
        batched: bool = False,
) -> Tuple[
    Union[Dict[int, Tensor], Dict[StatMetric, Dict[int, Tensor]]],
    Union[Tensor, Dict[StatMetric, Tensor]],
    Union[Tensor, Dict[StatMetric, Tensor]],
    Union[Tensor, Dict[StatMetric, Tensor]],
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

    # (C, 5) -> (C + 1, 5) or (N, C, 5) -> (N, C + 1, 5)
    stat_scores = torch.cat((
        stat_scores, torch.sum(stat_scores, dim=-2, keepdim=True)),
        dim=-2
    )

    # (C + 1) or (N, C + 1)
    tp, fp, tn, fn, sup = (
        x.squeeze(-1)
        for x in torch.chunk(stat_scores, dim=-1, chunks=5)
    )
    total = tp + fp + tn + fn
    all_tn = total == tn
    no_tp = tp == 0
    require_calc = torch.logical_not(
        torch.logical_or(all_tn, no_tp))

    def _calc_a_metric(numerator, denominator):
        if torch.all(require_calc):
            return torch.div(numerator, denominator)
        m = torch.empty_like(denominator, dtype=torch.float64)
        m[all_tn] = 1
        m[no_tp] = 0
        m[require_calc] = torch.div(
            numerator[require_calc].to(torch.float64),
            denominator[require_calc].to(torch.float64),
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
            vals = _calc_a_metric(tp, tp + fp)
        elif tag == 'recall':
            vals = _calc_a_metric(tp, tp + fn)
        elif tag == 'specificity':
            vals = _calc_a_metric(tn, tn + fp)
        elif tag == 'accuracy':
            vals = torch.div(tp + tn, total)
        else:
            raise NotImplementedError(f'{tag=}')

        metric_res[tag] = vals

    weights = torch.divide(
        sup[..., :-1],
        torch.sum(sup[..., :-1], dim=-1, keepdim=True),
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
        macro_res[tag] = torch.mean(
            vals[..., :-1], dim=-1, keepdim=False)
        # (1, ) or (N,)
        weighted_avg_res[tag] = torch.sum(
            torch.mul(vals[..., :-1], weights),
            dim=-1, keepdim=False,
        )
    if len(macro_res) == 1:
        assert tag is not None
        individual_res = individual_res[tag]
        micro_res = micro_res[tag]
        macro_res = macro_res[tag]
        weighted_avg_res = weighted_avg_res[tag]
    return individual_res, micro_res, macro_res, weighted_avg_res