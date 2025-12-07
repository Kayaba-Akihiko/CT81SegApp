#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from itertools import repeat
from typing import (
    Any, Callable, Iterable, Tuple, Sequence, TypeVar, Union, List, Dict
)
import math

import random
import logging

import numpy as np
import numpy.typing as npt

_logger = logging.getLogger(__name__)

def _ntuple(n) -> Callable[[int], Tuple[Any, ...]]:
    def parse(x) -> Tuple[Any, ...]:
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def sample_n_items[T](
    container: Sequence[T],
    n_samples: int,
    return_indexes: bool = False,
) -> Union[List[T], Tuple[List[T], List[int]]]:
    assert n_samples > 0, f"{n_samples}"
    assert n_samples <= len(container), "n_samples cannot exceed container length"

    sampled_indexes = random.sample(range(len(container)), n_samples)
    sampled_items = [container[i] for i in sampled_indexes]

    if return_indexes:
        return sampled_items, sampled_indexes
    return sampled_items

def update_dict(target: Dict, update: Dict) -> Dict:
    overrides = target.keys() & update.keys()
    if overrides:
        _logger.info(
            "Overriding keys: " +
            ", ".join(f"{k} ({target[k]} → {update[k]})" for k in overrides)
        )

    target.update(update)
    return target

def calc_sampling_weights(
    labels: Sequence[int], balanced: bool
) -> npt.NDArray[np.float64]:
    n = len(labels)
    if n == 0:
        raise ValueError("labels must be non-empty")

    # Uniform weights by default
    base = np.full(n, 1.0 / n, dtype=np.float64)
    if not balanced:
        return base

    arr = np.asarray(labels)
    classes, inv, counts = np.unique(
        arr, return_inverse=True, return_counts=True)
    n_class = classes.size
    class_weight = 1.0 / n_class

    # Per-sample weights: (1 / n_class) / count[class_of_sample]
    weights = (class_weight / counts[inv]).astype(np.float64, copy=False)

    # Consolidated, informative logging (once per class)
    for c, cnt in zip(classes, counts):
        _logger.info(
            f"Class {c}: count={cnt}, "
            f"orig class proportion={cnt / n:.6f} → balanced class weight={class_weight:.6f}; "
            f"orig per-sample={1.0 / n:.6f} → balanced per-sample={class_weight / cnt:.6f}"
        )

    return weights


def split[T: Sequence[Any]](container: T, n_splits: int) -> Tuple[T, ...]:
    if n_splits <= 0:
        raise ValueError(f"n_splits={n_splits} must be positive")
    if (n_samples := len(container)) == 0:
        return tuple([] for _ in range(n_splits))
    split_size = math.ceil(n_samples / n_splits)
    out = []
    for i in range(n_splits):
        start = i * split_size
        end = start + split_size
        out.append(container[start:end] if start < n_samples else [])
    return tuple(out)