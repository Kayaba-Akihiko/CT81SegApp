#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path
import json
from abc import ABC
from typing import TypeVar, TypeAlias, Union

import numpy as np
import numpy.typing as npt

from ..xutils.lib_utils import import_available
from ..xutils import array_utils as xp

cpt = None
if HAS_CUPY := import_available('cupy'):
    import cupy.typing as cpt

if HAS_TORCH := import_available('torch'):
    import torch

_NPDType = TypeVar('_NPDType', bound=np.generic)
TypeArray: TypeAlias = xp.TypeArrayLike[_NPDType]

class DataNormalizer:
    def __init__(
            self,
            config: dict,
    ):
        super().__init__()
        ops = []

        op_name_l_map = {
            k.lower(): k for k in config.keys()
        }

        if 'clip' in op_name_l_map:
            # Masaki's clip operation that does rescaling to [0, 1] range
            param = config[op_name_l_map['clip']]
            param_param = param.get('param', None)
            min_val = param.get('min', None)
            max_val = param.get('max', None)

            if param_param is not None:
                if min_val is not None or max_val is not None:
                    raise ValueError(
                        'If "param" is specified, "min" and "max" should not be specified.'
                    )
            if min_val is not None or max_val is not None:
                if param_param is not None:
                    raise ValueError(
                        'If "min" or "max" is specified, "param" should not be specified.'
                    )

            if param_param is not None:
                min_val, max_val = param_param
            else:
                min_val, max_val = None, None
            rescale, clip = (
                param.get('rescale', True), param.get('clip', True))
            ops.append(
                MinMaxNormalizer(
                    min_val, max_val, rescale=rescale, clip=clip)
            )
        if 'minmax' in op_name_l_map:
            param = config[op_name_l_map['minmax']]
            min_val = param.get('min', None)
            max_val = param.get('max', None)
            rescale, clip = (
                param.get('rescale', True), param.get('clip', True))
            ops.append(MinMaxNormalizer(
                min_val, max_val, rescale=rescale, clip=clip)
            )
        if 'z' in op_name_l_map:
            param = config[op_name_l_map['z']]
            mean_val = param.get('mean', None)
            std_val = param.get('std', None)
            ops.append(ZNormalizer(mean_val, std_val))
        if 'quantize' in op_name_l_map:
            param = config[op_name_l_map['quantize']]
            n_bits = param['n_bit']
            x_min = param['x_min']
            x_max = param['x_max']
            ops.append(QuantizationNormalizer(n_bits, x_min, x_max))
        if 'shift' in op_name_l_map:
            param = config[op_name_l_map['shift']]
            shift_value = param['value']
            ops.append(ShiftNormalizer(shift_value))
        if 'rescale' in op_name_l_map:
            param = config[op_name_l_map['rescale']]
            rescale_factor = param['factor']
            ops.append(RescaleNormalizer(rescale_factor))
        self.ops = ops

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

    @classmethod
    def from_config_file(
            cls, config_path: Path):
        with config_path.open('rb') as f:
            config = json.load(f)
        return cls(config)

class _MinMaxZNormalizerBase(ABC):
    @staticmethod
    def _normalize(x: TypeArray[np.floating], numerator, denominator):
        if denominator == 0:
            return xp.zeros_like(x)
        return (x - numerator) / denominator


class MinMaxNormalizer(_MinMaxZNormalizerBase):
    def __init__(
            self, min_val: float = None, max_val: float = None,
            rescale=True, clip=True,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        if max_val is not None and min_val is not None:
            assert max_val >= min_val, \
                f'Max value {max_val} must be greater than or equal to min value {min_val}.'
        if not rescale and not clip:
            raise ValueError(
                'Rescale and clip cannot both be False. '
            )
        self.rescale = rescale
        self.clip = clip

    def __call__(self, x):
        if self.min_val is None:
            min_val = x.min()
        else:
            min_val = self.min_val
        if self.max_val is None:
            max_val = x.max()
        else:
            max_val = self.max_val
        if self.rescale:
            denominator = max_val - min_val
            numerator = min_val
            x = self._normalize(x, numerator, denominator)
            if self.clip:
                # Normalize first then clip to ensurer floating range.
                x = xp.clip(x, min=0., max=1., out=x)
        else:
            if self.clip:
                x = xp.copy(x)
                x = xp.clip(x, min=min_val, max=max_val, out=x)
            else:
                raise RuntimeError(
                    'Both rescale and clip are set to False.')
        return x

class ZNormalizer(_MinMaxZNormalizerBase):
    def __init__(self, mean_val = None, std_val = None):
        super().__init__()
        self.mean_val = mean_val
        self.std_val = std_val
        if std_val is not None and std_val <= 0:
            raise ValueError(f'Standard deviation must be greater than 0, got {std_val}.')

    def __call__(self, x):
        if self.mean_val is None:
            mean_val = x.mean()
        else:
            mean_val = self.mean_val
        if self.std_val is None:
            std_val = xp.std(x)
        else:
            std_val = self.std_val
        x = self._normalize(x, mean_val, std_val)
        return x


class QuantizationNormalizer:
    def __init__(self, n_bits=8, x_min=0., x_max=1.):
        super().__init__()
        if n_bits == 256:
            self.enabled = False
            return
        self.enabled = True

        self.n_bits = n_bits
        self.x_min = x_min
        self.x_max = x_max
        self.n_levels = 1 << self.n_bits  # 2**n_bits
        self.inv_step = (self.n_levels - 1) / (self.x_max - self.x_min)  # scale
        self.step = 1.0 / self.inv_step  # quantization step in x-units

    def __call__(self, x):
        if not self.enabled:
            return x
        # clamp to valid range before quantizing (optional but common)
        x_clamped = xp.clip(x, self.x_min, self.x_max)
        # quantize indices (integer bins)
        q = xp.round((x_clamped - self.x_min) * self.inv_step)  # round to nearest level
        q = xp.clip(q, 0, self.n_levels - 1, out=q)
        # dequantize back to x-domain
        xq = q * self.step + self.x_min
        return xq


class RescaleNormalizer:
    def __init__(self, rescale_factor: float):
        self.rescale_factor = rescale_factor

    def __call__(self, x):
        return x * self.rescale_factor


class ShiftNormalizer:
    def __init__(self, shift_value: float):
        self.shift_value = shift_value

    def __call__(self, x):
        return x + self.shift_value