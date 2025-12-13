#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import logging
from typing import Literal, Any, Optional, Union, TypeAlias, Dict, List, Sequence
from dataclasses import dataclass
from pathlib import Path
import functools

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import onnx
from sklearn.tests.test_multiclass import n_classes

from .data_normalizer import DataNormalizer
from ..xutils import os_utils, array_utils as xp, image_utils
from ..xutils.lib_utils import import_available
from ..tqdm import tqdm

cp = None
cpt = None
if HAS_CUPY := import_available('cupy'):
    import cupy as cp
    import cupy.typing as cpt

NPIntOrFloat: TypeAlias = Union[np.floating, np.integer]

STRING_TO_ONNX_DTYPE_MAP = {
    'float64': onnx.TensorProto.DOUBLE,
    'float32': onnx.TensorProto.FLOAT,
    'int64': onnx.TensorProto.INT64,
    'int32': onnx.TensorProto.INT32,
    'int16': onnx.TensorProto.INT16,
}

_logger = logging.getLogger(__name__)

# Preload necessary DLLs
ort.preload_dlls()

_logger.info(f'ONNX Runtime device: {ort.get_device()}.')
_logger.info(f'Available execution providers: {ort.get_available_providers()}.')

@dataclass
class ModelData:
    session: ort.InferenceSession
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    data_norm: DataNormalizer


class Inferencer:
    @staticmethod
    def get_model(
            model_path: Path,
            norm_config_path: Path,
            in_shape: tuple[int, ...],
            out_shape: tuple[int, ...],
            onnx_providers: Sequence[str | tuple[str, dict[Any, Any]]] | None,
    ):
        so = ort.SessionOptions()
        so.intra_op_num_threads = os_utils.get_max_n_worker()
        _logger.info(f'Load ONNX model from {model_path} .')
        session = ort.InferenceSession(
            # onnx_model.SerializeToString(),
            str(model_path.resolve()),
            sess_options=so,
            providers=onnx_providers,
        )
        _logger.info(f'Load normalization config from {norm_config_path} .')
        data_norm = DataNormalizer.from_config_file(norm_config_path)
        return ModelData(
            session=session,
            in_shape=in_shape,
            out_shape=out_shape,
            data_norm=data_norm,
        )

    @staticmethod
    def ct_inference_proxy(
            image: xp.TypeArrayLike[NPIntOrFloat],
            model_data: ModelData,
            batch_size: int = 1,
            process_dtype: Literal['float32', 'float64'] = 'float64',
            prepro_device: Literal['cpu', 'cuda', 'auto'] = 'auto',
            progress_bar=True,
            progress_desc: str = '',
    ):
        sess = model_data.session
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        if prepro_device == 'auto':
            if (
                    xp.HAS_CUPY
                    and xp.CUPY_CUDA_AVAILABLE
                    and ('CUDAExecutionProvider' in sess.get_providers())
            ):
                prepro_device = 'cuda'
            else:
                prepro_device = 'cpu'

        if prepro_device == 'cpu':
            def bind_in(_iob: ort.IOBinding, _x: npt.NDArray, ):
                _iob.bind_cpu_input(input_name, _x)

            def bind_out(_iob: ort.IOBinding, _x: cpt.NDArray):
                _iob.bind_output(
                    name=output_name,
                    device_type='cpu',
                    device_id=-1,
                    element_type=_x.dtype,
                    shape=_x.shape,
                    buffer_ptr=_x.ctypes.data,
                )

            to_array = xp.to_numpy
            empty_fn = np.empty
            argmax_fn = np.argmax
        elif prepro_device == 'cuda':
            def bind_in(
                    _iob: ort.IOBinding,
                    _x: xp.TypeArrayLike[NPIntOrFloat],
            ):
                _iob.bind_input(
                    name=input_name,
                    device_type='cuda',
                    device_id=_x.device.id,
                    element_type=_x.dtype,
                    shape=_x.shape,
                    buffer_ptr=_x.data.ptr,
                )
            def bind_out(
                    _iob: ort.IOBinding,
                    _x: xp.TypeArrayLike[NPIntOrFloat],
            ):
                _iob.bind_output(
                    name=output_name,
                    device_type='cuda',
                    device_id=_x.device.id,
                    element_type=_x.dtype,
                    shape=_x.shape,
                    buffer_ptr=_x.data.ptr,
                )
            to_array = xp.to_cupy
            empty_fn = cp.empty
            argmax_fn = cp.argmax
        else:
            raise ValueError(f'Unknown device {prepro_device}')

        # (C, H, W)
        model_out_shape = sess.get_outputs()[0].shape
        model_H, model_W = model_out_shape[-2:]

        n_samples, H, W = image.shape
        pred_labelmap = np.empty((n_samples, H, W), dtype=np.int16)
        n_batches = (n_samples + batch_size - 1) // batch_size

        if (H, W) != (model_H, model_W):
            resize_fn = functools.partial(
                image_utils.resize,
                output_shape=(model_H, model_W), order=1, batched=True,
            )
            def restore_fn(_image):
                _image = xp.einsum(
                    'bchw->bhwc', _image)
                _image = image_utils.resize(
                    _image, (H, W), order=1, batched=True)
                _image = xp.einsum(
                    'bhwc->bchw', _image)
                return _image
        else:
            resize_fn = lambda x: x
            restore_fn = lambda x: x

        iterator = range(n_batches)
        if progress_bar:
            iterator = tqdm(iterator, desc=progress_desc, total=n_batches)


        for i in iterator:
            start = i * batch_size
            end = start + batch_size
            # (batch_size, H, W)
            batch_slice = image[start: end, ...]
            batch_slice = to_array(batch_slice, dtype=process_dtype)

            batch_slice = resize_fn(batch_slice).copy()

            batch_slice = model_data.data_norm(
                batch_slice.astype(process_dtype, copy=False)
            ).astype(np.float32, copy=False)

            batch_slice = batch_slice[:, None].copy()
            pred_logits = empty_fn(
                (len(batch_slice), *model_data.out_shape), dtype=np.float32,
            )
            io_binding = sess.io_binding()
            bind_in(io_binding, batch_slice)
            bind_out(io_binding, pred_logits)
            sess.run_with_iobinding(io_binding)

            # (B, C, H, W)
            pred_logits = restore_fn(pred_logits)
            # (B, H, W)
            batch_labelmap = argmax_fn(pred_logits, axis=1)
            del pred_logits
            pred_labelmap[start: end] = xp.to_numpy(batch_labelmap, dtype=np.int16)

            batch_slice = xp.to(batch_slice, dtype=process_dtype)
            # calculate mean hu etc.

            batch_labelmap = xp.one_hot(batch_labelmap, n_classes=n_classes)

            del batch_slice, batch_labelmap

        return pred_labelmap

