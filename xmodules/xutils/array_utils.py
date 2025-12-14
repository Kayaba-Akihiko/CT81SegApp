#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import re
from typing import Union, Optional, TypeAlias, Literal, TypeVar, Sequence, Dict, Tuple, Type, List, overload
import copy as pycopy
import functools
from contextlib import nullcontext

import numpy as np
import numpy.typing as npt

from .lib_utils import import_available

torch = None
F = None
TORCH_CUDA_AVAILABLE = None
if HAS_TORCH := import_available('torch'):
    import torch
    import torch.nn.functional as F
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()

cp = None
cpt = None
CUPY_CUDA_AVAILABLE = None
if HAS_CUPY := import_available('cupy'):
    import cupy as cp
    import cupy.typing as cpt
    try:
        CUPY_CUDA_AVAILABLE = cp.cuda.is_available()
    except cp.cuda.runtime.CUDARuntimeError:
        CUPY_CUDA_AVAILABLE = False

NPGeneric: TypeAlias = np.generic
NPFloating: TypeAlias = np.floating
NPInteger: TypeAlias = np.integer
NPIntOrFloat: TypeAlias = Union[NPInteger, NPFloating]
_NPDType = TypeVar('_NPDType', bound=NPGeneric)
TypeArrayLike: TypeAlias = Union[
    npt.NDArray[_NPDType], 'torch.Tensor', 'cpt.NDArray[_NPDType]', Sequence]

TypeDTypeString: TypeAlias = Literal['float32', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'bool']
TypeDTypeLike: TypeAlias = Union[TypeDTypeString, npt.DTypeLike, 'torch.dtype']
TypeDeviceType: TypeAlias = Literal['cpu', 'cuda']
TypeDeviceIndex: TypeAlias = Union[int, None]
TypeDeviceString: TypeAlias = str
TypeDeviceLike: TypeAlias = Union[
    TypeDeviceType, TypeDeviceIndex, TypeDeviceString, 'torch.device', 'cp.cuda.Device']

TypeArrayBackend: TypeAlias = Literal['numpy', 'cupy', 'torch']
TypePadWidth: TypeAlias = Union[
    int, Sequence[int], Sequence[Sequence[int]], TypeArrayLike[NPInteger]]

DEVICE_RE = re.compile(r"^(cpu|cuda)(?::(\d+))?$")

STRING_TO_NUMPY_DTYPE_MAP: Dict[TypeDTypeString, Type[NPGeneric]] = {
    'float32': np.float32,
    'float64': np.float64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'bool': np.bool_,
}

NUMPY_DTYPE_TO_STRING_MAP = {
    v: k for k, v in STRING_TO_NUMPY_DTYPE_MAP.items()
}

TORCH_TO_NUMPY_DTYPE_MAP = None
NUMPY_TO_TORCH_DTYPE_MAP = None
TORCH_BLOCKING_DEVICES = None
torch_no_grad = None
if HAS_TORCH:
    TORCH_TO_NUMPY_DTYPE_MAP: Union[
        Dict[torch.dtype, Type[NPGeneric]],
        None,
    ] = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.uint16: np.uint16,
        torch.uint32: np.uint32,
        torch.uint64: np.uint64,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
    }
    NUMPY_TO_TORCH_DTYPE_MAP = {
        v: k for k, v in TORCH_TO_NUMPY_DTYPE_MAP.items()
    }

    STRING_TO_TORCH_DTYPE_MAP = {
        k: NUMPY_TO_TORCH_DTYPE_MAP[v] for k, v in STRING_TO_NUMPY_DTYPE_MAP.items()
    }

    TORCH_DTYPE_TO_STRING_MAP = {
        v: k for k, v in STRING_TO_TORCH_DTYPE_MAP.items()
    }

    TORCH_BLOCKING_DEVICES = {'cpu', 'mps'}

    torch_no_grad = torch.no_grad
else:
    # --- Define a fallback that works both as decorator and context manager ---
    class _DummyNoGrad:
        """A dummy replacement for torch.no_grad that supports both
        decorator and context manager syntax.
        """

        def __call__(self, func=None):
            # Used as a decorator
            if func is None:
                return lambda f: f
            return func

        def __enter__(self):
            # Used as a context manager
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # No-op for context exit
            return False
    torch_no_grad = _DummyNoGrad()


class ArrayUtils:

    @staticmethod
    def is_array(array: TypeArrayLike) -> bool:
        if isinstance(array, np.ndarray):
            return True
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return True
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            return True
        return False


    @staticmethod
    @overload
    def _convert_dtype(
            dtype: TypeDTypeLike, to: Literal['numpy']
    ) -> Type[NPGeneric]: ...

    @staticmethod
    @overload
    def _convert_dtype(
            dtype: TypeDTypeLike, to: Literal['torch']
    ) -> 'torch.dtype': ...

    @staticmethod
    @overload
    def _convert_dtype(
            dtype: TypeDTypeLike, to: Literal['string']
    ) -> TypeDTypeString: ...

    @staticmethod
    def _convert_dtype(
            dtype: TypeDTypeLike,
            to: Literal['numpy', 'torch', 'string']
    ) -> Union[Type[NPGeneric], 'torch.dtype', TypeDTypeString]:
        if isinstance(dtype, str):
            if to == 'numpy':
                conv_map = STRING_TO_NUMPY_DTYPE_MAP
            elif to == 'torch':
                conv_map = STRING_TO_TORCH_DTYPE_MAP
            elif to == 'string':
                conv_map = {dtype: dtype}
            else:
                raise ValueError(
                    f"Unknown conversion target: {to}. "
                    f"Expected 'numpy', 'torch', or 'string'."
                )
        elif (
                isinstance(dtype, np.dtype)
                or (
                        isinstance(dtype, type)
                        and issubclass(dtype, np.generic)
                )
        ):
            dtype = np.dtype(dtype).type  # Ensure it's a numpy dtype
            if to == 'numpy':
                conv_map = {dtype: dtype}
            elif to == 'torch':
                conv_map = NUMPY_TO_TORCH_DTYPE_MAP
            elif to == 'string':
                conv_map = NUMPY_DTYPE_TO_STRING_MAP
            else:
                raise ValueError(
                    f"Unknown conversion target: {to}. "
                    f"Expected 'numpy', 'torch', or 'string'."
                )
        elif HAS_TORCH and isinstance(dtype, torch.dtype):
            if to == 'numpy':
                conv_map = TORCH_TO_NUMPY_DTYPE_MAP
            elif to == 'torch':
                conv_map = {dtype: dtype}
            elif to == 'string':
                conv_map = TORCH_DTYPE_TO_STRING_MAP
            else:
                raise ValueError(
                    f"Unknown conversion target: {to}. "
                    f"Expected 'numpy', 'torch', or 'string'."
                )
        else:
            raise TypeError(
                f"Unsupported dtype type: {type(dtype)}. "
                "Expected str, np.dtype, or torch.dtype."
            )
        return conv_map[dtype]

    @classmethod
    def convert_dtype_to_string(
            cls, dtype: TypeDTypeLike,
    ) -> str:
        return cls._convert_dtype(dtype, to='string')

    @classmethod
    def convert_dtype_to_numpy(
            cls, dtype: TypeDTypeLike,
    ) -> Type[NPGeneric]:
        return cls._convert_dtype(dtype, to='numpy')

    @classmethod
    def convert_dtype_to_torch(
            cls, dtype: TypeDTypeLike,
    ) -> 'torch.dtype':
        return cls._convert_dtype(dtype, to='torch')

    @staticmethod
    @functools.lru_cache()
    def is_cuda_available() -> bool:
        if HAS_CUPY and CUPY_CUDA_AVAILABLE:
            return True
        elif HAS_TORCH and TORCH_CUDA_AVAILABLE:
            return True
        return False

    @functools.singledispatch
    @staticmethod
    def parse_device(
            device: TypeDeviceLike,
    ) -> Tuple[TypeDeviceType, TypeDeviceIndex, TypeDeviceString]:
        dev_type: TypeDeviceType
        dev_id: TypeDeviceIndex
        if HAS_CUPY and isinstance(device, cp.cuda.Device):
            dev_type = "cuda"
            dev_id_str = str(device.id)
            dev_id = device.id
        elif HAS_TORCH and isinstance(device, torch.device):
            if device.type not in {'cpu', 'cuda'}:
                raise ValueError(f"Unsupported device type: {device.type}")
            dev_type = device.type
            dev_id = device.index
            if dev_id is None:
                dev_id = -1
            dev_id_str = str(dev_id)
        else:
            raise TypeError(f"Unsupported device type: {type(device)}.")
        if dev_id is None or dev_id < 0:
            dev_string = dev_type
        else:
            dev_string = f"{dev_type}:{dev_id_str}"
        return dev_type, dev_id, dev_string

    @parse_device.register
    @staticmethod
    def _(device: int) -> Tuple[TypeDeviceType, TypeDeviceIndex, TypeDeviceString]:
        dev_id: TypeDeviceIndex
        dev_type: TypeDeviceType
        if device < -1:
            raise ValueError(f"Invalid device index: {device}")
        if device == -1:
            dev_type = "cpu"
            dev_id_str = "-1"
            dev_id = -1
        else:
            dev_type = "cuda"
            dev_id_str = str(device)
            dev_id = device
        if dev_id is None or dev_id < 0:
            dev_string = dev_type
        else:
            dev_string = f"{dev_type}:{dev_id_str}"
        return dev_type, dev_id, dev_string

    @parse_device.register
    @staticmethod
    def _(device: str) -> Tuple[TypeDeviceType, TypeDeviceIndex, TypeDeviceString]:
        dev_id: TypeDeviceIndex
        dev_type: TypeDeviceType
        s = device.strip().lower()
        m = DEVICE_RE.fullmatch(s)
        if not m:
            raise ValueError(f"Unrecognized device string: '{device}'")
        dev_type, dev_id_str = m.group(1), m.group(2)
        # CPU must NOT have an index
        if dev_type == "cpu":
            if dev_id_str is not None:
                raise ValueError(f"CPU device cannot have index: '{device}'")
            dev_id = -1
        elif dev_type == "cuda":
            # CUDA rules
            if dev_id_str is None:
                dev_id = None
            else:
                dev_id = int(dev_id_str)
        else:
            # Should never reach here since regex restricts device types
            raise ValueError(f"Unknown device type parsed: '{dev_type}'")
        if dev_id is None or dev_id < 0:
            dev_string = dev_type
        else:
            dev_string = f"{dev_type}:{dev_id_str}"
        return dev_type, dev_id, dev_string

    @staticmethod
    def get_backend_n_device(
            x: TypeArrayLike
    ) -> Tuple[TypeArrayBackend, TypeDeviceLike]:
        if isinstance(x, np.ndarray):
            return 'numpy', 'cpu'
        elif HAS_CUPY and isinstance(x, cp.ndarray):
            return 'cupy', x.device
        elif HAS_TORCH and isinstance(x, torch.Tensor):
            return 'torch', x.device
        else:
            raise TypeError(
                f"Unsupported array type: {type(x)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )

    @classmethod
    @overload
    def _convert_device(
            cls, device: TypeDeviceLike, to: Literal['type'],
    ) -> TypeDeviceType: ...

    @classmethod
    @overload
    def _convert_device(
            cls, device: TypeDeviceLike, to: Literal['index'],
    ) -> TypeDeviceIndex: ...

    @classmethod
    @overload
    def _convert_device(
            cls, device: TypeDeviceLike, to: Literal['string'],
    ) -> TypeDeviceString: ...

    @classmethod
    @overload
    def _convert_device(
            cls, device: TypeDeviceLike, to: Literal['cupy'],
    ) -> 'cp.cuda.Device': ...

    @classmethod
    @overload
    def _convert_device(
            cls, device: TypeDeviceLike, to: Literal['torch'],
    ) -> 'torch.device': ...

    @classmethod
    def _convert_device(
            cls,
            device: TypeDeviceLike,
            to: Literal['type', 'index', 'string', 'cupy', 'torch'],
    ) -> Union[TypeDeviceLike]:
        device_type, device_id, device_string = cls.parse_device(device)
        if to == 'type':
            return device_type
        elif to == 'index':
            return device_id
        elif to == 'string':
            return device_string
        elif to == 'cupy':
            if not HAS_CUPY:
                raise RuntimeError("CUPY is not available.")
            if device_id is None:
                # Using default
               device_id = cp.cuda.runtime.getDevice()
            elif device_id < 0:
                raise ValueError(f"Invalid device index: {device_id}")
            return cp.cuda.Device(device_id)
        elif to == 'torch':
            if not HAS_TORCH:
                raise RuntimeError("PyTorch is not available.")
            return torch.device(device_string)
        else:
            raise ValueError(
                f"Unknown conversion target: {to}. "
                f"Expected 'string', 'int', or 'torch'."
            )

    @classmethod
    def convert_device_to_type(
            cls, device: TypeDeviceLike,
    ) -> TypeDeviceType:
        return cls._convert_device(device, to='type')

    @classmethod
    def convert_device_to_string(
            cls, device: TypeDeviceLike,
    ) -> TypeDeviceString:
        return cls._convert_device(device, to='string')

    @classmethod
    def convert_device_to_index(
            cls, device: TypeDeviceLike,
    ) -> TypeDeviceIndex:
        return cls._convert_device(device, to='index')

    @classmethod
    def convert_device_to_cupy(
            cls, device: TypeDeviceLike,
    ) -> 'cp.cuda.Device':
        return cls._convert_device(device, to='cupy')

    @classmethod
    def convert_device_to_torch(
            cls, device: TypeDeviceLike,
    ) -> 'torch.device':
        return cls._convert_device(device, to='torch')

    @classmethod
    def to_numpy(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> npt.NDArray[NPGeneric]:
        if dtype is not None:
            dtype = cls.convert_dtype_to_numpy(dtype)
        if device is not None:
            device_type, device_id, device_string = cls.parse_device(device)
            if device_type != 'cpu' or device_id != -1:
                raise ValueError(
                    f"array_to only supports device='cpu', got device='{device_string}'."
                )

        if HAS_TORCH and isinstance(array, torch.Tensor):
            array = np.asarray(
                array.detach().cpu().numpy(),
                dtype=dtype
            )
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            array = cp.asnumpy(array)
        elif isinstance(array, np.ndarray):
            if (
                    dtype is not None
                    and array.ndim > 0
                    and array.dtype.itemsize < np.dtype(dtype).itemsize
            ):
                array = np.ascontiguousarray(array)
        else:
            array = np.asarray(array)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    @classmethod
    def to_list(
            cls,
            array: TypeArrayLike,
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> List:
        if dtype is not None:
            dtype = cls.convert_dtype_to_numpy(dtype)

        if dtype is not None and dtype not in {
            'float64', 'int64'}:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                "Expected one of: 'float32', 'float64', 'int32', 'int64'."
            )

        if device is not None:
            device_type, device_id, device_string = cls.parse_device(device)
            if device_type != 'cpu' or device_id != -1:
                raise ValueError(
                    f"array_to only supports device='cpu', got device='{device_string}'."
                )

        if isinstance(array, np.ndarray):
            return array.tolist()
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            return array.tolist()
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.asnumpy(array).tolist()
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )

    @classmethod
    def to_cupy(
            cls,
            array: TypeArrayLike,
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> 'cpt.NDArray[NPGeneric]':
        if dtype is not None:
            dtype = cls.convert_dtype_to_numpy(dtype)

        device_context = nullcontext()
        if device is not None:
            device_type, device_id, device_string = cls.parse_device(device)
            if device_type != 'cuda':
                raise ValueError('The device must be a cuda device')
            if device_id is None:
                device_id = cp.cuda.runtime.getDevice()
            elif device_id < 0:
                raise ValueError(f"Invalid device index: {device_id}")
            device_context = cp.cuda.Device(device_id)

        if HAS_TORCH and isinstance(array, torch.Tensor) and array.device.type == 'cuda':
            # This is needed because of https://github.com/cupy/cupy/issues/7874#issuecomment-1727511030
            if array.dtype == torch.bool:
                array = array.detach().to(torch.uint8)
                if dtype is None:
                    dtype = bool  # type: ignore
            with device_context:
                array = cp.asarray(array, dtype)
        else:
            with device_context:
                array = cp.asarray(array, dtype)
        return array

    @classmethod
    def to_torch(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> 'torch.Tensor':
        if dtype is not None:
            dtype = cls.convert_dtype_to_torch(dtype)
        if device is not None:
            device = cls.convert_device_to_torch(device)
        if isinstance(array, np.ndarray):
            # skip array of string classes and object, refer to:
            # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
            if re.search(r"[SaUO]", array.dtype.str) is None:
                # numpy array with 0 dims is also sequence iterable,
                # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
                if array.ndim > 0:
                    array = np.ascontiguousarray(array)
        if isinstance(array, torch.Tensor):
            non_blocking = True
            if device is not None and device.type in TORCH_BLOCKING_DEVICES:
                non_blocking = False
            data_out = array.to(
                dtype=dtype, device=device, non_blocking=non_blocking)
            if data_out is not None:
                return data_out
            return array
        return torch.as_tensor(array, dtype=dtype, device=device)

    @classmethod
    def torch_to_np_or_cp(
            cls,
            array: 'torch.Tensor',
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> Union[npt.NDArray[NPGeneric], 'cpt.NDArray[NPGeneric]']:
        if device is None:
            device = array.device
        device_type, device_id, device_string = cls.parse_device(device)
        if device_type == 'cpu':
            return cls.to_numpy(array, dtype=dtype)
        elif device_type == 'cuda':
            return cls.to_cupy(array, dtype=dtype, device=device_string)
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

    @classmethod
    @overload
    def to(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['numpy'] = 'numpy',
    ) -> npt.NDArray[NPGeneric]: ...

    @classmethod
    @overload
    def to(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['cupy'] = 'cupy',
    ) -> 'cpt.NDArray[NPGeneric]': ...

    @classmethod
    @overload
    def to(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['torch'] = 'torch',
    ) -> 'torch.Tensor': ...

    @classmethod
    @overload
    def to(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['list'] = 'list',
    ) -> List: ...

    @classmethod
    def to(
            cls,
            array: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Optional[Union[
                TypeArrayBackend,
                TypeArrayLike[NPGeneric],
            ]] = None,
    ) -> TypeArrayLike[NPGeneric]:
        if backend is None:
            if isinstance(array, list):
                backend = 'list'
            elif cls.is_array(array):
                backend, _ = cls.get_backend_n_device(array)
            else:
                raise TypeError(
                    f"Unsupported array type: {type(array)}. "
                    "Expected one of: np.ndarray, torch.Tensor, cp.ndarray, list, tuple."
                )
        elif isinstance(backend, str):
            if backend not in {'numpy', 'cupy', 'torch', 'list'}:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'numpy', 'cupy', 'torch', 'list'."
                )
        elif isinstance(backend, list):
            backend = 'list'
            if device is not None:
                device_type, device_id, device_string = cls.parse_device(device)
                if device_type != 'cpu' or device_id != -1:
                    raise ValueError(
                        f"only supports device='cpu', got device='{device_string}'."
                    )
            elif dtype is not None:
                raise ValueError(
                    "dtype must be None when backend is a list."
                )
        elif cls.is_array(backend):
            backend, inferred_device = cls.get_backend_n_device(array)
            if device is None:
                device = inferred_device

        if backend == 'numpy':
            to_array_fn = cls.to_numpy
        elif backend == 'cupy':
            to_array_fn = cls.to_cupy
        elif backend == 'torch':
            to_array_fn = cls.to_torch
        elif backend == 'list':
            to_array_fn = cls.to_list
        else:
            raise ValueError(
                f"Unsupported to: {backend}. "
                "Expected one of: 'numpy', 'cupy', 'torch', 'list', 'tuple'."
            )
        return to_array_fn(array=array, dtype=dtype, device=device)

    @classmethod
    def to_dst(
            cls,
            array: TypeArrayLike[NPGeneric],
            dst: TypeArrayLike[NPGeneric],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> TypeArrayLike[NPGeneric]:
        return cls.to(
            array=array, dtype=dtype, device=device, backend=dst)

    @classmethod
    @overload
    def to_cuda(
            cls,
            array: TypeArrayLike[NPGeneric],
            backend: Literal['cupy'] = 'cupy',
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None
    ) -> 'cpt.NDArray[NPGeneric]': ...

    @classmethod
    @overload
    def to_cuda(
            cls,
            array: TypeArrayLike[NPGeneric],
            backend: Literal['torch'] = 'torch',
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None
    ) -> 'torch.Tensor': ...

    @classmethod
    def to_cuda(
            cls,
            array: TypeArrayLike[NPGeneric],
            backend: Optional[TypeArrayBackend] = None,
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> Union['cpt.NDArray[NPGeneric]', 'torch.Tensor']:
        if device is not None:
            device_type, device_id, device_string = cls.parse_device(device)
            if device_type != 'cuda':
                raise ValueError('The device must be a cuda device')
        if HAS_CUPY and isinstance(array, cp.ndarray):
            if backend is None or backend == 'cupy':
                to_array_fn = cls.to_cupy
            elif backend == 'torch':
                to_array_fn = cls.to_torch
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'cupy', 'torch'."
                )
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            if backend is None or backend == 'torch':
                to_array_fn = cls.to_torch
            elif backend == 'cupy':
                to_array_fn = cls.to_cupy
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'cupy', 'torch'."
                )
        else:
            if backend is None:
                # infer backend
                if HAS_CUPY and CUPY_CUDA_AVAILABLE:
                    to_array_fn = cls.to_cupy
                elif HAS_TORCH and TORCH_CUDA_AVAILABLE:
                    to_array_fn = cls.to_torch
                else:
                    raise RuntimeError("Neither CUPY nor PyTorch is available.")
            elif backend == 'cupy':
                to_array_fn = cls.to_cupy
            elif backend == 'torch':
                to_array_fn = cls.to_torch
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'cupy', 'torch'."
                )
        return to_array_fn(array=array, dtype=dtype, device=device)

    @classmethod
    def to_cpu(
            cls,
            array: TypeArrayLike[NPGeneric],
            backend: Optional[TypeArrayBackend] = None,
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> Union['cpt.NDArray[NPGeneric]', 'torch.Tensor']:
        if device is not None:
            device_type, device_id, device_string = cls.parse_device(device)
            if device_type != 'cpu':
                raise ValueError('The device must be a cpu device')

        if isinstance(array, np.ndarray):
            # all ready on gpu, try change to torch
            if backend is None:
                to_array_fn = lambda x: x
            elif backend == 'torch':
                to_array_fn = functools.partial(
                    cls.to_torch, device='cpu')
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'torch'."
                )
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            if backend is None or backend == 'numpy':
                to_array_fn = cls.to_numpy
            elif backend == 'torch':
                to_array_fn = functools.partial(
                    cls.to_torch, device='cpu')
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'numpy', 'torch'."
                )
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            if backend is None or backend == 'torch':
                to_array_fn = functools.partial(
                    cls.to_torch, device='cpu')
            elif backend == 'numpy':
                to_array_fn = cls.to_numpy
            else:
                raise ValueError(
                    f"Unsupported backend: {backend}. "
                    "Expected one of: 'numpy', 'torch'."
                )
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )
        return to_array_fn(array=array, dtype=dtype)

    @classmethod
    @overload
    def _create_array(
            cls,
            target: Union[int, Sequence[int], TypeArrayLike[np.generic]],
            mode: Literal['empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like'],
            backend: Literal['numpy'] = 'numpy',
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> npt.NDArray[NPGeneric]: ...

    @classmethod
    @overload
    def _create_array(
            cls,
            target: Union[int, Sequence[int], TypeArrayLike[np.generic]],
            mode: Literal['empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like'],
            backend: Literal['cupy'] = 'cupy',
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> 'cpt.NDArray[NPGeneric]': ...

    @classmethod
    @overload
    def _create_array(
            cls,
            target: Union[int, Sequence[int], TypeArrayLike[np.generic]],
            mode: Literal['empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like'],
            backend: Literal['torch'] = 'torch',
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> 'torch.Tensor': ...

    @classmethod
    def _create_array(
            cls,
            target: Union[int, Sequence[int], TypeArrayLike[np.generic]],
            mode: Literal['empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like'],
            backend: Optional[Union[
                TypeArrayBackend,
                TypeArrayLike,
            ]] = None,
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> TypeArrayLike[np.generic]:

        # --- Normalize mode
        is_like_mode = mode in {'empty_like', 'zeros_like', 'ones_like'}
        is_shape_mode = mode in {'empty', 'zeros', 'ones'}
        if not (is_like_mode or is_shape_mode):
            raise ValueError(
                "Unknown mode: {mode}. Expected one of: "
                "'empty', 'zeros', 'ones', 'empty_like', 'zeros_like', 'ones_like'."
            )

        # --- Resolve backend & device
        if is_like_mode:
            if backend is not None:
                raise ValueError(
                    f"backend is not allowed for mode='{mode}'.")

            # validate first to avoid helper raising obscure errors
            if not cls.is_array(target):
                raise TypeError(
                    f"Unsupported target type: {type(target)}. "
                    "Expected one of: np.ndarray, torch.Tensor, cp.ndarray"
                )

            creation_backend, inferred_device = cls.get_backend_n_device(target)
            creation_device = device if device is not None else inferred_device
        else:
            # shape-based creation requires a backend
            if backend is None:
                raise ValueError(
                    f"backend is required for mode='{mode}'."
                )
            if isinstance(backend, str):
                if backend not in {'numpy', 'cupy', 'torch'}:
                    raise ValueError(
                        f"Unsupported backend: {backend}. "
                        "Expected one of: 'numpy', 'cupy', 'torch'."
                    )
                creation_backend = backend
                creation_device = device
            elif cls.is_array(backend):
                creation_backend, inferred_device = cls.get_backend_n_device(backend)
                creation_device = device if device is not None else inferred_device
            else:
                raise TypeError(
                    f"Unsupported backend type: {type(backend)}. "
                    "Expected one of: str, np.ndarray, torch.Tensor, cp.ndarray"
                )

        creation_dtype = dtype
        device_type, device_idx, device_string = None, None, None
        if creation_device is not None:
            device_type, device_idx, device_string = cls.parse_device(creation_device)

        # --- Prepare backend-specific kwargs
        if creation_backend == 'numpy':
            # device constraint (only if explicitly provided)
            if device_type is not None and device_type != 'cpu':
                raise ValueError(f"NumPy only supports device='cpu', got '{device_type}'.")
            if creation_dtype is not None:
                creation_dtype = cls.convert_dtype_to_numpy(creation_dtype)
            nplib = np
            creation_kwargs = {'dtype': creation_dtype}
            context = nullcontext()

        elif creation_backend == 'cupy':
            if not HAS_CUPY:
                raise RuntimeError(
                    "cupy is not installed. Please install cupy to use cupy arrays."
                )
            if device_type is not None and device_type != 'cuda':
                raise ValueError(f"CuPy only supports device='cuda', got '{creation_device}'.")
            if creation_dtype is not None:
                creation_dtype = cls.convert_dtype_to_numpy(creation_dtype)
            nplib = cp
            creation_kwargs = {'dtype': creation_dtype}
            if device_idx is None:
                context = cp.cuda.Device(cp.cuda.runtime.getDevice())
            else:
                context = cp.cuda.Device(device_idx)

        elif creation_backend == 'torch':
            if not HAS_TORCH:
                raise RuntimeError(
                    "torch is not installed. Please install torch to use torch arrays."
                )
            if creation_dtype is not None:
                creation_dtype = cls.convert_dtype_to_torch(creation_dtype)
            creation_kwargs = {'dtype': creation_dtype, 'device': device_string}
            nplib = torch
            context = nullcontext()
        else:
            raise ValueError(f"Unsupported backend: {creation_backend}.")
        with context:
            return getattr(nplib, f'{mode}')(target, **creation_kwargs)

    @classmethod
    def empty_like[T: NPGeneric](
            cls,
            array: TypeArrayLike[T],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> TypeArrayLike[T]:
        return cls._create_array(
            target=array, mode='empty_like', dtype=dtype, device=device)

    @classmethod
    def zeros_like[T: NPGeneric](
            cls,
            array: TypeArrayLike[np.generic],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> TypeArrayLike[T]:
        return cls._create_array(
            target=array, mode='zeros_like', dtype=dtype, device=device)

    @classmethod
    def ones_like[T: NPGeneric](
            cls,
            array: TypeArrayLike[T],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
    ) -> TypeArrayLike[T]:
        return cls._create_array(
            target=array, mode='ones_like', dtype=dtype, device=device)

    @classmethod
    @overload
    def empty(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['numpy'] = 'numpy',
    ) -> npt.NDArray[NPGeneric]: ...

    @classmethod
    @overload
    def empty(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['cupy'] = 'cupy',
    ) -> 'cpt.NDArray[NPGeneric]': ...

    @classmethod
    @overload
    def empty(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['torch'] = 'torch',
    ) -> 'torch.Tensor': ...

    @classmethod
    def empty(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Optional[Union[
                TypeArrayBackend,
                TypeArrayLike,
            ]] = None,
    ) -> TypeArrayLike[NPGeneric]:
        return cls._create_array(
            target=shape, mode='empty', dtype=dtype, device=device, backend=backend)

    @classmethod
    @overload
    def zeros(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['numpy'] = 'numpy',
    ) -> npt.NDArray[NPGeneric]: ...

    @classmethod
    @overload
    def zeros(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['cupy'] = 'cupy',
    ) -> 'cpt.NDArray[NPGeneric]': ...

    @classmethod
    @overload
    def zeros(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['torch'] = 'torch',
    ) -> 'torch.Tensor': ...

    @classmethod
    def zeros(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Optional[Union[
                TypeArrayBackend,
                TypeArrayLike,
            ]] = None,
    ) -> TypeArrayLike[NPGeneric]:
        return cls._create_array(
            target=shape, mode='zeros', dtype=dtype, device=device, backend=backend)

    @classmethod
    @overload
    def ones(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['numpy'] = 'numpy',
    ) -> npt.NDArray[NPGeneric]: ...

    @classmethod
    @overload
    def ones(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['cupy'] = 'cupy',
    ) -> 'cpt.NDArray[NPGeneric]': ...

    @classmethod
    @overload
    def ones(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Literal['torch'] = 'torch',
    ) -> 'torch.Tensor': ...

    @classmethod
    def ones(
            cls,
            shape: Union[int, Sequence[int]],
            dtype: Optional[TypeDTypeLike] = None,
            device: Optional[TypeDeviceLike] = None,
            backend: Optional[Union[
                TypeArrayBackend,
                TypeArrayLike,
            ]] = None,
    ) -> TypeArrayLike[NPGeneric]:
        return cls._create_array(
            target=shape, mode='ones', dtype=dtype, device=device, backend=backend)

    @classmethod
    def isin(
            cls,
            array: TypeArrayLike[NPGeneric],
            test_elements: TypeArrayLike[NPGeneric]
    ) -> TypeArrayLike[np.bool_]:
        if isinstance(array, np.ndarray):
            _isin_fn = np.isin
            _to_array_fn = cls.to_numpy
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            _isin_fn = cp.isin
            _to_array_fn = cls.to_cupy
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            _isin_fn = torch.isin
            _to_array_fn = cls.to_torch
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, cp.ndarray, torch.Tensor."
            )
        if cls.is_array(test_elements) or isinstance(test_elements, Sequence):
            test_elements = _to_array_fn(test_elements)
        return _isin_fn(array, test_elements)

    @classmethod
    def where(
            cls,
            condition: TypeArrayLike[NPGeneric],
            x: TypeArrayLike[NPGeneric],
            y: TypeArrayLike[NPGeneric],
    ) -> TypeArrayLike[NPGeneric]:
        if isinstance(condition, np.ndarray):
            _where_fn = np.where
            _to_array_fn = cls.to_numpy
        elif HAS_CUPY and isinstance(condition, cp.ndarray):
            _where_fn = cp.where
            _to_array_fn = cls.to_cupy
        elif HAS_TORCH and isinstance(condition, torch.Tensor):
            _where_fn = torch.where
            _to_array_fn = cls.to_torch
        else:
            raise TypeError(
                f"Unsupported condition type: {type(condition)}. "
                "Expected one of: np.ndarray, cp.ndarray, torch.Tensor."
            )
        if cls.is_array(x):
            x = _to_array_fn(x)
        if cls.is_array(y):
            y = _to_array_fn(y)
        return _where_fn(condition, x, y)

    @classmethod
    def unique(
            cls,
            array: TypeArrayLike[NPGeneric],
            sorted=True,
            return_inverse=False,
            return_counts=False,
    ):
        kwargs = {
            'return_inverse': return_inverse,
            'return_counts': return_counts,
            'sorted': sorted,
        }
        if isinstance(array, np.ndarray):
            return np.unique(array, **kwargs)
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.unique(array, **kwargs)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            return torch.unique(array, **kwargs)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, cp.ndarray, torch.Tensor."
            )

    @staticmethod
    def clip[T: NPGeneric](
            x: TypeArrayLike[T], min=None, max=None, out=None,
    ) -> TypeArrayLike[T]:
        if isinstance(x, np.ndarray):
            x_out = np.clip(x, a_min=min, a_max=max, out=out)
        elif HAS_CUPY and isinstance(x, cp.ndarray):
            x_out = cp.clip(x, a_min=min, a_max=max, out=out)
        elif HAS_TORCH and isinstance(x, torch.Tensor):
            x_out = torch.clamp(x, min=min, max=max, out=out)
        else:
            raise TypeError(
                f"Unsupported array type: {type(x)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )
        return x_out if x_out is not None else x

    @staticmethod
    def copy[T](array: TypeArrayLike[T]) -> TypeArrayLike[T]:
        if isinstance(array, np.ndarray):
            return array.copy()
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.copy(array)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            out_array = array.clone()
            out_array.detach_()
            return out_array
        elif isinstance(array, Sequence):
            return pycopy.deepcopy(array)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray, list, tuple."
            )

    @staticmethod
    def _var_or_std[T: NPGeneric](
            x: TypeArrayLike[T],
            op: Literal['var', 'std'],
            axis: Union[int, Sequence[int]] = None,
            keepdims: bool = False,
            correction: Union[int, float] = 1,
            out: Optional[TypeArrayLike[T]] = None,
    ) -> TypeArrayLike[T]:
        if op not in {'var', 'std'}:
            raise ValueError(
                f"Unknown op: {op}. Expected 'var' or 'std'.")
        if isinstance(x, np.ndarray):
            _nplib = np
            _fn_kwargs = dict(
                axis=axis, keepdims=keepdims, correction=correction, out=out)
        elif HAS_CUPY and isinstance(x, cp.ndarray):
            _nplib = cp
            _fn_kwargs = dict(
                axis=axis, keepdims=keepdims, correction=correction, out=out)
        elif HAS_TORCH and isinstance(x, torch.Tensor):
            _nplib = torch
            _fn_kwargs = dict(
                dim=axis, keepdim=keepdims, correction=correction, out=out)
        else:
            raise TypeError(
                f"Unsupported array type: {type(x)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )

        _op_fn = getattr(_nplib, op)
        out_x = _op_fn(x, **_fn_kwargs)
        return out_x if out_x is not None else out_x

    @classmethod
    def std[T: NPGeneric](
            cls,
            x: TypeArrayLike[T],
            axis: Union[int, Sequence[int]] = None,
            keepdims: bool = False,
            correction: Union[int, float] = 1,
            out: Optional[TypeArrayLike[T]] = None,
    ) -> TypeArrayLike[T]:
        return cls._var_or_std(
            x, op='std', axis=axis, keepdims=keepdims, correction=correction, out=out)

    @classmethod
    def var[T: NPGeneric](
            cls,
            x: TypeArrayLike[T],
            axis: Union[int, Sequence[int]] = None,
            keepdims: bool = False,
            correction: Union[int, float] = 1,
            out: Optional[TypeArrayLike[T]] = None,
    ) -> TypeArrayLike[T]:
        return cls._var_or_std(
            x, op='var', axis=axis, keepdims=keepdims, correction=correction, out=out)

    @staticmethod
    def round[T: NPGeneric](
            x: TypeArrayLike[T],
            decimals: int = 0,
            out: Optional[TypeArrayLike[T]] = None,
    ) -> TypeArrayLike[T]:
        if isinstance(x, np.ndarray):
            _nplib = np
        elif HAS_CUPY and isinstance(x, cp.ndarray):
            _nplib = cp
        elif HAS_TORCH and isinstance(x, torch.Tensor):
            _nplib = torch
        else:
            raise TypeError(
                f"Unsupported array type: {type(x)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )
        out_x = _nplib.round(x, decimals=decimals, out=out)
        return out_x if out_x is not None else out_x

    @staticmethod
    def argmax(
            array: TypeArrayLike[NPGeneric], axis: Optional[int] = None
    ) -> TypeArrayLike[np.int64]:
        if isinstance(array, np.ndarray):
            return np.argmax(array, axis=axis)
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.argmax(array, axis=axis)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            return torch.argmax(array, dim=axis)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
            )

    @classmethod
    def one_hot(cls, array: TypeArrayLike[NPInteger], num_classes: int) -> TypeArrayLike[NPInteger]:
        if (array < 0).any() or (array >= num_classes).any():
            raise ValueError(f"Invalid class label: {array}.")
        dtype = cls.convert_dtype_to_string(array.dtype)
        if dtype not in {'int8', 'int16', 'int32', 'int64'}:
            raise ValueError(f"Unsupported dtype: {dtype}.")
        if dtype != 'int64':
            array_ = cls.to(array, 'int64')
        else:
            array_ = array

        if isinstance(array_, np.ndarray):
            return np.eye(num_classes)[array_]
        elif HAS_CUPY and isinstance(array_, cp.ndarray):
            return cp.eye(num_classes)[array_]
        elif HAS_TORCH and isinstance(array_, torch.Tensor):
            return F.one_hot(array_, num_classes=num_classes)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array_)}. "
            )

    @classmethod
    def _norm_pad_width(
            cls,
            pad_width: TypePadWidth,
            n_dims,
    ) -> Sequence[Sequence[int]]:
        if cls.is_array(pad_width):
            pad_width = cls.to_list(pad_width)

        # ---- normalize pad_width -> tuple[tuple[int, int], ...] length n_dims
        # int -> same (k,k) for all dims
        if isinstance(pad_width, int):
            return tuple((pad_width, pad_width) for _ in range(n_dims))

        # nested sequence: ((b0,a0), (b1,a1), ...)
        if isinstance(pad_width, Sequence) and pad_width and isinstance(pad_width[0], Sequence):
            if len(pad_width) != n_dims:
                raise ValueError(f"pad_width has {len(pad_width)} dims; expected {n_dims}.")
            out = []
            for i, p in enumerate(pad_width):
                if len(p) != 2:
                    raise ValueError(f"pad_width[{i}] must be (before, after); got {p}.")
                b, a = int(p[0]), int(p[1])
                if b < 0 or a < 0:
                    raise ValueError("Negative padding is not supported.")
                out.append((b, a))
            return tuple(out)

        # flat sequence: length n  -> symmetric per-dim
        #               length 2n -> explicit (before0, after0, before1, after1, ...)
        if isinstance(pad_width, Sequence):
            flat = [int(v) for v in pad_width]
            if any(v < 0 for v in flat):
                raise ValueError("Negative padding is not supported.")
            if len(flat) == n_dims:
                return tuple((k, k) for k in flat)
            if len(flat) == 2 * n_dims:
                return tuple((flat[2 * i], flat[2 * i + 1]) for i in range(n_dims))

        raise ValueError(
            "pad_width must be int, sequence of length N, "
            "sequence of length 2N, or sequence of N pairs."
        )

    @classmethod
    def pad[T: TypeArrayLike[NPGeneric]](
            cls,
            array: T,
            pad_width: TypePadWidth,
            mode: str,
            constant_values=0,
    ) -> T:
        n_dims = array.ndim

        pad_pairs = cls._norm_pad_width(pad_width, n_dims)  # ((b0,a0), (b1,a1), ...)

        if isinstance(array, np.ndarray):
            _nplib = np
            _pad_fn = functools.partial(
                _nplib.pad,
                pad_width=pad_pairs, mode=mode, constant_values=constant_values)
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            _nplib = cp
            _pad_fn = functools.partial(
                _nplib.pad,
                pad_width=pad_pairs, mode=mode, constant_values=constant_values)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            _nplib = torch

            # map numpy-like names to torch
            if mode == "edge":
                torch_mode = "replicate"
            elif mode == "wrap":
                torch_mode = "circular"
            elif mode in ("constant", "reflect", "replicate", "circular"):
                torch_mode = mode
            elif mode == "symmetric":
                # PyTorch does not support 'symmetric' (NumPy's includes edge pixel)
                raise ValueError("PyTorch does not support mode='symmetric'. Use 'reflect' or 'replicate'.")
            else:
                raise ValueError(f"Unsupported pad mode for torch: {mode!r}")

            # torch expects (pad_last_before, pad_last_after, pad_second_last_before, pad_second_last_after, ...)
            torch_pad = []
            for (b, a) in reversed(pad_pairs):
                torch_pad.extend([b, a])
            torch_pad = tuple(torch_pad)

            _pad_fn = functools.partial(
                F.pad, pad=torch_pad, mode=torch_mode, value=constant_values)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )
        out_array = _pad_fn(array)
        return out_array

    @classmethod
    def unpad[T: TypeArrayLike[NPGeneric]](
            cls,
            array: T,
            pad_width: TypePadWidth,
    ) -> T:
        norm_pad_width = cls._norm_pad_width(pad_width, array.ndim)
        slices: List[slice] = []
        for c in cls.to_list(norm_pad_width):
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return array[tuple(slices)]

    @staticmethod
    def expand_dims[T: TypeArrayLike[NPGeneric]](array: T, axis: int) -> T:
        if isinstance(array, np.ndarray):
            return np.expand_dims(array, axis=axis)
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.expand_dims(array, axis=axis)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            return torch.unsqueeze(array, dim=axis)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray.")

    @staticmethod
    def squeeze[T: TypeArrayLike[NPGeneric]](array: T, axis: int) -> T:
        if isinstance(array, np.ndarray):
            return np.squeeze(array, axis=axis)
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.squeeze(array, axis=axis)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            return torch.squeeze(array, dim=axis)
        else:
            return array.squeeze(axis=axis)

    @classmethod
    def einsum[T: TypeArrayLike[NPGeneric]](
            cls, subscripts: str, *operands: T
    ) -> T:
        backend, device = cls.get_backend_n_device(operands[0])
        if backend == 'numpy':
            to_array = cls.to_numpy
            nplib = np
        elif backend == 'cupy':
            to_array = functools.partial(cls.to_cupy, device=device)
            nplib = cp
        elif backend == 'torch':
            to_array = functools.partial(cls.to_torch, device=device)
            nplib = torch
        else:
            raise TypeError(
                f"Unsupported backend: {backend}. "
                "Expected one of: numpy, cupy, torch."
            )
        operands = list(map(to_array, operands))
        return nplib.einsum(subscripts, *operands)

    @staticmethod
    def flip[T: TypeArrayLike[NPGeneric]](
            array: T, axis: Union[int, Sequence[int]]
    ) -> T:
        if isinstance(array, np.ndarray):
            return np.flip(array, axis=axis)
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.flip(array, axis=axis)
        elif HAS_TORCH and isinstance(array, torch.Tensor):
            if isinstance(axis, int):
                axis = [axis]
            return torch.flip(array, dims=axis)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array)}. "
            )

    @classmethod
    def concatenate[T: TypeArrayLike[NPGeneric]](
            cls,
            arrays: Union[List[T], Tuple[T, ...]], axis: int = 0
    ) -> T:
        if not isinstance(arrays, (list, tuple)):
            raise TypeError(f"Expected list or tuple, got {type(arrays)}.")
        if len(arrays) == 0:
            raise ValueError("Cannot concatenate empty list of arrays.")
        if len(arrays) == 1:
            return arrays[0]
        array0 = arrays[0]
        if isinstance(array0, np.ndarray):
            to_array_fn = cls.to_numpy
            cat_fn = functools.partial(np.concatenate, axis=axis)
        elif HAS_CUPY and isinstance(array0, cp.ndarray):
            to_array_fn = functools.partial(cls.to_cupy, device=array0.device)
            cat_fn = functools.partial(cp.concatenate, axis=axis)
        elif HAS_TORCH and isinstance(array0, torch.Tensor):
            to_array_fn = functools.partial(cls.to_torch, device=array0.device)
            cat_fn = functools.partial(torch.cat, dim=axis)
        else:
            raise TypeError(
                f"Unsupported array type: {type(array0)}. "
                "Expected one of: np.ndarray, torch.Tensor, cp.ndarray."
            )
        arrays = map(to_array_fn, arrays)
        return cat_fn(arrays)



if not HAS_CUPY:
    def cupy_not_available(*args, **kwargs):
        raise RuntimeError("cupy is not available.")
    ArrayUtils.to_cupy = cupy_not_available
    del cupy_not_available

if not HAS_TORCH:
    def torch_not_available(*args, **kwargs):
        raise RuntimeError("torch is not available.")
    ArrayUtils.to_torch = torch_not_available
    ArrayUtils.torch_to_np_or_cp = torch_not_available
    del torch_not_available

is_array = ArrayUtils.is_array
convert_dtype_to_string = ArrayUtils.convert_dtype_to_string
convert_dtype_to_numpy = ArrayUtils.convert_dtype_to_numpy
convert_dtype_to_torch = ArrayUtils.convert_dtype_to_torch
is_cuda_available = ArrayUtils.is_cuda_available
parse_device = ArrayUtils.parse_device
get_backend_n_device = ArrayUtils.get_backend_n_device
convert_device_to_type = ArrayUtils.convert_device_to_type
convert_device_to_string = ArrayUtils.convert_device_to_string
convert_device_to_index = ArrayUtils.convert_device_to_index
convert_device_to_cupy = ArrayUtils.convert_device_to_cupy
convert_device_to_torch = ArrayUtils.convert_device_to_torch
to_numpy = ArrayUtils.to_numpy
to_list = ArrayUtils.to_list
to_cupy = ArrayUtils.to_cupy
to_torch = ArrayUtils.to_torch
torch_to_np_or_cp = ArrayUtils.torch_to_np_or_cp
to = ArrayUtils.to
to_dst = ArrayUtils.to_dst
to_cuda = ArrayUtils.to_cuda
to_cpu = ArrayUtils.to_cpu
empty_like = ArrayUtils.empty_like
zeros_like = ArrayUtils.zeros_like
ones_like = ArrayUtils.ones_like
empty = ArrayUtils.empty
zeros = ArrayUtils.zeros
ones = ArrayUtils.ones
isin = ArrayUtils.isin
where = ArrayUtils.where
unique = ArrayUtils.unique
clip = ArrayUtils.clip
copy = ArrayUtils.copy
std = ArrayUtils.std
var = ArrayUtils.var
round = ArrayUtils.round
argmax = ArrayUtils.argmax
one_hot = ArrayUtils.one_hot
pad = ArrayUtils.pad
unpad = ArrayUtils.unpad
expand_dims = ArrayUtils.expand_dims
squeeze = ArrayUtils.squeeze
einsum = ArrayUtils.einsum
flip = ArrayUtils.flip
concatenate = ArrayUtils.concatenate