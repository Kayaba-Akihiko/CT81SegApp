#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Union, Any, Optional, Dict, Tuple, Callable, ContextManager, List, override, Literal, Protocol, runtime_checkable, TypeVar
from contextlib import nullcontext
from pathlib import Path
import logging

import torch
from torch.optim import Optimizer
from torch._dynamo import OptimizedModule

from ..xutils import array_utils as xp
from ..protocol import Stateful

from .base_distributor import BaseDistributor, TypeFloatingMatmulPrecision
from .protocol import TypeModule, TypeCkptState, TypeStrategy, TypePrecision

_logger = logging.getLogger(__name__)


class DummyDistributor(BaseDistributor):

    @override
    def __init__(
            self,
            seed=831,
            tracker='tb',
            accelerator='auto',
            devices: Union[str, int] = 'auto',
            float32_matmul_precision: TypeFloatingMatmulPrecision = 'highest',
            precision: TypePrecision = '32-true',
            strategy: TypeStrategy = 'auto',
    ):
        super().__init__(
            backend='none',
            seed=seed,
            tracker=tracker,
            float32_matmul_precision=float32_matmul_precision,
        )
        if accelerator not in ['auto', 'cuda', 'cpu']:
            raise ValueError(f'accelerator={accelerator} is not supported.')

        if devices == 'auto':
            devices = 1
        elif devices == 1:
            pass
        else:
            raise ValueError(f'devices={devices} is not supported.')

        if precision not in ['32-true', '32', 32]:
            raise ValueError(f'precision={precision} is not supported.')

        if strategy != 'auto':
            raise ValueError(f'strategy={strategy} is not supported.')

        if accelerator == 'auto':
            self._device = (
                torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu')
            )
        else:
            self._device = torch.device(accelerator)


    def _backend_launch(self) -> None:
        return

    def to_device(
            self,
            *x: Union[
                torch.Tensor,
                Union[TypeModule],
            ]
    ) -> Optional[Union[
        Union[torch.Tensor, TypeModule],
        Tuple[Union[torch.Tensor, TypeModule]], ...]
    ]:
        if len(x) == 0:
            return None
        if len(x) == 1:
            return xp.to(x[0], device=self._device)
        return tuple(xp.to(x_, device=self._device) for x_ in x)

    @property
    def device(self) -> torch.device:
        return self._device

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        if 'model' in kwargs:
            kwargs.pop('model')  # for Fabric
        loss.backward(**kwargs)

    def no_sync(
            self,
            *model: TypeModule,
            enabled=True
    ) -> ContextManager:
        return nullcontext()

    def barrier(self) -> None:
        return

    def all_reduce(
            self, x: torch.Tensor, reduce_op='sum') -> torch.Tensor:
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)

    def _backend_unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return self._unwrap_compiled(model)[0]

    @staticmethod
    def _unwrap_compiled(
            obj: Union[Any, OptimizedModule]
    ) -> Tuple[Union[Any, torch.nn.Module], Optional[Dict[str, Any]]]:
        """Removes the :class:`torch._dynamo.OptimizedModule` around the object if it is wrapped.

        Use this function before instance checks against e.g. :class:`_FabricModule`.

        """
        if isinstance(obj, OptimizedModule):
            if (compile_kwargs := getattr(obj, "_compile_kwargs", None)) is None:
                raise RuntimeError(
                    "Failed to determine the arguments that were used to compile the module. Make sure to import"
                    " lightning before `torch.compile` is used."
                )
            return obj._orig_mod, compile_kwargs
        return obj, None

    def _save_state(
            self,
            save_path: Path,
            state: TypeCkptState,
    ) -> None:
        converted_state = {}
        for k, v in state.items():
            if isinstance(v, Stateful):
                converted_state[k] = v.state_dict()
            else:
                converted_state[k] = v
        torch.save(converted_state, save_path)

    def _load_state(
            self,
            load_path: Path,
            state: TypeCkptState,
    ):
        state_dict = torch.load(
            load_path, map_location='cpu', weights_only=True)

        required_keys = set(state.keys())
        loaded_keys = set(state_dict.keys())
        if required_keys != loaded_keys:
            raise RuntimeError(
                f'Keys in the checkpoint do not match the keys in the model: '
                f'checkpoint keys: {sorted(loaded_keys)}, '
                f'model keys: {sorted(required_keys)}'
            )

        for k, v in state.items():
            if isinstance(v, Stateful):
                v.load_state_dict(state_dict[k])
            else:
                state[k] = state_dict[k]

    def save_module(
            self, module: TypeModule, save_path: Path) -> None:
        module = self.unwrap_model(module)
        torch.save({'module': module.state_dict()}, save_path)

    def load_module(
            self, module: TypeModule, load_path: Path) -> None:
        module = self.unwrap_model(module)
        state_dict = torch.load(load_path, map_location='cpu')
        module.load_state_dict(state_dict['module'])

    def is_main_process(self) -> bool:
        return True

    def setup_model(
            self,
            model: TypeModule,
            *optimizers: Optimizer,
    ) -> Union[TypeModule, Tuple[TypeModule, Optimizer]]:
        if optimizers:
            return model, *optimizers
        return model

    def setup_optimizer(
            self, *optimizers: Optimizer
    ) -> Union[None, Optimizer, Tuple[Optimizer, ...]]:
        if not optimizers:
            return None
        if len(optimizers) == 1:
            return optimizers[0]
        return tuple(*optimizers)


    def init_model(self, *args, **kwargs) -> ContextManager:
        return nullcontext()

