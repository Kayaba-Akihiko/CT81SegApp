#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import (
    ContextManager, Union, TypeAlias, override, Literal, Optional,
    Tuple, Callable, List, Dict, Any
)
import os
import logging
import shutil
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import importlib.util

from fsspec import register_implementation
from fsspec.implementations.local import LocalFileOpener, LocalFileSystem
import torch
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from lightning import Fabric
from lightning.fabric.fabric import _FabricModule, _FabricOptimizer
from lightning.fabric.wrappers import _unwrap_objects, _unwrap_compiled
from lightning.fabric.strategies import SingleDeviceStrategy, DDPStrategy, FSDPStrategy
from lightning.fabric.strategies.fsdp import _is_sharded_checkpoint, _get_full_state_dict_context

from ..xutils import dist_utils

from .base_distributor import BaseDistributor, TypeFloatingMatmulPrecision
from .protocol import TypeModule as _TypeModule, TypeCkptState, TypeStrategy, TypePrecision

TypeModule: TypeAlias = Union[_FabricModule, _TypeModule]
TypeOptimizer: TypeAlias = Union[_FabricOptimizer, Optimizer]

_logger = logging.getLogger(__name__)


class FabricDistributor(BaseDistributor):
    @override
    def __init__(
            self,
            seed=831,
            tracker='tb',
            accelerator='auto',
            devices: Union[str, int] = 'auto',
            float32_matmul_precision: TypeFloatingMatmulPrecision = 'highest',
            precision: TypePrecision= '32-true',
            strategy: TypeStrategy = 'auto',
    ):
        super().__init__(
            backend='fabric',
            seed=seed,
            tracker=tracker,
            float32_matmul_precision=float32_matmul_precision,
        )
        assert strategy in ['auto', 'ddp', 'ddp_spawn', 'fsdp'], strategy

        # Hack to replace file system that use copy2 with ours that use copyfile
        register_implementation(
            'file', XLocalFileSystem, clobber=True)
        register_implementation(
            'local', XLocalFileSystem, clobber=True)

        n_nodes = 1
        _logger.info(
            f'CUDA available is {torch.cuda.is_available()}.')
        # Check if in slurm
        if dist_utils.is_running_on_slurm():
            # in slurm check if multi devices
            n_nodes = int(os.environ['SLURM_NNODES'])
            n_devices_tpn = int(os.environ.get('SLURM_NTASKS_PER_NODE', 0))
            n_devices_gpn = int(os.environ.get('SLURM_GPUS_PER_NODE', 0))
            if devices == 'auto':
                if n_devices_tpn > 0:
                    devices = n_devices_tpn
                if n_devices_gpn > 0:
                    devices = n_devices_gpn
                if n_devices_tpn > 0 and n_devices_gpn > 0:
                    assert n_devices_tpn == n_devices_gpn
                    devices = n_devices_tpn
                _logger.info(
                    f'Fabric num_nodes={n_nodes}, num_devices={devices}.')

            if devices == 0:
                # Running CPU only
                devices = 'auto'

        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            num_nodes=n_nodes,
            devices=devices,
            precision=precision,
        )
        supported_strategies = (SingleDeviceStrategy, DDPStrategy, FSDPStrategy)
        strategy = self.fabric.strategy
        if not isinstance(strategy, supported_strategies):
            raise NotImplementedError(strategy)
        _logger.info(f'Using strategy {strategy}.')

    def _backend_launch(self) -> None:
        self.fabric.launch()
        _logger.info('Running fabric.')
        self.global_rank = self.fabric.global_rank
        self.local_rank = self.fabric.local_rank
        self.world_size = self.fabric.world_size
        _logger.info(
            f'World:\n'
            f'global_rank {self.global_rank}\n'
            f'local_rank {self.local_rank}\n'
            f'world_size {self.world_size}'
        )

        if self.is_distributed():
            try:
                import cupy as cp
                try:
                    cp.cuda.Device(self.fabric.local_rank).use()
                    _logger.info(f'Set Cupy device {self.fabric.local_rank}.')
                except cp.cuda.runtime.CUDARuntimeError:
                    pass
            except ImportError:
                cp = None

            if importlib.util.find_spec('vtk') is not None:
                os.environ['VTK_DEFAULT_EGL_DEVICE_INDEX'] = str(self.fabric.local_rank)
                _logger.info(f'Set VTK EGL device to {self.fabric.local_rank}.')

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
            return self.fabric.to_device(x[0])
        return self.fabric.to_device(x)

    @property
    def device(self) -> torch.device:
        return self.fabric.device

    def backward(
            self,
            loss: torch.Tensor,
            model: Optional[torch.nn.Module] = None,
            **kwargs) -> None:
        self.fabric.backward(loss, model=model, **kwargs)

    def no_sync(
            self,
            *model: TypeModule,
            enabled=True
    ) -> ContextManager:
        if not enabled:
            return nullcontext()
        if len(model) == 0:
            return nullcontext()
        if len(model) == 1:
            return self.fabric.no_backward_sync(
                model[0], enabled=enabled)
        contexts = (
            self.fabric.no_backward_sync(m, enabled=enabled)
            for m in model
        )
        return _multi_context(*contexts)

    def barrier(self) -> None:
        self.fabric.barrier()

    def all_reduce(
            self, x: torch.Tensor, reduce_op='sum') -> torch.Tensor:
        return self.fabric.all_reduce(x, reduce_op=reduce_op)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed():
            return x.unsqueeze(0)  # (1,)
        return self.fabric.all_gather(x)  # (ws, ...) or (...)

    @override
    def seed_everything(self, seed: int) -> None:
        super().seed_everything(seed)
        self.fabric.seed_everything(seed)

    def _backend_unwrap_model(
            self, model: Union[torch.nn.Module, torch.optim.Optimizer]
    ) -> Union[torch.nn.Module, torch.optim.Optimizer]:
        return _fabric_unwrap_objects(model)

    def _save_state(
            self,
            save_path: Path,
            state: TypeCkptState,
    ) -> None:
        self.fabric.save(path=save_path, state=state)

    def _load_state(
            self,
            load_path: Path,
            state: TypeCkptState,
    ) -> None:
        self.fabric.load(path=load_path, state=state)

    def save_module(
            self, module: TypeModule, save_path: Path) -> None:
        module = self.unwrap_model(module)
        if isinstance(module, FullyShardedDataParallel):
            if _is_sharded_checkpoint(save_path):
                shutil.rmtree(save_path)
            state_dict_ctx = _get_full_state_dict_context(
                module, world_size=self.world_size)
            with state_dict_ctx:
                state_dict = module.state_dict()
        else:
            state_dict = module.state_dict()
        if self.global_rank == 0:
            torch.save(state_dict, save_path)

    def load_module(
            self, module: TypeModule, load_path: Path) -> None:
        module = self.unwrap_model(module)
        self.fabric.load_raw(path=load_path, obj=module)

    def is_main_process(self) -> bool:
        return self.fabric.is_global_zero

    def init_model(self, *args, **kwargs) -> ContextManager:
        return self.fabric.init_module(*args, **kwargs)

    def setup_model(
            self,
            model: TypeModule,
            *optimizers: Optimizer,
    ) -> Union[TypeModule, Tuple[TypeModule, TypeOptimizer, ...]]:
        return self.fabric.setup(model, *optimizers)

    def setup_optimizer(
            self, *optimizers: Optimizer
    ) -> Union[TypeOptimizer, Tuple[TypeOptimizer, ...]]:
        return self.fabric.setup_optimizers(*optimizers)

class XLocalFileOpener(LocalFileOpener):

    def commit(self):
        # Replace copy2 with copy to work on our system
        if self.autocommit:
            raise RuntimeError("Can only commit if not already set to autocommit")
        shutil.move(self.temp, self.path, copy_function=shutil.copyfile)


class XLocalFileSystem(LocalFileSystem):

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        path = self._strip_protocol(path)
        if self.auto_mkdir and "w" in mode:
            self.makedirs(self._parent(path), exist_ok=True)
        return XLocalFileOpener(path, mode, fs=self, **kwargs)


@contextmanager
def _multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


def _fabric_unwrap_objects(model: TypeModule) -> torch.nn.Module:
    unwrapped = _unwrap_objects(model)
    if isinstance(unwrapped, OptimizedModule):
        unwrapped = _unwrap_compiled(unwrapped)[0]
    return unwrapped