#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from abc import ABC, abstractmethod
from typing import (
    Literal, Any, Dict, Optional, Union, Tuple, List, Sequence, TypeAlias, Callable)
from pathlib import Path
import logging
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
import functools
import math

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP

from .protocol import DistributorProtocol, TypeModule, TypeCkptState
from .utils import pl_worker_init_function

TypeBackend: TypeAlias = Literal['none', 'fabric']
TypeFloatingMatmulPrecision: TypeAlias = Literal['highest', 'high', 'medium']

_logger = logging.getLogger(__name__)



class BaseDistributor(ABC, DistributorProtocol):

    def __init__(
            self,
            backend: TypeBackend,
            seed=831,
            tracker='tb',
            float32_matmul_precision: TypeFloatingMatmulPrecision = 'highest',
    ):
        super().__init__()
        self.backend = backend
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.seed = seed

        if torch.cuda.is_available():
            _logger.info(
                f'Using '
                f'float32_matmul_precision={float32_matmul_precision}.'
            )
            torch.set_float32_matmul_precision(float32_matmul_precision)
            # torch.backends.cuda.matmul.allow_tf32 = False
            previous_cdnn_allow_tf = torch.backends.cudnn.allow_tf32
            cdnn_allow_tf = previous_cdnn_allow_tf
            if float32_matmul_precision == 'highest':
                cdnn_allow_tf = False
            else:
                assert float32_matmul_precision in ['high', 'medium']

            if previous_cdnn_allow_tf != cdnn_allow_tf:
                _logger.info(
                    f'CuDNN allow_tf32 changed '
                    f'from {previous_cdnn_allow_tf} to {cdnn_allow_tf} '
                    f'By the policy float32_matmul_precision='
                    f'{float32_matmul_precision}.')
            torch.backends.cudnn.allow_tf32 = cdnn_allow_tf

        if tracker != 'tb':
            raise NotImplementedError(tracker)
        self.tracker_name = tracker
        self.tracker = None

        torch.serialization.add_safe_globals([
            np.ndarray,
            np._core.multiarray._reconstruct,
            np.dtype,
            np.dtypes.UInt32DType,
        ])

        self._launched = False

    def launch(self) -> None:
        if self._launched:
            raise RuntimeError('Already launched.')
        self._backend_launch()
        self._launched = True

    @abstractmethod
    def _backend_launch(self) -> None:
        raise NotImplementedError()

    def save_checkpoint(
            self,
            save_dir: Path,
            state_groups: Dict[str, TypeCkptState],
            prefix='',
    ):
        if self.tracker_name == 'tb':
            from tensorboardX import SummaryWriter
            if self.is_main_process():
                # reopen tracker
                logdir = self.tracker.logdir
                self.tracker.flush()
                self.tracker.close()
                self.tracker = SummaryWriter(logdir=logdir)
        else:
            raise NotImplementedError()

        save_dir.mkdir(parents=True, exist_ok=True)
        for group_name, state in state_groups.items():
            group_save_path = save_dir / f'{prefix}{group_name}.ckpt'
            self._save_state(save_path=group_save_path, state=state)

    @abstractmethod
    def _save_state(
            self,
            save_path: Path,
            state: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def load_checkpoint(
            self,
            load_dir: Path,
            state_groups: Dict[str, TypeCkptState],
            prefix='',
    ) -> None:
        for group_name, state in state_groups.items():
            group_load_path = load_dir / f'{prefix}{group_name}.ckpt'
            self._load_state(load_path=group_load_path, state=state)

    @abstractmethod
    def _load_state(
            self,
            load_path: Path,
            state: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def compile(
            self,
            *module: TypeModule,
            enabled=True,
            mode='default'
    ) -> Optional[Union[TypeModule, Tuple[TypeModule, ...]]]:
        if len(module) == 0:
            return None
        compile_fn = functools.partial(
            torch.compile, disable=not enabled, mode=mode)
        if len(module) == 1:
            return compile_fn(module[0])
        return tuple(compile_fn(m) for m in module)

    def all_gather_object[T](self, x: T) -> List[T]:
        if x is None:
            raise ValueError('No object to gather.')
        if not self.is_distributed():
            return [x]
        out_list = [None] * self.world_size
        torch.distributed.all_gather_object(out_list, x)
        return out_list

    def broadcast_object(self, *x: Any, src=0) -> Any:
        if len(x) == 0:
            raise ValueError('No object to broadcast.')
        if not self.is_distributed():
            if len(x) == 1:
                return x[0]
            return x
        objects = [*x]
        torch.distributed.broadcast_object_list(
            objects, src=src)
        return objects[0] if len(objects) == 1 else objects

    @staticmethod
    def collect_rng_states() -> Dict[str, Any]:
        r"""Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
        states = {
            "torch": torch.get_rng_state(),
            "python": python_get_rng_state(),
            "numpy": np.random.get_state(),
            "torch.cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        return states

    @staticmethod
    def set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
        r"""Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
        process."""
        torch.set_rng_state(rng_state_dict["torch"])
        # torch.cuda rng_state is only included since v1.8.
        if "torch.cuda" in rng_state_dict:
            torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])

        np.random.set_state(rng_state_dict["numpy"])
        version, state, gauss = rng_state_dict["python"]
        python_set_rng_state((version, tuple(state), gauss))

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda_deterministic = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
            else:  # faster, less reproducible
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                torch.use_deterministic_algorithms(False)

    def get_dataloader_worker_init_fn(self) -> Callable[[int, ], None]:
        init_fn = functools.partial(
            pl_worker_init_function, rank=self.global_rank)
        return init_fn

    def initialize_tracker(
            self, log_dir: Optional[Path] = None, **kwargs) -> None:
        if not self.is_main_process():
            return
        if self.tracker is not None:
            raise RuntimeError('Already initialized.')
        if self.tracker_name == 'tb':
            if log_dir is None:
                raise RuntimeError('No log_dir specified.')
            from tensorboardX import SummaryWriter
            _logger.info(f'Using, Tracker log dir {log_dir}.')
            log_dir.mkdir(exist_ok=True, parents=True)
            self.tracker = SummaryWriter(str(log_dir))
        else:
            raise NotImplementedError(self.tracker_name)

    def tracker_log(self, tag: str, value: float, step: int) -> None:
        if not self.is_main_process():
            return
        if self.tracker is None:
            raise RuntimeError('tracker is not initialized.')
        if self.tracker_name == 'tb':
            self.tracker.add_scalar(tag, value, step)
        else:
            raise NotImplementedError(self.tracker_name)

    def unwrap_model(
            self, model: Union[TypeModule, torch.optim.Optimizer]
    ) -> Union[torch.nn.Module, torch.optim.Optimizer]:
        unwrapped = self._backend_unwrap_model(model)
        if isinstance(unwrapped, (DP, DDP)):
            unwrapped = unwrapped.module
        else:
            unwrapped = unwrapped
        return unwrapped

    @abstractmethod
    def _backend_unwrap_model(
            self,
            model: Union[TypeModule, torch.optim.Optimizer]
    ) -> Union[torch.nn.Module, torch.optim.Optimizer]:
        raise NotImplementedError

    def shard_data_pool[T](
            self, data_pool: Sequence[T]
    ) -> Tuple[int, Union[Sequence[T], List[T]]]:
        if self.world_size <= 1:
            return len(data_pool), data_pool

        n_samples_per_rank = math.ceil(
            (sample_pool_size := len(data_pool)) / self.world_size)

        sharded_pool = []
        for i in range(0, sample_pool_size, self.world_size):
            i_take = i + self.global_rank
            if i_take < sample_pool_size:
                sharded_pool.append(data_pool[i_take])
        return n_samples_per_rank, sharded_pool

    def is_distributed(self) -> bool:
        return self.world_size > 1