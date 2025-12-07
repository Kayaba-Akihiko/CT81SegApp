#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from abc import abstractmethod
from typing import (
    Literal, Dict, Any, Protocol, ContextManager, Optional, Callable,
    List, Tuple ,Union, Sequence, TypeAlias, TypeVar, runtime_checkable
)
from pathlib import Path

import torch
from torch._dynamo import OptimizedModule
from torch.optim import Optimizer

from ..protocol import Stateful

TypeModule: TypeAlias = Union[
    Optimizer,
    torch.nn.Module,
    OptimizedModule,
]

TypeCkptState: TypeAlias = Dict[str, Union[str, float, Stateful, Any]]
TypeStrategy: TypeAlias = Literal['auto', 'ddp', 'ddp_spawn', 'fsdp']
TypePrecision: TypeAlias = Literal[
    '32-true', '16-mixed', '16-true', 'bf16-mixed', 'bf16-true'
]


class DistributorProtocol(Protocol):
    backend: Literal['none', 'fabric']
    global_rank: int
    local_rank: int
    world_size: int
    seed: int
    tracker_name: str
    device: torch.device

    @abstractmethod
    def launch(self) -> None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def compile(
            self,
            *module: TypeModule,
            enabled=True,
            mode='default',
    ) -> Optional[Union[TypeModule, Tuple[TypeModule, ...]]]:
        raise NotImplementedError

    @abstractmethod
    def backward(
            self,
            loss: torch.Tensor,
            model: Optional[torch.nn.Module] = None,
            **kwargs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def no_sync(
            self,
            *model: TypeModule,
            enabled=True
    ) -> ContextManager:
        raise NotImplementedError

    @abstractmethod
    def barrier(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def all_reduce(
            self, x: torch.Tensor, reduce_op='sum') -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def all_gather_object[T](self, x: T) -> List[T]:
        raise NotImplementedError

    @abstractmethod
    def broadcast_object(self, x: Any, src=0) -> Any:
        raise NotImplementedError

    @abstractmethod
    def unwrap_model(
            self, model: Union[torch.nn.Module, torch.optim.Optimizer]
    ) -> Union[torch.nn.Module, torch.optim.Optimizer]:
        raise NotImplementedError

    @abstractmethod
    def seed_everything(self, seed: int) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def collect_rng_states() -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_dataloader_worker_init_fn(self) -> Callable[[int], None]:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
            self,
            save_path: Path,
            state_groups: Dict[str, TypeCkptState],
            prefix='',
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(
            self,
            load_path: Path,
            state_groups: Dict[str, TypeCkptState],
            prefix='',
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_module(
            self, module: TypeModule, save_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_module(
            self, module: TypeModule, load_path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_main_process(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_distributed(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def initialize_tracker(
            self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def tracker_log(self, tag: str, value: float, step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_model(self, *args, **kwargs)  -> ContextManager:
        raise NotImplementedError

    @abstractmethod
    def setup_model(
            self,
            model: TypeModule,
            *optimizers: Optimizer,
    ) -> Union[TypeModule, Tuple[TypeModule, Optimizer, ...]]:
        raise NotImplementedError

    @abstractmethod
    def setup_optimizer(
            self, *optimizers: Optimizer
    ) -> Union[Optimizer, Tuple[Optimizer, ...]]:
        raise NotImplementedError

    @abstractmethod
    def shard_data_pool[T](
            self, data_pool: Sequence[T]
    ) -> Tuple[int, Union[Sequence[T], List[T]]]:
        raise NotImplementedError