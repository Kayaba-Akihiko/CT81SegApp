#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import (
    Union, Dict, Any, TypeAlias, Protocol, runtime_checkable, TypeVar)
from abc import abstractmethod
from pathlib import Path
import os

_DictKey = TypeVar("_DictKey")

AnyStr: TypeAlias = Union[bytes, str]
TypePathLike: TypeAlias = Union[Union[AnyStr, os.PathLike[AnyStr]], Path]

TypeConfig: TypeAlias = Dict[str, Any]

@runtime_checkable
class Stateful(Protocol[_DictKey]):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    @abstractmethod
    def state_dict(self) -> dict[_DictKey, Any]: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[_DictKey, Any]) -> None: ...
