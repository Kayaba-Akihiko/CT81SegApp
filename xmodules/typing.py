#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Union, Dict, Any, TypeAlias
from pathlib import Path
import os

AnyStr: TypeAlias = Union[bytes, str]
TypePathLike: TypeAlias = Union[AnyStr, os.PathLike[AnyStr], Path]

TypeConfig: TypeAlias = Dict[str, Any]