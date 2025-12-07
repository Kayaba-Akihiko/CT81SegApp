#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import functools
from importlib import import_module
from importlib.util import resolve_name
from typing import Tuple, Union, Optional
from types import ModuleType


class LibUtils:
    @staticmethod
    @functools.lru_cache()
    def import_health(
            name: str, package: Optional[str] = None,
    ) -> Tuple[Optional[ModuleType], str]:
        """
        Import a module safely.

        Supports both:
            import_health("torch.nn.functional")
            import_health("functional", package="torch.nn")

        Returns (module_or_None, reason):
          - "ok"                         -> imported successfully
          - "not_found"                  -> the target module or a prefix not found
          - "dependency_missing: <name>" -> missing dependency during import
          - "invalid_relative: <Exc>"    -> relative import invalid for given package
          - "import_error: <ExcType>"    -> other import-time error
        """
        # Normalize the target name
        if package and not name.startswith((".", package)):
            # If user passed "functional" with package="torch.nn", treat it like ".functional"
            qualified_name = f".{name}"
        else:
            qualified_name = name

        # Resolve to fully qualified name for consistent error checks
        try:
            target_fullname = resolve_name(qualified_name, package) if qualified_name.startswith(".") else qualified_name
        except (TypeError, ValueError) as e:
            return None, f"invalid_relative: {type(e).__name__}"

        try:
            module = import_module(qualified_name, package)
            return module, "ok"

        except ModuleNotFoundError as e:
            missing = e.name
            if missing == target_fullname or target_fullname.startswith(missing + "."):
                return None, "not_found"
            else:
                return None, f"dependency_missing: {missing}"

        except Exception as e:
            return None, f"import_error: {type(e).__name__}"

    @classmethod
    @functools.lru_cache()
    def try_import(cls, mod_name: str, package: Optional[str] = None, ) -> Union[ModuleType, None]:
        module, status = cls.import_health(mod_name, package=package)
        return module

    @classmethod
    @functools.lru_cache()
    def import_available(cls, mod_name: str, package: Optional[str] = None, ) -> bool:
        module, status = cls.import_health(mod_name, package=package)
        return status == "ok"


import_health = LibUtils.import_health
try_import = LibUtils.try_import
import_available = LibUtils.import_available