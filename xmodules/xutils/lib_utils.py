#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import functools
from importlib import import_module
from importlib.util import resolve_name
from typing import Tuple, Union, Optional, Any
from types import ModuleType


class LibUtils:
    @staticmethod
    @functools.lru_cache()
    def import_health(
        name: str, package: Optional[str] = None,
    ) -> Tuple[Optional[Any], str]:
        """
        Import a module or attribute safely.

        Supports:
            import_health("torch.nn.functional")
            import_health("functional", package="torch.nn")
            import_health("lightning.Fabric")
            import_health("Fabric", package="lightning")

        Returns (obj_or_None, reason):
          - "ok"                         -> imported successfully (module or attribute)
          - "not_found"                  -> the target (module/attr) or a prefix not found
          - "dependency_missing: <name>" -> missing dependency during import
          - "invalid_relative: <Exc>"    -> relative import invalid for given package
          - "import_error: <ExcType>"    -> other import-time error
        """
        # Normalize the target name for resolve_name/import_module
        if package and not name.startswith((".", package)):
            # If user passed "functional" with package="torch.nn", treat it like ".functional"
            qualified_name = f".{name}"
        else:
            qualified_name = name

        # Resolve to fully qualified name for consistent error checks
        try:
            if qualified_name.startswith("."):
                target_fullname = resolve_name(qualified_name, package)
            else:
                target_fullname = qualified_name
        except (TypeError, ValueError) as e:
            return None, f"invalid_relative: {type(e).__name__}"

        # Helper: try interpreting target_fullname as "<parent>.<attr>"
        def _try_attribute_import(fullname: str):
            if "." not in fullname:
                return None, "not_found"

            parent_name, attr_name = fullname.rsplit(".", 1)
            try:
                parent_mod = import_module(parent_name)
            except ModuleNotFoundError:
                # parent module itself not importable -> treat as not_found
                return None, "not_found"
            except Exception as e:
                return None, f"import_error: {type(e).__name__}"

            if not hasattr(parent_mod, attr_name):
                return None, "not_found"

            try:
                return getattr(parent_mod, attr_name), "ok"
            except Exception as e:
                return None, f"import_error: {type(e).__name__}"

        # First attempt: treat the name as a module
        try:
            obj = import_module(qualified_name, package)
            return obj, "ok"

        except ModuleNotFoundError as e:
            # Fallback: treat as attribute path on a parent module
            attr_obj, attr_status = _try_attribute_import(target_fullname)
            if attr_status == "ok":
                return attr_obj, "ok"

            # Attribute fallback failed → classify original error
            missing = e.name
            if missing == target_fullname or target_fullname.startswith(missing + "."):
                return None, "not_found"
            else:
                return None, f"dependency_missing: {missing}"

        except Exception as e:
            return None, f"import_error: {type(e).__name__}"

    @classmethod
    @functools.lru_cache()
    def try_import(cls, mod_name: str, package: Optional[str] = None, ) -> Union[Any, None]:
        obj, status = cls.import_health(mod_name, package=package)
        return obj if status == "ok" else None

    @classmethod
    @functools.lru_cache()
    def import_available(cls, mod_name: str, package: Optional[str] = None, ) -> bool:
        _, status = cls.import_health(mod_name, package=package)
        return status == "ok"


import_health = LibUtils.import_health
try_import = LibUtils.try_import
import_available = LibUtils.import_available