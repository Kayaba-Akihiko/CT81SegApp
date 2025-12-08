#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import functools
import os
import shutil
from typing import Callable, Iterable, Union, List, Tuple, Optional, Sequence

from ..protocol import TypePathLike
from pathlib import Path
import re


class OSUtils:

    @staticmethod
    @functools.lru_cache()
    def get_max_n_worker() -> int:
        max_num_worker_suggest = 0
        if hasattr(os, 'sched_getaffinity'):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
            except Exception:
                pass
        if max_num_worker_suggest == 0:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satify mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if "SLURM_CPUS_PER_TASK" in os.environ:
            max_num_worker_suggest = int(os.environ["SLURM_CPUS_PER_TASK"])
        return max_num_worker_suggest

    @staticmethod
    def copy(src: TypePathLike, dst: TypePathLike, *, follow_symlinks=True):
        return shutil.copy(src, dst, follow_symlinks=follow_symlinks)

    @staticmethod
    def copyfile(src: TypePathLike, dst: TypePathLike, *, follow_symlinks=True):
        return shutil.copyfile(src, dst, follow_symlinks=follow_symlinks)

    @staticmethod
    def copy2(src: TypePathLike, dst: TypePathLike, *, follow_symlinks=True):
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)

    @classmethod
    def format_path_string(
            cls,
            path: Union[
                TypePathLike,
                List[TypePathLike],
                Tuple[TypePathLike, ...],
            ], src_sep='\\'
    ) -> Union[
        Path,
        List[Path],
    ]:
        if isinstance(path, (tuple, list)):
            if len(path) == 0:
                return []
            res: List[Path] = []
            for p in path:
                assert isinstance(p, (str, Path)), f'Unsupported type of path ({type(p)}) in list'
                p = cls.format_path_string(p)
                assert isinstance(p, Path), f'Expected Path after formatting, got {type(p)}'
                res.append(p)
            return res

        if isinstance(path, Path):
            ret = str(path)
        elif isinstance(path, str):
            ret = path.replace(src_sep, '/')
        else:
            raise TypeError(f"Unsupported type of path ({type(path)})")
        for server in ["conger", "scallop", "salmon", "flounder"]:
            ret = re.sub(
                f"^//{server}/user", f"/win/{server}/user", ret,
                flags=re.IGNORECASE)
        return Path(ret)

    @staticmethod
    def _scan_dirs(
            paths: TypePathLike | list[TypePathLike] | tuple[TypePathLike, ...],
            allow_fun: Callable[[os.DirEntry], bool],
            name_regex: Optional[Union[str, re.Pattern]] = None,
            recursive: bool = False,
            follow_symlinks=False,
    ) -> Iterable[os.DirEntry]:
        if not isinstance(paths, list) and not isinstance(paths, tuple):
            paths = [paths]
        if name_regex is None:
            name_regex = re.compile(".*")
        elif isinstance(name_regex, str):
            name_regex = re.compile(name_regex)
        if not isinstance(name_regex, re.Pattern):
            raise RuntimeError(
                f"Unsupported type of name_re_pattern ({type(name_regex)}) ")
        #
        # for path in paths:
        #     with scandir(path) as it:
        #         entry: os.DirEntry
        #         for entry in it:
        #             name_match = name_re_pattern.match(entry.name) is not None
        #             if allow_fun(entry) and name_match:
        #                 yield entry
        #
        def _walk_dir(root: TypePathLike) -> Iterable[os.DirEntry]:
            with os.scandir(root) as it:
                for entry in it:
                    if allow_fun(entry) and name_regex.match(entry.name):
                        yield entry
                    if recursive and entry.is_dir(follow_symlinks=follow_symlinks):
                        yield from _walk_dir(entry.path)

        for path in paths:
            yield from _walk_dir(path)

    @classmethod
    def scan_dirs_for_file(
            cls,
            paths: Union[
                TypePathLike,
                List[TypePathLike],
                Tuple[TypePathLike, ...],
                Sequence[TypePathLike]
            ],
            name_regex: Optional[Union[str, re.Pattern]] = None,
            recursive: bool = False,
            follow_symlinks=False,
    ) -> Iterable[os.DirEntry]:
        def allow_file(entry: os.DirEntry) -> bool:
            return entry.is_file()

        return cls._scan_dirs(
            paths=paths,
            allow_fun=allow_file,
            name_regex=name_regex,
            recursive=recursive,
            follow_symlinks=follow_symlinks,
        )

    @classmethod
    def scan_dirs_for_folder(
            cls,
            paths: Union[
                Union[TypePathLike, List[TypePathLike]],
                Tuple[TypePathLike, ...]
            ],
            name_regex: Optional[Union[str, re.Pattern]] = None,
            recursive: bool = False,
            follow_symlinks=False,
    ) -> Iterable[os.DirEntry]:
        def allow_dir(entry: os.DirEntry) -> bool:
            return entry.is_dir()

        return cls._scan_dirs(
            paths=paths,
            allow_fun=allow_dir,
            name_regex=name_regex,
            recursive=recursive,
            follow_symlinks=follow_symlinks,
        )

get_max_n_worker = OSUtils.get_max_n_worker
copy = OSUtils.copy
copyfile = OSUtils.copyfile
copy2 = OSUtils.copy2
format_path_string = OSUtils.format_path_string
scan_dirs_for_file = OSUtils.scan_dirs_for_file
scan_dirs_for_folder = OSUtils.scan_dirs_for_folder