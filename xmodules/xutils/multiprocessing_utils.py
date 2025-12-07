#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Any, Callable, Generator, Optional, Dict, Union, Literal, TypeAlias, Sequence, Iterable

import multiprocessing.pool as mpp

from .lib_utils import import_available
from ..tqdm import tqdm

if HAS_TORCH := import_available('torch'):
    TypeMPBackend: TypeAlias = Literal['torch', 'python']
else:
    TypeMPBackend: TypeAlias = Literal['python']
TypeMPContext: TypeAlias = Literal['spawn', 'fork', 'forkserver']


class MultiProcessingUtils:

    @staticmethod
    def _function_proxy(fun: Callable, kwargs: Dict[str, Any]) -> Any:
        return fun(**kwargs)

    @classmethod
    def run_jobs(
            cls,
            args: Union[Sequence[Any], Iterable[Any]],
            n_workers: int,
            func: Optional[Callable] = None,
            mp_backend: TypeMPBackend = 'torch' if HAS_TORCH else 'python',
            mp_context: Optional[TypeMPContext] = None,
            progress_bar=True,
            progress_desc: Optional[str] = None,
            progress_mininterval: Optional[float] = 0.1,
            progress_maxinterval: Optional[float] = 10,
            total: Optional[int] = None,
    ):
        tqdm_args: Dict[str, Any] = {"desc": progress_desc}
        if total is not None:
            tqdm_args['total'] = total
        if isinstance(args, Sequence):
            tqdm_args['total'] = len(args)
        if progress_mininterval is not None:
            tqdm_args["mininterval"] = progress_mininterval
        if progress_maxinterval is not None:
            tqdm_args["maxinterval"] = progress_maxinterval

        exec_fun = cls._function_proxy if func is None else func
        if n_workers > 0:

            if HAS_TORCH and mp_backend == 'torch':
                import torch.multiprocessing as mp
                from torch.multiprocessing import Pool
            elif mp_backend == 'python':
                import multiprocessing as mp
                from multiprocessing import Pool
            else:
                raise ValueError(
                    f"Unknown multiprocessing backend: {mp_backend}. "
                    f"Expected 'torch' or 'python'."
                )

            if mp_context is not None:
                Pool = mp.get_context(mp_context).Pool

            with Pool(n_workers) as pool:
                provider = pool.istarmap(exec_fun, iterable=args)
                if progress_bar:
                    provider = tqdm(provider, **tqdm_args)
                for data in provider:
                    yield data
        else:
            iterator = args
            if progress_bar:
                iterator = tqdm(args, **tqdm_args)
            for item in iterator:
                yield exec_fun(*item)

def _istarmap(
        self: mpp.Pool, func: Callable, iterable: Iterable, chunksize=1
) -> Generator[Any, Any, None]:
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    task_data = (
        self._guarded_task_generation(
            result._job, mpp.starmapstar, task_batches),
        result._set_length)
    self._taskqueue.put(task_data)
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = _istarmap

run_jobs = MultiProcessingUtils.run_jobs