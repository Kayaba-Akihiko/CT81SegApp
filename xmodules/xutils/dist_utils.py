#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from socket import gethostname
import os

from .lib_utils import import_available

dist = None
if HAS_TORCH := import_available('torch'):
    import torch.distributed as dist

def hostname() -> str:
    return gethostname()

def is_main_proc() -> bool:
    assert is_initialized(), f'Distributed is not initialized.'
    return dist.get_rank() == 0

def is_initialized() -> bool:
    # `is_initialized` is only defined conditionally
    # https://github.com/pytorch/pytorch/blob/v2.1.0/torch/distributed/__init__.py#L25
    # this might happen to MacOS builds from source (default) or any build from source that sets `USE_DISTRIBUTED=0`
    assert HAS_TORCH, "PyTorch is not available."
    return dist.is_available() and dist.is_initialized()

def get_global_rank() -> int:
    assert HAS_TORCH, "PyTorch is not available."
    return dist.get_rank()


def get_slurm_job_id() -> str:
    return os.environ['SLURM_JOB_ID']


def is_running_on_slurm() -> bool:
    return 'SLURM_JOB_ID' in os.environ