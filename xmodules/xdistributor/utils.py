#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import torch
import logging
import numpy as np
import random
from typing import List

_logger = logging.getLogger(__name__)

def pl_worker_init_function(worker_id: int, rank: int) -> None:  # pragma: no cover
    r"""The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.

    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.

    """
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # global_rank = rank if rank is not None else rank_zero_only.rank
    global_rank = rank
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    _logger.debug(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    seed_sequence = _generate_seed_sequence(base_seed, worker_id, global_rank, count=4)
    torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
    random.seed((seed_sequence[1] << 32) | seed_sequence[2])  # combine two 64-bit seeds
    np.random.seed(seed_sequence[3] & 0xFFFFFFFF)  # numpy takes 32-bit seed only


def _generate_seed_sequence(base_seed: int, worker_id: int, global_rank: int, count: int) -> List[int]:
    """Generates a sequence of seeds from a base seed, worker id and rank using the linear congruential generator (LCG)
    algorithm."""
    # Combine base seed, worker id and rank into a unique 64-bit number
    combined_seed = (base_seed << 32) | (worker_id << 16) | global_rank
    seeds = []
    for _ in range(count):
        # x_(n+1) = (a * x_n + c) mod m. With c=1, m=2^64 and a is D. Knuth's constant
        combined_seed = (combined_seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
        seeds.append(combined_seed)
    return seeds