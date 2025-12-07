#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


import logging
import tomllib
from pathlib import Path

from ..protocol import TypeConfig


_logger = logging.getLogger(__name__)

def load_config(conf_path: Path) -> TypeConfig:
    configs = []
    with open(conf_path, 'rb') as f:
        conf = tomllib.load(f)
    configs.append(conf)

    while conf.get('base', None) is not None:
        conf_path = conf_path.parent / conf.pop('base')
        _logger.info(
            f'Load base {len(configs)} config '
            f'at {conf_path.resolve()}.'
        )
        with open(conf_path, 'rb') as f:
            conf = tomllib.load(f)
        configs.append(conf)

    conf = configs.pop()
    while len(configs) > 0:
        _logger.info(f'Update base config {len(configs)}.')
        conf = update_base_config(conf, configs.pop())
    return conf


def update_base_config(
        base_config: TypeConfig, current_config: TypeConfig):

    for curr_k, curr_v in current_config.items():
        if curr_k not in base_config:
            base_config[curr_k] = curr_v
            continue

        base_v = base_config[curr_k]
        if isinstance(base_v, dict) and isinstance(curr_v, dict):
            base_config[curr_k] = update_base_config(base_v, curr_v)
            continue

        base_config[curr_k] = curr_v
    return base_config