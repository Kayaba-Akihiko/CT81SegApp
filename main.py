#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import json
import logging
import argparse
import os
from pathlib import Path

from xmodules.logging import Logger
from xmodules.xutils import os_utils, dist_utils
from xmodules.xdistributor import get_distributor

logging.setLoggerClass(Logger)

_logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-o', '--output_dir', type=str, default=None,
        help='Output directory. If not specified, the output will be the current working dir.'
    )
    parser.add_argument(
        '-l', '--logging_level', type=str, default='INFO',
        help='INFO, DEBUG ...',
    )
    parser.add_argument(
        '--dist_backend', type=str, default='none')

    opt = parser.parse_args()

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = os_utils.format_path_string(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=opt.logging_level,
        format=(
            '[%(asctime)s][%(levelname)s][%(name)s] '
            '- %(message)s'
        ),
        handlers=[
            logging.FileHandler(output_dir / 'inference.log'),
            logging.StreamHandler(),
        ],
        force=True,
    )


if __name__ == '__main__':
    main()