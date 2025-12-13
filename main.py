#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

__version__ = '0.0.1'

import json
import logging
import argparse
from pathlib import Path
import time
import copy
import traceback

import numpy as np

from xmodules.logging import Logger
from xmodules.xutils import os_utils, lib_utils, metaimage_utils, dicom_utils, array_utils as xp
from xmodules.xdistributor import get_distributor
from xmodules.xqct2bmd.inferencer import Inferencer

from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator, XLAAccelerator

HAS_FABRIC = lib_utils.import_available('lightning.fabric')

logging.setLoggerClass(Logger)

_logger = logging.getLogger(__name__)


def main():
    this_file = Path(__file__)
    this_dir = this_file.parent

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        '--image_path', type=str,
        help='CT image path. Can be a folder of dicom dataset, '
             'a single image in .mhd or mha, or a text file listing dicom files.'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory. If not specified, the output will be the current working dir.'
    )
    parser.add_argument(
        '--model', type=str, default=None)
    parser.add_argument(
        '--batch_size', type=int, default=2,
        help='Batch size. Must be a positive integer',
    )
    parser.add_argument(
        '--dicom_name_regex', type=str, default='.*\\.dcm$',
    )
    parser.add_argument(
        '--n_workers',
        type=int, default=min(4, os_utils.get_max_n_worker()),
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda' if xp.is_cuda_available() else 'cpu',
    )
    parser.add_argument(
        '--dist_backend',
        type=str,
        choices=['none', 'fabric'],
        default='fabric' if HAS_FABRIC else 'none',
    )
    parser.add_argument(
        '--dist_accelerator',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
    )
    parser.add_argument(
        '--dist_devices',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-l', '--logging_level', type=str, default='INFO',
        help='INFO, DEBUG ...',
    )

    opt = parser.parse_args()

    logging.basicConfig(
        level=opt.logging_level,
        format=(
            '[%(asctime)s][%(levelname)s][%(name)s] '
            '- %(message)s'
        ),
    )

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = os_utils.format_path_string(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    distributor = get_distributor(
        opt.dist_backend,
        seed=831,
        accelerator=opt.dist_accelerator,
        devices=opt.dist_devices,
    )
    distributor.launch()

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