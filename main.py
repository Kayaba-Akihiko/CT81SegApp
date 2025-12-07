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
import time

import numpy as np

from xmodules.logging import Logger
from xmodules.xutils import os_utils, dist_utils, lib_utils, metaimage_utils, dicom_utils
from xmodules.xdistributor import get_distributor
from xmodules.xqct2bmd.inferencer import Inferencer

HAS_FABRIC = lib_utils.import_available('lightning.fabric')

logging.setLoggerClass(Logger)

_logger = logging.getLogger(__name__)


def main():
    this_file = Path(__file__)
    this_dir = this_file.parent

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--image_path', type=str,
        help='CT image path. Can be a folder of dicom dataset, '
             'a single image in .mhd or mha, or a text file listing dicom files.'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default=None,
        help='Output directory. If not specified, the output will be the current working dir.'
    )
    parser.add_argument(
        '-s', '--resolution', type=int, default=512)
    parser.add_argument(
        '-z', '--norm_config_path', type=str)
    parser.add_argument(
        '-c', '--n_classes', type=int, default=81)
    parser.add_argument(
        '-b', '--batch_size', type=int, default=2,
        help='Batch size. Must be a positive integer',
    )
    parser.add_argument(
        '-r', '--rendering_config_path',
        type=str, default=None,
    )
    parser.add_argument(
        '--dicom_name_regex', type=str, default='.*\\.dcm$',
    )
    parser.add_argument(
        '-n', '--n_workers',
        type=int, default=min(4, os_utils.get_max_n_worker()),
    )
    parser.add_argument(
        '--dist_backend',
        type=str,
        default='fabric' if HAS_FABRIC else 'none',
    )
    parser.add_argument(
        '--dist_accelerator',
        type=str,
        default='auto',
    )
    parser.add_argument(
        '--dist_devices',
        type=str,
        default='auto',
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

    if distributor.is_main_process():
        config_log_str = (
                "\n---- Configuration ----\n" +
                json.dumps(vars(opt), indent=2) +
                "\n---- End of configuration ----"
        )
        _logger.info(config_log_str)

    total_start_time = time.perf_counter()
    image_path = os_utils.format_path_string(opt.image_path)
    model_path = opt.model_path
    if model_path is None:
        model_path = this_dir / 'nnunet_1res_ct_81_seg.onnx'
        _logger.info(f'Model path not specified. Using default: {model_path}.')
    else:
        model_path = os_utils.format_path_string(model_path)

    norm_config_path = opt.norm_config_path
    if norm_config_path is None:
        norm_config_path = this_dir / 'nnunet_1res_ct_81_seg_norm.json'
        _logger.info(f'Normalization config path not specified. Using default: {norm_config_path}.')
    else:
        norm_config_path = os_utils.format_path_string(norm_config_path)
    n_classes = int(opt.n_classes)
    resolution = opt.resolution
    batch_size = opt.batch_size
    rendering_config_path = opt.rendering_config_path
    if rendering_config_path is None:
        rendering_config_path = this_dir / 'naist_totalsegmentator_81.json'
        _logger.info(f'Rendering config path not specified. Using default: {rendering_config_path}.')
    else:
        rendering_config_path = os_utils.format_path_string(
            opt.rendering_config_path
        )
    n_workers = min(opt.n_workers, os_utils.get_max_n_worker())

    if distributor.device.type == 'cpu':
        onnx_providers = ['CPUExecutionProvider']
    elif distributor.device.type == 'cuda':
        onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        raise ValueError(f'Device {distributor.device.type} is not supported.')

    if distributor.is_main_process():
        _logger.info(f'Load model and normalization configuration')
    start_time = time.perf_counter()
    model_data = Inferencer.get_model(
        model_path=model_path,
        norm_config_path=norm_config_path,
        in_shape=(1, resolution, resolution),
        out_shape=(n_classes, resolution, resolution),
        onnx_providers=onnx_providers,
    )
    model_loading_time = time.perf_counter() - start_time
    if distributor.is_main_process():
        _logger.info(f'Model loading time: {model_loading_time:.2f} seconds.')

    image, spacing, position = None, None, None
    image_loading_time = None
    if distributor.is_main_process():
        _logger.info(f'Load image from {image_path} .')
        start_time = time.perf_counter()
        image, spacing, position = read_image(
            image_path,
            n_workers=n_workers,
            progress_bar=True,
            progress_desc='Reading image',
            dicom_name_regex=opt.dicom_name_regex,
        )
        image_loading_time = time.perf_counter() - start_time
        _logger.info(f'Image loading time: {image_loading_time:.2f} seconds.')
    if distributor.is_distributed():
        image, spacing, position, image_loading_time = distributor.broadcast_object(
            image, spacing, position, image_loading_time
        )

    if distributor.is_distributed():
        # Shard image
        if (n_slices := len(image)) < distributor.world_size:
            raise ValueError(f'Image size {len(image)} is less than world size {distributor.world_size}.')
        n_slices_per_rank = (n_slices + distributor.world_size - 1) // distributor.world_size
        start = distributor.global_rank * n_slices_per_rank
        end = start + n_slices_per_rank
        image = image[start: end]

    if distributor.is_main_process():
        _logger.info(f'Run model inference')

    start_time = time.perf_counter()
    pred_label = Inferencer.ct_inference_proxy(
        image=image,
        model_data=model_data,
        batch_size=batch_size,
        process_dtype='float32',
        prepro_device='auto',
        progress_bar=distributor.is_main_process(),
        progress_desc='Inferencing',
    )
    model_inference_time = time.perf_counter() - start_time
    del image
    if distributor.is_main_process():
        _logger.info(
            f'Model inference time: {model_inference_time:.2f} seconds.')

    if distributor.is_distributed():
        pred_label = (distributor.global_rank, pred_label)
        pred_labels = distributor.all_gather_object(pred_label)
        pred_label = None
        if distributor.is_main_process():
            pred_labels = sorted(pred_labels, key=lambda x: x[0])
            pred_labels = [x[1] for x in pred_labels]
            pred_label = np.concatenate(pred_labels, axis=0)
        del pred_labels

    prediction_saving_time = None
    if distributor.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        labelmap_save_path = output_dir / 'pred_label.mha'
        _logger.info(f'Save prediction to {labelmap_save_path} .')
        start_time = time.perf_counter()
        metaimage_utils.write(
            labelmap_save_path,
            pred_label, spacing=spacing, position=position
        )
        prediction_saving_time = time.perf_counter() - start_time
        _logger.info(f'Prediction saving time: {prediction_saving_time:.2f} seconds.')
    if distributor.is_distributed():
        prediction_saving_time = distributor.broadcast_object(prediction_saving_time)


def read_image(
        image_path: Path,
        n_workers: int = 4,
        progress_bar: bool = True,
        progress_desc='',
        dicom_name_regex='.*\\.dcm$',
):
    path_name = image_path.name
    if path_name.endswith('.mhd') or path_name.endswith('.mha'):
        # (N, H, W)
        image, spacing, position = metaimage_utils.read(image_path)
    else:
        image, spacing, position = dicom_utils.read_dicom_folder(
            image_path,
            name_regex=dicom_name_regex,
            n_workers=n_workers,
            progress_bar=progress_bar,
            progress_desc=progress_desc,
        )
    return image, spacing, position

if __name__ == '__main__':
    main()