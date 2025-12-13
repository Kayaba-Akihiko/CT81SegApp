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
from typing import Tuple, List, Dict, Union, Literal

import numpy as np
import numpy.typing as npt

from xmodules.logging import Logger
from xmodules.xutils import os_utils, lib_utils, metaimage_utils, dicom_utils, array_utils as xp
from xmodules.xdistributor import get_distributor, DistributorProtocol
from xmodules.xqct2bmd.inferencer import Inferencer

from modules.report_generator import ReportGenerator, PatientInfoData

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
        '--image_process_device',
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

    distributor = get_distributor(
        opt.dist_backend,
        seed=831,
        accelerator=opt.dist_accelerator,
        devices=opt.dist_devices,
    )
    _logger.info(f'Launch distributor {distributor.backend}')
    distributor.launch()
    _logger.info(f'Distributor launched.')

    total_time_start = None
    if distributor.is_main_process():
        total_time_start = time.perf_counter()

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = os_utils.format_path_string(output_dir)
    if distributor.is_main_process():
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

    if distributor.is_main_process():
        config_log_str = (
                "\n---- Configuration ----\n" +
                json.dumps(vars(opt), indent=2) +
                "\n---- End of configuration ----"
        )
        _logger.info(config_log_str)

    if distributor.is_main_process():
        _logger.info(f'Using distributor accelerator {opt.accelerator}')
        _logger.info(f'Using distributor devices: {opt.devices}')

    image_path = os_utils.format_path_string(opt.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image path {image_path} not found.')
    model_name = opt.model
    if model_name is None:
        model_name = 'nnunet_1res_ct_81_seg'
    if distributor.is_main_process():
        _logger.info(f'Using model {model_name}.')
    model_load_path = this_dir / 'resources' / f'{model_name}.onnx'
    if not model_load_path.exists():
        raise FileNotFoundError(
            f'Model {model_name} not found: {model_load_path}')
    norm_config_load_path = this_dir / 'resources' / f'{model_name}.json'
    if not norm_config_load_path.exists():
        raise FileNotFoundError(
            f'Normalization config {model_name} not found: {norm_config_load_path}')

    n_classes = 81
    resolution = 512
    batch_size = opt.batch_size
    if batch_size <= 0:
        raise ValueError(f'Invalid batch size: {batch_size=}')

    rendering_config_path = this_dir / 'resources' / 'rendering_config.json'
    if not rendering_config_path.exists():
        raise FileNotFoundError(
            f'Rendering config not found: {rendering_config_path}')

    n_workers = max(0, min(opt.n_workers, os_utils.get_max_n_worker()))
    if distributor.is_main_process():
        _logger.info(f'Using {n_workers} workers.')

    device = opt.device
    if distributor.is_main_process():
        _logger.info(f'Using device {device}.')
    if device == 'cpu':
        onnx_providers = ['CPUExecutionProvider']
    elif device == 'cuda':
        onnx_providers = [
            ('CUDAExecutionProvider', {"device_id": distributor.local_rank}),
        ]
    else:
        raise ValueError(
            f'Device {distributor.device.type} is not supported.')
    image_process_device = opt.image_process_device
    if device == 'cpu':
        image_process_device = 'cpu'

    # ----
    # Load every things
    # ----
    # Load model
    model_data_load_time_start = None
    if distributor.is_main_process():
        model_data_load_time_start = time.perf_counter()
    model_data = Inferencer.get_model(
        model_path=model_load_path,
        norm_config_path=norm_config_load_path,
        in_shape=(1, resolution, resolution),
        out_shape=(n_classes, resolution, resolution),
        onnx_providers=onnx_providers,
    )
    if distributor.is_main_process():
        model_data_load_time = time.perf_counter() - model_data_load_time_start
        _logger.info(f'Model loading time: {model_data_load_time:.2f} seconds.')
    else:
        model_data_load_time = None

    image, spacing, position, patient_info = None, None, None, None

    _logger.info(f'Load image from {image_path} .')
    image_load_time_start = time.perf_counter()
    if distributor.is_main_process():
        image, spacing, position, patient_info = read_image(
            image_path,
            n_workers=n_workers,
            progress_bar=True,
            progress_desc='Reading image',
            dicom_name_regex=opt.dicom_name_regex,
        )
    if distributor.is_distributed():
        image, spacing, position, patient_info = distributor.broadcast_object(
            image, spacing, position, patient_info
        )
    image_load_time = None
    if distributor.is_main_process():
        image_load_time = time.perf_counter() - image_load_time_start
        _logger.info(f'Image loading time: {image_load_time:.2f} seconds.')

    if distributor.is_distributed():
        # Shard image
        if (n_slices := len(image)) < distributor.world_size:
            raise ValueError(f'Image size {len(image)} is less than world size {distributor.world_size}.')
        n_slices_per_rank = (n_slices + distributor.world_size - 1) // distributor.world_size
        start = distributor.global_rank * n_slices_per_rank
        end = start + n_slices_per_rank
        image = image[start: end]

    prepro_device: Literal['cpu', 'cuda'] = 'cpu'
    if image_process_device == 'cuda':
        if xp.HAS_CUPY and xp.CUPY_CUDA_AVAILABLE:
            image = xp.to_cupy(image)
            prepro_device = 'cuda'

    if distributor.is_main_process():
        _logger.info(f'Run model inference')

    start_time = None
    if distributor.is_main_process():
        start_time = time.perf_counter()
    pred_label = Inferencer.ct_inference_proxy(
        image=image,
        model_data=model_data,
        batch_size=batch_size,
        process_dtype='float32',
        prepro_device=prepro_device,
        out_device=image_process_device,
        progress_bar=distributor.is_main_process(),
        progress_desc='Inferencing',
    )
    if distributor.is_distributed():
        pred_label = xp.concatenate(_gather_in_rank_order(distributor, pred_label))
    model_inference_time = None
    if distributor.is_main_process():
        model_inference_time = time.perf_counter() - start_time
        _logger.info(
            f'Model inference time: {model_inference_time:.2f} seconds.')

    del model_data

    if distributor.is_distributed():
        image = xp.concatenate(_gather_in_rank_order(distributor, image))

    mean_hu_calc_time_start = None
    if distributor.is_main_process():
        mean_hu_calc_time_start = time.perf_counter()
    if image_process_device == 'cpu':
        hus, voxel_counts = _compute_class_mean_hu_cpu(
            image, pred_label, n_classes=n_classes)
    elif image_process_device == 'cuda':
        image = xp.to_cuda(image)
        pred_label = xp.to_cuda(pred_label)
        hus, voxel_counts = _compute_class_mean_hu_cuda(
            distributor, image, pred_label, n_classes=n_classes)
        hus = xp.to_numpy(hus)
        voxel_counts = xp.to_numpy(voxel_counts)
    else:
        raise ValueError(f'Invalid image_process_device: {image_process_device}')
    mean_hu_calc_time = None
    if distributor.is_main_process():
        mean_hu_calc_time = time.perf_counter() - mean_hu_calc_time_start
        _logger.info(f'Class mean HU calculation time: {mean_hu_calc_time:.2f} seconds.')



def read_image(
        image_path: Path,
        n_workers: int = 4,
        progress_bar: bool = True,
        progress_desc='',
        dicom_name_regex='.*\\.dcm$',
) -> Tuple[
    npt.NDArray[xp.NPIntOrFloat],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    PatientInfoData,
]:
    path_name = image_path.name
    if path_name.endswith('.mhd') or path_name.endswith('.mha'):
        # (N, H, W)
        image, spacing, position = metaimage_utils.read(image_path)
        patient_info = PatientInfoData()
    else:
        tag_name_map = {
            'name': (0x0010, 0x0010),
            'sex': (0x0010, 0x0040),
            'age': (0x0010, 0x1010)
        }
        image, spacing, position, tag_res = dicom_utils.read_dicom_folder(
            image_path,
            name_regex=dicom_name_regex,
            n_workers=n_workers,
            progress_bar=progress_bar,
            progress_desc=progress_desc,
            required_tag=list(tag_name_map.values())
        )
        creation_kwargs = {}
        for tag_name, tag_hex in tag_name_map.items():
            creation_kwargs[tag_name] = tag_res[tag_hex]
        patient_info = PatientInfoData.from_dict(**creation_kwargs)

    if position is None:
        position = np.zeros(len(spacing), dtype=np.float64)
    return image, spacing, position, patient_info


def _compute_class_mean_hu_cpu(
        image: npt.NDArray,
        labelmap: npt.NDArray[np.integer],
        n_classes: int = 81,
        process_dtype: str = 'float32',
):
    hu_results, voxel_results = [], []
    for class_id in range(n_classes):
        mask = (labelmap == class_id)
        if not mask.any():
            hu_results.append(None)
            voxel_results.append(0)
            continue
        hu_results.append(image[mask].astype(process_dtype, copy=False).mean())
        voxel_results.append(mask.sum())
    hu_results = np.array(hu_results, dtype=np.float64)
    voxel_results = np.array(voxel_results, dtype=np.int64)
    return hu_results, voxel_results

def _compute_class_mean_hu_cuda(
        distributor: DistributorProtocol,
        image: xp.TypeArrayLike,
        labelmap: xp.TypeArrayLike[np.integer],
        n_classes: int = 81,
        process_dtype: str = 'float32',
):
    classes = list(range(n_classes))
    if distributor.is_distributed():
        # shard class_jobs
        if n_classes < distributor.world_size:
            raise ValueError(f'Number of classes {n_classes} is less than world size {distributor.world_size}.')
        n_classes_per_rank = (n_classes + distributor.world_size - 1) // distributor.world_size
        start = distributor.global_rank * n_classes_per_rank
        end = start + n_classes_per_rank
        classes = classes[start: end]

    hu_results, voxel_results = [], []
    for class_id in classes:
        mask = (labelmap == class_id)
        if not mask.any():
            hu_results.append(None)
            voxel_results.append(0)
            continue
        hu_results.append(xp.to(image[mask], dtype=process_dtype).mean())
        voxel_results.append(mask.sum())
    if distributor.is_distributed():
        hu_results = sum(_gather_in_rank_order(distributor, hu_results), [])
    hu_results = xp.to_dst(hu_results, dst=image, dtype=np.float64)
    voxel_results = xp.to_dst(voxel_results, dst=image, dtype=np.int64)
    return hu_results, voxel_results


def _gather_in_rank_order(distributor, x):
    # Pack
    x = (distributor.global_rank, x)
    xs = distributor.all_gather_object(x)
    xs = sorted(xs, key=lambda _x: _x[0])
    # Unpack
    xs = [x[1] for x in xs]
    return xs


if __name__ == '__main__':
    main()