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
import traceback
from typing import Tuple, Literal, OrderedDict
import gc

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from xmodules.logging import Logger
from xmodules.xutils import os_utils, lib_utils, metaimage_utils, dicom_utils, array_utils as xp
from xmodules.xdistributor import get_distributor
from xmodules.xqct2bmd.inferencer import Inferencer

from modules.report_generator import ReportGenerator, PatientInfoData

HAS_FABRIC = lib_utils.import_available('lightning.fabric')
THIS_FILE = Path(__file__)
THIS_DIR = THIS_FILE.parent

logging.setLoggerClass(Logger)
_logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            '--version', action='version', version=f'{__version__}')

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
            default='cuda' if ort.get_device().lower() == 'gpu' else 'cpu',
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
            '--report_dpi',
            type=int, default=200,
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

        _logger.info('Initializing ...')
        distributor = get_distributor(
            opt.dist_backend,
            seed=831,
            accelerator=opt.dist_accelerator,
            devices=opt.dist_devices,
        )
        _logger.info(f'Launching distributor {distributor.backend}')
        distributor.launch()

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
            _logger.info(f'Distributor launched.')

            config_log_str = (
                    "\n---- Configuration ----\n" +
                    json.dumps(vars(opt), indent=2) +
                    "\n---- End of configuration ----"
            )
            _logger.info(config_log_str)

            if distributor.is_distributed():
                _logger.info(f'Using distributed mode with {distributor.world_size} processes.')

            _logger.info(f'Using distributor accelerator {opt.dist_accelerator}')
            _logger.info(f'Using distributor devices: {opt.dist_devices}')

        self._opt = opt
        self._distributor = distributor
        self._output_dir = output_dir
        self._time_summary = {}

    def run(self):
        if self._distributor.is_main_process():
            _logger.info('Start running ...')
        try:
            self._run_body()
        except Exception as e:
            error_message = f'{e}\n{traceback.format_exc()}'
            _logger.error(error_message)
        if self._distributor.is_main_process():
            time_summary = OrderedDict()
            keys = [
                'Loading config',
                'Loading model',
                'Loading image',
                'HU calculation',
                'Model Inference',
                'Rendering report',
                'Saving HU table',
                'Saving labelmap',
                'Saving report'
            ]
            for k in keys:
                if k in self._time_summary:
                    time_summary[k] = self._time_summary[k]
                else:
                    time_summary[k] = None
            _logger.info(
                f'\n ---- Time summary ----\n'
                f'{json.dumps(time_summary, indent=2)}'
                f'\n ---- End of time summary ----'
            )

            _logger.info('Done.')

    def _run_body(self):
        distributor = self._distributor
        opt = self._opt
        output_dir = self._output_dir

        total_time_start = None
        if distributor.is_main_process():
            total_time_start = time.perf_counter()

        resources_root = THIS_DIR / 'resources'
        self._check_path_exists(resources_root)

        image_path = os_utils.format_path_string(opt.image_path)
        self._check_path_exists(image_path)
        model_name = opt.model
        if model_name is None:
            model_name = 'nnunet_1res_ct_81_seg'
        if distributor.is_main_process():
            _logger.info(f'Using model {model_name}.')

        n_classes = 81
        resolution = 512
        batch_size = opt.batch_size
        if batch_size <= 0:
            raise ValueError(f'Invalid batch size: {batch_size=}')

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

        # Load configuration
        template_path = resources_root / 'MICBON_AI_report_template_p3.pptx'
        hu_statistics_table_path = resources_root / 'hu_statistics.xlsx'
        rendering_config = resources_root / 'rendering_config.toml'
        class_table_path = resources_root / 'class_table.csv'
        class_groups_path = resources_root / 'class_groups.json'
        observation_messages_path = resources_root / 'observation_messages_jp.toml'
        self._check_path_exists(
            template_path, hu_statistics_table_path, rendering_config,
            class_table_path, class_groups_path, observation_messages_path,
        )
        config_load_time_start = None
        if distributor.is_main_process():
            config_load_time_start = time.perf_counter()
            _logger.info('Loading configuration...')
        report_generator = ReportGenerator(
            distributor=distributor,
            template_ppt=template_path,
            hu_statistics_table=hu_statistics_table_path,
            rendering_config=rendering_config,
            class_info_table=class_table_path,
            class_groups=class_groups_path,
            observation_messages=observation_messages_path,
        )
        config_load_time = None
        if distributor.is_main_process():
            config_load_time = time.perf_counter() - config_load_time_start
            _logger.info(f'Config loading time: {config_load_time:.2f} seconds.')
            self._time_summary['Loading config'] = config_load_time

        # Load model
        if distributor.is_main_process():
            _logger.info(f'ONNX Runtime device: {ort.get_device()}.')
            _logger.info(f'Available execution providers: {ort.get_available_providers()}.')
        model_load_path = resources_root / f'{model_name}.onnx'
        norm_config_load_path = resources_root / f'{model_name}_norm.json'
        self._check_path_exists(model_load_path, norm_config_load_path)
        model_data_load_time_start = None
        if distributor.is_main_process():
            model_data_load_time_start = time.perf_counter()
        _logger.info(f'Loading model ...')
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
            self._time_summary['Loading model'] = model_data_load_time
        else:
            model_data_load_time = None

        # Load image
        image, spacing, position, patient_info = None, None, None, None
        _logger.info(f'Loading image from {image_path} .')
        image_load_time_start = time.perf_counter()
        if distributor.is_main_process():
            image, spacing, position, patient_info = self._read_image(
                image_path,
                n_workers=n_workers,
                progress_bar=True,
                progress_desc='Reading image',
                dicom_name_regex=opt.dicom_name_regex,
            )
        image_load_time = None
        if distributor.is_main_process():
            image_load_time = time.perf_counter() - image_load_time_start
            _logger.info(f'Image loading time: {image_load_time:.2f} seconds.')
            self._time_summary['Loading image'] = image_load_time

            image, spacing, position, patient_info = distributor.broadcast_object(
                image, spacing, position, patient_info
            )

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
        ).astype(np.uint8, copy=False)
        if distributor.is_distributed():
            pred_label = xp.concatenate(distributor.all_gather_object(pred_label))
        model_inference_time = None
        if distributor.is_main_process():
            model_inference_time = time.perf_counter() - start_time
            _logger.info(
                f'Model inference time: {model_inference_time:.2f} seconds.')
            self._time_summary['Model Inference'] = model_inference_time
        del model_data
        gc.collect()

        if distributor.is_distributed():
            image = xp.concatenate(distributor.all_gather_object(image))

        mean_hu_calc_time_start = None
        if distributor.is_main_process():
            _logger.info(f'Calculating class mean HU...')
            mean_hu_calc_time_start = time.perf_counter()
        if image_process_device == 'cpu':
            class_mean_hus, class_volumes, class_counts = self._compute_class_mean_hu_cpu(
                image, pred_label, spacing, n_classes=n_classes)
        elif image_process_device == 'cuda':
            image = xp.to_cupy(image)
            pred_label = xp.to_cupy(pred_label)
            class_mean_hus, class_volumes, class_counts = self._compute_class_mean_hu_cuda(
                image, pred_label, spacing, n_classes=n_classes)
            class_mean_hus = xp.to_numpy(class_mean_hus)
            class_counts = xp.to_numpy(class_counts)
            class_volumes = xp.to_numpy(class_volumes)
        else:
            raise ValueError(f'Invalid image_process_device: {image_process_device}')

        # Done using of image
        del image

        mean_hu_calc_time = None
        if distributor.is_main_process():
            mean_hu_calc_time = time.perf_counter() - mean_hu_calc_time_start
            _logger.info(f'Class mean HU calculation time: {mean_hu_calc_time:.2f} seconds.')
            self._time_summary['HU calculation'] = mean_hu_calc_time

        hu_table_save_time = None
        if distributor.is_main_process():
            hu_df = report_generator.generate_hu_table(
                class_mean_hus=class_mean_hus, class_volumes=class_volumes
            )
            hu_table_save_time_start = time.perf_counter()
            hu_df.write_csv(output_dir / 'hu_table.csv')
            hu_table_save_time = time.perf_counter() - hu_table_save_time_start
            _logger.info(f'HU table saving time: {hu_table_save_time:.2f} seconds.')
            self._time_summary['Saving HU table'] = hu_table_save_time
            del hu_df

        labelmap: npt.NDArray[np.uint8] = xp.to_numpy(pred_label).astype(np.uint8, copy=False)
        labelmap_save_time = None
        if distributor.is_main_process():
            labelmap_save_time_start = time.perf_counter()
            metaimage_utils.write(
                output_dir / 'pred_label.mhd',
                labelmap.astype(np.int16, copy=False),
                spacing=spacing,
                position=position,
            )
            labelmap_save_time = time.perf_counter() - labelmap_save_time_start
            _logger.info(f'Labelmap saving time: {labelmap_save_time:.2f} seconds.')
            self._time_summary['Saving labelmap'] = labelmap_save_time


        report_rendering_time_start = None
        if distributor.is_main_process():
            _logger.info(f'Generating report ...')
            report_rendering_time_start = time.perf_counter()
        report_ppt = report_generator.generate_report(
            patient_info=patient_info,
            labelmap=labelmap,
            spacing=spacing,
            class_mean_hus=class_mean_hus,
            device='cuda',
        )
        report_rendering_time = None
        if distributor.is_main_process():
            report_rendering_time = time.perf_counter() - report_rendering_time_start
            _logger.info(f'Report rendering time: {report_rendering_time:.2f} seconds.')
            self._time_summary['Rendering report'] = report_rendering_time

        report_save_dpi = opt.report_dpi
        if report_save_dpi < 10 or report_save_dpi > 1000:
            # ignore
            if distributor.is_main_process():
                _logger.warning(f'Invalid report dpi: {report_save_dpi}. Reset to 200')
            report_save_dpi = 200

        report_saving_time = None
        if distributor.is_main_process():
            report_saving_time_start = time.perf_counter()
            report_ppt.save(
                pdf_save_path=output_dir / 'report.pdf',
                image_save_path=output_dir / 'report.png',
                dpi=report_save_dpi,
            )
            report_saving_time = time.perf_counter() - report_saving_time_start
            _logger.info(f'Report saving time: {report_saving_time:.2f} seconds.')
            self._time_summary['Saving report'] = report_saving_time

        total_time = None
        if distributor.is_main_process():
            total_time = time.perf_counter() - total_time_start
            _logger.info(f'Total time: {total_time:.2f} seconds.')
            self._time_summary['Total'] = total_time

    @staticmethod
    def _read_image(
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
            patient_info = PatientInfoData.from_dict(creation_kwargs)

        if position is None:
            position = np.zeros(len(spacing), dtype=np.float64)
        return image, spacing, position, patient_info

    @staticmethod
    def _compute_class_mean_hu_cpu(
            image: npt.NDArray,
            labelmap: npt.NDArray[np.integer],
            spacing: npt.NDArray[np.float64],
            n_classes: int = 81,
            process_dtype: str = 'float32',
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
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
        volume_results = voxel_results.astype(
            np.float64, copy=False) * spacing.prod() * 1e-3  # mm3 -> cm3
        return hu_results, volume_results, voxel_results,

    def _compute_class_mean_hu_cuda(
            self,
            image: xp.TypeArrayLike,
            labelmap: xp.TypeArrayLike[np.integer],
            spacing: xp.TypeArrayLike[np.float64],
            n_classes: int = 81,
            process_dtype: str = 'float32',
    ):
        distributor = self._distributor
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
            hu_results.append(xp.to(image[mask], dtype=process_dtype).mean().item())
            voxel_results.append(mask.sum().item())
        if distributor.is_distributed():
            hu_results = sum(distributor.all_gather_object(hu_results), [])
            voxel_results = sum(distributor.all_gather_object(voxel_results), [])
        hu_results = xp.to_dst(hu_results, dst=image, dtype=np.float64)
        voxel_results = xp.to_dst(voxel_results, dst=image, dtype=np.int64)
        volume_results = xp.to(voxel_results, dtype='float64') * spacing.prod().item() * 1e-3  # mm3 -> cm3
        return hu_results, volume_results, voxel_results,

    @classmethod
    def _check_path_exists(cls, *path: Path):
        if len(path) == 0:
            return
        if len(path) == 1:
            if not path[0].exists():
                raise FileNotFoundError(f'Path {path[0]} not found.')
            return
        for p in path:
            cls._check_path_exists(p)

if __name__ == '__main__':
    main = Main()
    main.run()