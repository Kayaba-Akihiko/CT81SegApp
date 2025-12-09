#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

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
        '-m', '--model_path', type=str, default=None)
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
        '--device',
        type=str,
        default='cuda' if xp.is_cuda_available() else 'cpu',
    )
    parser.add_argument(
        '--dist_backend',
        type=str,
        default='fabric' if HAS_FABRIC else 'none',
    )
    parser.add_argument(
        '--dist_accelerator',
        type=str,
        default='cpu',
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
        assert isinstance(output_dir, Path)
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

    if xp.HAS_CUPY and xp.CUPY_CUDA_AVAILABLE:
        import cupy as cp
        cp.cuda.runtime.setDevice(distributor.local_rank)
    if xp.HAS_TORCH and xp.TORCH_CUDA_AVAILABLE:
        import torch
        torch.cuda.set_device(distributor.local_rank)

    device = opt.device

    total_start_time = None
    if distributor.is_main_process():
        total_start_time = time.perf_counter()
    image_path = os_utils.format_path_string(opt.image_path)
    assert isinstance(image_path, Path)
    model_path = opt.model_path
    if model_path is None:
        model_path = this_dir / 'resources' / 'nnunet_1res_ct_81_seg.onnx'
        _logger.info(f'Model path not specified. Using default: {model_path}.')
    else:
        model_path = os_utils.format_path_string(model_path)
        assert isinstance(model_path, Path)

    norm_config_path = opt.norm_config_path
    if norm_config_path is None:
        norm_config_path = this_dir / 'resources' / 'nnunet_1res_ct_81_seg_norm.json'
        _logger.info(f'Normalization config path not specified. Using default: {norm_config_path}.')
    else:
        norm_config_path = os_utils.format_path_string(norm_config_path)
        assert isinstance(norm_config_path, Path)
    n_classes = int(opt.n_classes)
    resolution = opt.resolution
    batch_size = opt.batch_size
    rendering_config_path = opt.rendering_config_path
    if rendering_config_path is None:
        rendering_config_path = this_dir / 'resources' / 'naist_totalsegmentator_81.json'
        _logger.info(f'Rendering config path not specified. Using default: {rendering_config_path}.')
    else:
        rendering_config_path = os_utils.format_path_string(
            opt.rendering_config_path
        )
    n_workers = min(opt.n_workers, os_utils.get_max_n_worker())

    if device == 'cpu':
        onnx_providers = ['CPUExecutionProvider']
    elif device == 'cuda':
        onnx_providers = [
            ('CUDAExecutionProvider', {"device_id": distributor.local_rank}),
            'CPUExecutionProvider'
        ]
    else:
        raise ValueError(f'Device {distributor.device.type} is not supported.')

    if distributor.is_main_process():
        _logger.info(f'Load model and normalization configuration')
    try:
        start_time = None
        if distributor.is_main_process():
            start_time = time.perf_counter()
        model_data = Inferencer.get_model(
            model_path=model_path,
            norm_config_path=norm_config_path,
            in_shape=(1, resolution, resolution),
            out_shape=(n_classes, resolution, resolution),
            onnx_providers=onnx_providers,
        )
        model_loading_time = None
        if distributor.is_main_process():
            model_loading_time = time.perf_counter() - start_time
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
            image, spacing, position = distributor.broadcast_object(
                image, spacing, position
            )
        assert isinstance(image, np.ndarray)

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

        start_time = None
        if distributor.is_main_process():
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
        model_inference_time = None
        if distributor.is_main_process():
            model_inference_time = time.perf_counter() - start_time
            _logger.info(
                f'Model inference time: {model_inference_time:.2f} seconds.')
        del image

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
            assert isinstance(pred_label, np.ndarray)
            start_time = time.perf_counter()
            metaimage_utils.write(
                labelmap_save_path,
                pred_label, spacing=spacing, position=position
            )
            prediction_saving_time = time.perf_counter() - start_time
            _logger.info(f'Prediction saving time: {prediction_saving_time:.2f} seconds.')

        if distributor.is_main_process():
            rendering_config_loading_time = None
            rendering_time = None
            if rendering_config_path is not None:
                assert isinstance(rendering_config_path, Path)
                _logger.info(f'Load rendering configuration from {rendering_config_path} .')
                start_time = time.perf_counter()
                rend_config = read_rendering_config(
                    read_path=rendering_config_path)
                rendering_config_loading_time = time.perf_counter() - start_time
                _logger.info(f'Rendering configuration loading time: {rendering_config_loading_time:.2f} seconds.')

                _logger.info(f'Render prediction previews')
                start_time = time.perf_counter()
                preview_labelmap(
                    labelmap=pred_label,
                    spacing=spacing,
                    rendering_config=rend_config,
                    output_dir=output_dir,
                    device=device,
                )
                rendering_time = time.perf_counter() - start_time
                _logger.info(f'Rendering and preview saving time: {rendering_time:.2f} seconds.')

            total_time = time.perf_counter() - total_start_time
            _logger.info(f'Total time: {total_time:.2f} seconds ({total_time / 60. :.2f} minutes).')
            time_summary = {
                'model_loading_time': model_loading_time,
                'image_loading_time': image_loading_time,
                'model_inference_time': model_inference_time,
                'prediction_saving_time': prediction_saving_time,
            }
            if rendering_config_loading_time is not None:
                time_summary['rendering_config_loading_time'] = rendering_config_loading_time
            if rendering_time is not None:
                time_summary['rendering_time'] = rendering_time
            time_summary['total_time'] = total_time
            _logger.info(
                f'\n ---- Time summary ----\n'
                f'{json.dumps(time_summary, indent=2)}'
                f'\n ---- End of time summary ----'
            )
    except Exception as e:
        error_message = f'{e}\n{traceback.format_exc()}'
        _logger.error(error_message)
        raise e

    if distributor.is_main_process():
        _logger.info('Inference program end.')


def preview_labelmap(
        labelmap: xp.TypeArrayLike[xp.NPInteger],
        spacing: xp.TypeArrayLike[np.float64],
        rendering_config: dict,
        output_dir: Path,
        device: str = 'cpu',
):
    import vtk
    from vtkmodules.util import numpy_support as vtknp


    colors = vtk.vtkNamedColors()
    colors.SetColor("BkgColor", [255, 255, 255, 255])

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(ren)
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)

    # The gradient opacity function is used to decrease the opacity
    # in the "flat" regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
    # For most medical data, the unit distance is 1mm.
    # gradient_opacity = vtk.vtkPiecewiseFunction()

    # The color transfer function maps voxel intensities to colors.
    color = vtk.vtkColorTransferFunction()
    scalar_opacity = vtk.vtkPiecewiseFunction()

    prop = vtk.vtkVolumeProperty()

    rendering_config = copy.deepcopy(rendering_config)
    if 'color' in rendering_config:
        for class_name, (class_id, r, g, b, a) in rendering_config['color'].items():
            color.AddRGBPoint(class_id, r, g, b)
            scalar_opacity.AddPoint(class_id, a)
        del rendering_config['color']

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    # opacityValue = 0.4
    # scalar_opacity.AddPoint(0, 0, 0.5, 0.0)
    # scalar_opacity.AddPoint(1, opacityValue, 0.5, 0.0)
    # scalar_opacity.AddPoint(len(lut) - 1, opacityValue, 0.5, 0.0)
    # scalar_opacity.AddPoint(39, 0, 0.5, 0.0)
    # scalar_opacity.AddPoint(40, 0, 0.5, 0.0)

    # gradient_opacity.AddPoint(0, 1)
    # gradient_opacity.AddPoint(90, 1)
    # gradient_opacity.AddPoint(180, 1)
    prop.SetInterpolationTypeToNearest()
    if (specular := rendering_config.pop('specular', None)) is not None:
        prop.SetSpecular(specular)
    if (specular_power := rendering_config.pop('specular_power', None)) is not None:
        prop.SetSpecularPower(specular_power)
    if (ambient := rendering_config.pop('ambient', None)) is not None:
        prop.SetAmbient(ambient)
    if (diffuse := rendering_config.pop('diffuse', None)) is not None:
        prop.SetDiffuse(diffuse)
    if (scalar_opacity_unit_distance := rendering_config.pop('scalar_opacity_unit_distance', None)) is not None:
        prop.SetScalarOpacityUnitDistance(scalar_opacity_unit_distance)
    prop.SetColor(color)
    prop.SetScalarOpacity(scalar_opacity)
    # prop.SetGradientOpacity(gradient_opacity)

    prop.ShadeOff()
    if (shade := rendering_config.pop('shade', None)) is not None:
        if isinstance(shade, bool):
            if shade:
                prop.ShadeOn()
        elif isinstance(shade, str):
            shade_l = shade.lower()
            if shade_l in ['t', 'true', 'yes', 'on']:
                prop.ShadeOn()
            elif shade_l in ['f', 'false', 'no', 'off']:
                pass
            else:
                raise ValueError(f'Invalid shade value: {shade}')
        elif isinstance(shade, int):
            if shade == 1:
                prop.ShadeOn()
            elif shade == 0:
                pass
            else:
                raise ValueError(f'Invalid shade value: {shade}')
        else:
            raise ValueError(f'Invalid shade value: {shade}')

    if len(rendering_config) > 0:
        _logger.warning(
            f'Rendering config contains unknown parameter(s): {json.dumps(rendering_config, indent=2)}')

    # Set a background color for the renderer
    ren.SetBackground(colors.GetColor3d("BkgColor"))

    # Increase the size of the render window
    # renWin.SetSize(600, 600)
    renWin.SetSize(600, 1500)

    # renWin.Render()
    renWin.SetOffScreenRendering(True)

    # The following reader is used to read a series of 2D slices (images)
    # that compose the volume. The slice dimensions are set, and the
    # pixel spacing. The data Endianness must also be specified. The reader
    # uses the FilePrefix in combination with the slice number to construct
    # filenames using the format FilePrefix.%d. (In this case the FilePrefix
    # is the root name of the file: quarter.)
    # reader = vtk.vtkMetaImageReader()  # for .mhd or .mha
    # reader = vtk.vtkNIFTIImageReader()  # for .nii or .nii.gz
    # reader = vtk.vtkNrrdReader()  # for .nii or .nii.gz
    # reader.SetFileName(str(label_path))

    # labelmap: np.ndarray with shape (N, H, W)
    # spacing: np.ndarray with shape (3,)  (e.g., [dz, dy, dx])
    N, H, W = labelmap.shape

    # Create vtkImageData
    image = vtk.vtkImageData()
    image.SetDimensions(W, H, N)  # VTK expects (x, y, z) = (W, H, N)
    # Adjust this depending on your spacing order
    # If spacing = (dz, dy, dx):
    image.SetSpacing(float(spacing[2]), float(spacing[1]), float(spacing[0]))
    # If spacing is already (dx, dy, dz), then:
    # image.SetSpacing(*[float(s) for s in spacing])

    # Convert NumPy array to vtkArray
    # Ensure the dtype matches what you want (e.g., uint8, uint16, etc.)
    labelmap_np = np.ascontiguousarray(labelmap)  # shape (N, H, W), C-order
    vtk_array = vtknp.numpy_to_vtk(
        num_array=labelmap_np.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_SHORT,  # or VTK_UNSIGNED_CHAR / VTK_SHORT etc.
    )
    vtk_array.SetName("labelmap")
    image.GetPointData().SetScalars(vtk_array)

    if device == 'cpu':
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    elif device == 'cuda':
        mapper = vtk.vtkGPUVolumeRayCastMapper()
    else:
        raise ValueError(f'Device {device} is not supported.')

    # mapper.SetInputConnection(reader.GetOutputPort())
    mapper.SetInputData(image)
    # mapper.UseDepthPassOn()
    # mapper.SetImageSampleDistance(0.05)
    # mapper.SetAutoAdjustSampleDistances(False)
    # print('vtkGPUVolumeRayCastMapper::GetSampleDistance: {}'.format(mapper.GetSampleDistance()))
    # print('vtkGPUVolumeRayCastMapper::GetUseDepthPass: {}'.format(mapper.GetUseDepthPass()))
    # print('vtkGPUVolumeRayCastMapper::GetScalarModeAsString: {}'.format(mapper.GetScalarModeAsString()))
    mapper.SetBlendModeToComposite()
    # mapper.SetBlendModeToAverageIntensity()

    # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    # and orientation of the volume in world coordinates.
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    # Finally, add the volume to the renderer
    ren.RemoveAllViewProps()
    ren.AddViewProp(volume)

    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the
    # patient's left (which is our right).
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    # print("volume.GetCenter(): ", c)
    camera_distance = 2400  # 700
    # camera_distance = 1200 #700
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(c[0], c[1] + camera_distance, c[2])
    camera.SetClippingRange(100, 5000)
    camera.SetFocalPoint(c[0], c[1], c[2])
    # camera.Azimuth(90.0)
    camera.Elevation(0.0)
    camera.Azimuth(180.0)

    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    writer = vtk.vtkPNGWriter()
    save_path = output_dir / 'preview_front.png'
    _logger.info(f'Saving front view to {save_path}')
    writer.SetFileName(str(save_path))
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

    camera.Azimuth(-90.0)
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    save_path = output_dir / 'preview_right.png'
    _logger.info(f'Saving right view to {save_path}')
    writer.SetFileName(str(save_path))
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

    camera.Azimuth(-90.0)
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    save_path = output_dir / 'preview_back.png'
    _logger.info(f'Saving back view to {save_path}')
    writer.SetFileName(output_dir / 'preview_back.png')
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

    camera.Azimuth(-90.0)
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    save_path = output_dir / 'preview_left.png'
    _logger.info(f'Saving left view to {save_path}')
    writer.SetFileName(str(save_path))
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()


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

def read_rendering_config(read_path: Path):
    with open(read_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    main()