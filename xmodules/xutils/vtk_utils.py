#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Tuple, Optional, Union, Dict, Any, TypeAlias, Literal
import io

import numpy as np
import numpy.typing as npt
import vtk
from vtkmodules.util import numpy_support as vtknp
import imageio.v3 as iio

NUMPY_VTK_DTYPE_MAP = {
    np.uint8: vtk.VTK_UNSIGNED_CHAR,
    np.uint16: vtk.VTK_UNSIGNED_SHORT,
    np.int16: vtk.VTK_SHORT,
    np.int32: vtk.VTK_INT,
    np.int64: vtk.VTK_LONG,
    np.float32: vtk.VTK_FLOAT,
    np.float64: vtk.VTK_DOUBLE,
}

TypeView: TypeAlias = Literal['front', 'back', 'left', 'right', 'top', 'bottom']

class VTKUtils:

    @staticmethod
    def new_renderer_window(
            window_size: Tuple[int, int],
            background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            alpha_bit_planes: Optional[bool] = None,
            multi_samples: Optional[int] = None,
            use_depth_peeling: Optional[bool] = None,
            use_depth_peeling_for_volumes: Optional[bool] = None,
            maximum_number_of_peels: Optional[int] = None,
            occlusion_ratio: Optional[float] = None,
    ):
        ren = vtk.vtkRenderer()
        ren.SetBackground(*background)
        renWin = vtk.vtkRenderWindow()
        renWin.OffScreenRenderingOn()
        renWin.SetSize(*window_size)
        renWin.AddRenderer(ren)
        if alpha_bit_planes is not None and alpha_bit_planes:
            renWin.AlphaBitPlanesOn()
        if multi_samples is not None:
            renWin.SetMultiSamples(multi_samples)
        if use_depth_peeling is not None and use_depth_peeling:
            ren.UseDepthPeelingOn()
        if use_depth_peeling_for_volumes is not None and use_depth_peeling_for_volumes:
            ren.UseDepthPeelingForVolumesOn()
        if maximum_number_of_peels is not None:
            ren.SetMaximumNumberOfPeels(maximum_number_of_peels)
        if occlusion_ratio is not None:
            ren.SetOcclusionRatio(occlusion_ratio)
        return ren, renWin

    @staticmethod
    def build_volume(
            image: vtk.vtkImageData,
            color_table: Dict[str, Tuple[int, float, float, float]],
            interpolation: str = 'linear',
            shade=None,
            specular=None,
            specular_power=None,
            ambient=None,
            diffuse=None,
            scalar_opacity_unit_distance=None,
            device='cpu'
    ):
        # The color transfer function maps voxel intensities to colors.
        color = vtk.vtkColorTransferFunction()
        scalar_opacity = vtk.vtkPiecewiseFunction()

        for class_name, (class_id, r, g, b, a) in color_table.items():
            color.AddRGBPoint(class_id, r, g, b)
            scalar_opacity.AddPoint(class_id, a)

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color)
        prop.SetScalarOpacity(scalar_opacity)
        if interpolation == 'linear':
            prop.SetInterpolationTypeToLinear()
        elif interpolation == 'nearest':
            prop.SetInterpolationTypeToNearest()
        else:
            raise ValueError(f'Invalid interpolation method: {interpolation}')
        if shade is not None:
            prop.ShadeOn() if shade else prop.ShadeOff()
        if specular is not None:
            prop.SetSpecular(specular)
        if specular_power is not None:
            prop.SetSpecularPower(specular_power)
        if ambient is not None:
            prop.SetAmbient(ambient)
        if diffuse is not None:
            prop.SetDiffuse(diffuse)
        if scalar_opacity_unit_distance is not None:
            prop.SetScalarOpacityUnitDistance(scalar_opacity_unit_distance)

        if device == 'cpu':
            mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        elif device == 'cuda':
            mapper = vtk.vtkGPUVolumeRayCastMapper()
        else:
            raise ValueError(f'Device {device} is not supported.')
        mapper.SetInputData(image)
        mapper.SetBlendModeToComposite()
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)
        return volume

    @staticmethod
    def position_camera(
            ren: vtk.vtkRenderer,
            view: TypeView,
            center: Tuple[float, float, float],
            cam_offset: Union[float, int] = 0,
    ):
        cam = ren.GetActiveCamera()

        # focal point always looks at the volume center
        cam.SetFocalPoint(*center)
        cx, cy, cz = center
        if view == 'front':  # looking from -Y toward +Y
            cam.SetPosition(cx, cy - cam_offset, cz)
            cam.SetViewUp(0, 0, 1)
        elif view == 'back':  # looking from +Y toward -Y
            cam.SetPosition(cx, cy + cam_offset, cz)
            cam.SetViewUp(0, 0, 1)
        elif view == 'left':  # looking from -X toward +X
            cam.SetPosition(cx - cam_offset, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif view == 'right':  # looking from +X toward -X
            cam.SetPosition(cx + cam_offset, cy, cz)
            cam.SetViewUp(0, 0, 1)
        elif view == 'top':  # looking from +Z downward
            cam.SetPosition(cx, cy, cz + cam_offset)
            cam.SetViewUp(0, 1, 0)  # redefine up for top view
        elif view == 'bottom':  # looking from -Z upward
            cam.SetPosition(cx, cy, cz - cam_offset)
            cam.SetViewUp(0, 1, 0)
        else:
            raise ValueError(f"Unknown view '{view}'")

        cam.Modified()

    @staticmethod
    def win_to_image(
            win: vtk.vtkRenderWindow
    ):
        # Capture window → vtkImageData
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(win)
        w2if.ReadFrontBufferOff()
        w2if.Update()

        # Encode PNG in memory
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.WriteToMemoryOn()
        writer.Write()

        # Extract bytes from VTK's internal array
        vtk_data = writer.GetResult()  # vtkUnsignedCharArray
        png_bytes = bytes(memoryview(vtk_data))

        # Convert to BytesIO object
        image_stream = io.BytesIO(png_bytes)
        image_stream.seek(0)

        return iio.imread(image_stream)

    @staticmethod
    def resample(
            image: vtk.vtkImageData,
            factor: Union[float, Tuple[float, float, float]],
            method: str = 'linear',
    ) -> vtk.vtkImageData:
        resample = vtk.vtkImageResample()
        resample.SetInputData(image)
        if isinstance(factor, (float, int)):
            factor = (factor, factor, factor)
        fx, fy, fz = factor
        resample.SetAxisMagnificationFactor(0, fx)  # X
        resample.SetAxisMagnificationFactor(1, fy)  # Y
        resample.SetAxisMagnificationFactor(2, fz)  # Z

        if method == 'linear':
            resample.SetInterpolationModeToLinear()
        elif method == 'nearest':
            resample.SetInterpolationModeToNearestNeighbor()
        else:
            raise ValueError(f'Invalid interpolation method: {method}')
        resample.Update()
        return resample.GetOutput()

    @staticmethod
    def np_image_to_vtk(
            image: npt.NDArray,
            spacing: Optional[npt.NDArray[np.float64]] = None,
            name: Optional[str] = None
    ) -> vtk.vtkImageData:
        if image.ndim != 3:
            raise ValueError(f'Invalid image shape: {image.shape}')
        N, H, W = image.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(W, H, N)  # VTK expects (x, y, z) = (W, H, N)
        if spacing is not None:
            if spacing.ndim != 1 or len(spacing) != 3:
                raise ValueError(f'Invalid spacing {spacing}')
            vtk_image.SetSpacing(float(spacing[2]), float(spacing[1]), float(spacing[0]))
        image = np.ascontiguousarray(image)  # shape (N, H, W), C-order
        vtk_array = vtknp.numpy_to_vtk(
            num_array=image.ravel(order="C"),
            deep=True,
            array_type=NUMPY_VTK_DTYPE_MAP[image.dtype],
        )
        if name is not None:
            vtk_array.SetName(name)
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image



new_renderer_window = VTKUtils.new_renderer_window
np_image_to_vtk = VTKUtils.np_image_to_vtk
resample = VTKUtils.resample
build_volume = VTKUtils.build_volume