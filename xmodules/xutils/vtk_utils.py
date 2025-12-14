#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Tuple, Optional, Union, Dict, Any, TypeAlias, Literal, Type, TypeVar, Protocol, List
import io

import numpy as np
import numpy.typing as npt
import vtk
from vtkmodules.util import numpy_support as vtknp
import imageio.v3 as iio

TypeView: TypeAlias = Literal['front', 'back', 'left', 'right', 'top', 'bottom']


NUMPY_VTK_DTYPE_MAP = {
    np.uint8: vtk.VTK_UNSIGNED_CHAR,
    np.uint16: vtk.VTK_UNSIGNED_SHORT,
    np.int16: vtk.VTK_SHORT,
    np.int32: vtk.VTK_INT,
    np.int64: vtk.VTK_LONG,
    np.float32: vtk.VTK_FLOAT,
    np.float64: vtk.VTK_DOUBLE,
    np.dtype(np.uint8): vtk.VTK_UNSIGNED_CHAR,
    np.dtype(np.uint16): vtk.VTK_UNSIGNED_SHORT,
    np.dtype(np.int16): vtk.VTK_SHORT,
    np.dtype(np.int32): vtk.VTK_INT,
    np.dtype(np.int64): vtk.VTK_LONG,
    np.dtype(np.float32): vtk.VTK_FLOAT,
    np.dtype(np.float64): vtk.VTK_DOUBLE,
}

class SupportsImageInput(Protocol):
    def SetInputData(self, data: vtk.vtkImageData) -> None: ...
    def SetInputConnection(self, out: vtk.vtkAlgorithmOutput) -> None: ...


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
    ) -> Tuple[vtk.vtkRenderer, vtk.vtkRenderWindow]:
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

    @classmethod
    def render_view_as_np_image(
            cls,
            ren: vtk.vtkRenderer,
            win: vtk.vtkRenderWindow,
            view: Union[TypeView, List[TypeView]],
            view_camera_position: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
            view_camera_offset: Union[float, List[float]] = 2500.0,
            out_size: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
    ) -> Union[npt.NDArray[np.uint8], List[npt.NDArray[np.uint8]]]:
        if isinstance(view, str):
            view = [view]
        elif isinstance(view, (list, tuple)):
            pass
        else:
            raise ValueError(f"Unknown view type: {type(view)}")

        if (n_views :=len(view)) == 0:
            raise ValueError("Empty view list.")

        if not isinstance(view_camera_position, (list, tuple)):
            raise ValueError(f"Invalid view_camera_center: {view_camera_position}")
        if len(view_camera_position) == 0:
            raise ValueError("Empty view_camera_center list.")
        if not isinstance(view_camera_position[0], (list, tuple)):
            view_camera_position = [view_camera_position] * n_views
        if len(view_camera_position) != n_views:
            raise ValueError(f"Invalid view_camera_center: {view_camera_position}")

        if not isinstance(view_camera_offset, (list, tuple)):
            view_camera_offset = [view_camera_offset] * n_views
        if len(view_camera_offset) != n_views:
            raise ValueError(f"Invalid view_camera_offset: {view_camera_offset}")


        if out_size is not None:
            prev_size = win.GetSize()
            if isinstance(out_size, (list, tuple)):
                if len(out_size) == 0:
                    raise ValueError("Empty out_size list.")
                if isinstance(out_size[0], (list, tuple)):
                    if len(out_size) != len(view):
                        raise ValueError(f"Invalid out_size list: {out_size}")
                else:
                    out_size = [out_size] * len(view)
        else:
            prev_size = None
            out_size = [None] * len(view)

        results = []
        for view_i, camera_center_i, camera_offset_i, out_size_i in zip(
                view, view_camera_position, view_camera_offset, out_size):
            view_i: TypeView
            camera_center_i: Tuple[float, float, float]
            if out_size_i is not None:
                win.SetSize(*out_size_i)
            cls.position_view_camera(
                cam=ren.GetActiveCamera(), view=view_i,
                position=camera_center_i, cam_offset=camera_offset_i)
            ren.ResetCameraClippingRange()
            win.Render()
            rendered_image = cls.capture_window_as_numpy(win)
            results.append(rendered_image)
        if prev_size is not None:
             win.SetSize(*prev_size)
        if len(results) == 0:
            raise RuntimeError("No rendered image.")
        if len(results) == 1:
            return results[0]
        return results

    @classmethod
    def build_volume(
            cls,
            image: Union[vtk.vtkImageData, vtk.vtkAlgorithmOutput],
            color: Optional[vtk.vtkColorTransferFunction] = None,
            scalar_opacity: Optional[vtk.vtkPiecewiseFunction] = None,
            interpolation=None,
            shade=None,
            specular=None,
            specular_power=None,
            ambient=None,
            diffuse=None,
            scalar_opacity_unit_distance=None,
            blend_mode=None,
            device='cpu'
    ) -> vtk.vtkVolume:
        prop = vtk.vtkVolumeProperty()
        if color is not None:
            prop.SetColor(color)
        if scalar_opacity is not None:
            prop.SetScalarOpacity(scalar_opacity)
        if interpolation is not None:
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

        mapper: vtk.vtkVolumeMapper = vtk.vtkSmartVolumeMapper()
        if device == 'cpu':
            mapper.SetRequestedRenderModeToRayCast()
        elif device == 'cuda':
            mapper.SetRequestedRenderModeToGPU()
        else:
            raise ValueError(f'Device {device} is not supported.')
        if blend_mode is not None:
            if blend_mode == 'composite':
                mapper.SetBlendModeToComposite()
            elif blend_mode == 'average_intensity':
                mapper.SetBlendModeToAverageIntensity()
            else:
                raise ValueError(f'Invalid blend mode: {blend_mode}')
        if interpolation is not None:
            if interpolation == 'linear':
                mapper.SetInterpolationModeToLinear()
            elif interpolation == 'nearest':
                mapper.SetInterpolationModeToNearestNeighbor()
            else:
                raise ValueError(f'Invalid interpolation method: {interpolation}')
        cls.set_input(mapper, image)

        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)
        return volume

    @staticmethod
    def position_view_camera(
            cam: vtk.vtkCamera,
            view: TypeView,
            position: Tuple[float, float, float],
            cam_offset: Union[float, int] = 0,
    ) -> vtk.vtkCamera:

        # focal point always looks at the volume center
        cam.SetFocalPoint(*position)
        cx, cy, cz = position
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
        return cam

    @staticmethod
    def capture_window_as_numpy(
            win: vtk.vtkRenderWindow
    ) -> npt.NDArray[np.uint8]:
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

    @classmethod
    def resample(
            cls,
            image: Union[vtk.vtkImageData, vtk.vtkAlgorithmOutput],
            factor: Union[float, Tuple[float, float, float]],
            method: str = 'linear',
            return_port: bool = False,
    ) -> Union[vtk.vtkAlgorithmOutput, vtk.vtkImageData]:
        resample = vtk.vtkImageResample()
        cls.set_input(resample, image, )
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
        if return_port:
            return resample.GetOutputPort()
        return resample.GetOutput()

    @classmethod
    def clip(
            cls,
            image: Union[vtk.vtkImageData, vtk.vtkAlgorithmOutput],
            extent: Tuple[int, int, int, int, int, int],
            return_port: bool = False
    ) -> Union[vtk.vtkAlgorithmOutput, vtk.vtkImageData]:
        image_clip = vtk.vtkImageClip()
        cls.set_input(image_clip, image,)
        image_clip.SetOutputWholeExtent(extent)
        image_clip.ClipDataOn()
        image_clip.Update()
        if return_port:
            return image_clip.GetOutputPort()
        return image_clip.GetOutput()

    @classmethod
    def build_labelmap_color(
            cls,
            color_table: Dict[
                int,
                Tuple[
                    Union[int, float],
                    Union[int, float],
                    Union[int, float],
                    Union[int, float],
                ]
            ]
    ) -> Tuple[vtk.vtkColorTransferFunction, vtk.vtkPiecewiseFunction]:
        def _convert_value(_val: Union[int, float]) -> float:
            if isinstance(_val, int):
                if _val < 0 or _val > 255:
                    raise ValueError(f'Invalid color value: {_val}')
                return _val / 255.0
            elif isinstance(_val, float):
                if _val < 0 or _val > 1.0:
                    raise ValueError(f'Invalid color value: {_val}')
                return _val
            else:
                raise ValueError(f'Invalid color value: {_val}')

        vtk_color = vtk.vtkColorTransferFunction()
        vtk_scalar_opacity = vtk.vtkPiecewiseFunction()
        half = 0.5
        for class_id, color in color_table.items():
            r, g, b, a = map(_convert_value, color)
            vtk_color.AddRGBPoint(class_id, r, g, b)
            vtk_scalar_opacity.AddPoint(class_id - half, 0.0)
            vtk_scalar_opacity.AddPoint(class_id, a)
            vtk_scalar_opacity.AddPoint(class_id + half, 0.0)
            # vtk_scalar_opacity.AddPoint(class_id, a)
        return vtk_color, vtk_scalar_opacity

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

    @staticmethod
    def vtk_image_to_np(
            image: vtk.vtkImageData
    ) -> Tuple[npt.NDArray, npt.NDArray[np.float64]]:
        dims = image.GetDimensions()  # (X, Y, Z)
        n_components = image.GetNumberOfScalarComponents()

        vtk_array = image.GetPointData().GetScalars()
        np_array = vtknp.vtk_to_numpy(vtk_array)

        # Reshape to (Z, Y, X, C)
        if n_components > 1:
            np_array = np_array.reshape(dims[2], dims[1], dims[0], n_components)
        else:
            np_array = np_array.reshape(dims[2], dims[1], dims[0])

        spacing = image.GetSpacing()
        spacing = np.array([spacing[2], spacing[1], spacing[0]], dtype=np.float64)
        return np_array, spacing

    @staticmethod
    def set_input(
            algorithm: SupportsImageInput,
            input: Union[vtk.vtkImageData, vtk.vtkAlgorithmOutput]
    ) -> SupportsImageInput:
        if isinstance(input, vtk.vtkImageData):
            algorithm.SetInputData(input)
        elif isinstance(input, vtk.vtkAlgorithmOutput):
            algorithm.SetInputConnection(input)
        else:
            raise TypeError(f'Invalid input type: {type(input)}')
        return algorithm


new_renderer_window = VTKUtils.new_renderer_window
build_volume = VTKUtils.build_volume
render_view_as_np_image = VTKUtils.render_view_as_np_image
position_view_camera = VTKUtils.position_view_camera
capture_window_as_numpy = VTKUtils.capture_window_as_numpy
resample = VTKUtils.resample
clip = VTKUtils.clip
build_labelmap_color = VTKUtils.build_labelmap_color
np_image_to_vtk = VTKUtils.np_image_to_vtk
vtk_image_to_np = VTKUtils.vtk_image_to_np

