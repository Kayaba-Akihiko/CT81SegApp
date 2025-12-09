#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Tuple, Optional, Union, Dict, Any, Sequence, List
import copy
import logging

import numpy as np
import numpy.typing as npt
import vtk

from xmodules.xutils import vtk_utils

_logger = logging.getLogger(__name__)

class LabelmapRenderer:


    def __init__(
            self,
            labelmap: npt.NDArray[np.uint8],  # (N, H, W)
            spacing: npt.NDArray[np.float64],
            voxel_limit: int = 2048,
            image_size: Tuple[int, int] = (1000, 2000),
            background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            alpha_bit_planes: Optional[bool] = None,
            multi_samples: Optional[int] = None,
            use_depth_peeling: Optional[bool] = None,
            use_depth_peeling_for_volumes: Optional[bool] = None,
            maximum_number_of_peels: Optional[int] = None,
            occlusion_ratio: Optional[float] = None,
    ):

        vtk_labelmap = vtk_utils.np_image_to_vtk(
            labelmap, spacing, name='labelmap')
        factor = self._compute_uniform_scale(vtk_labelmap, voxel_limit)
        vtk_labelmap = vtk_utils.resample(
            vtk_labelmap, factor, method='nearest')
        self._vtk_labelmap = vtk_labelmap

        self._renderer, self._window = vtk_utils.new_renderer_window(
            window_size=image_size,
            background=background,
            alpha_bit_planes=alpha_bit_planes,
            multi_samples=multi_samples,
            use_depth_peeling=use_depth_peeling,
            use_depth_peeling_for_volumes=use_depth_peeling_for_volumes,
            maximum_number_of_peels=maximum_number_of_peels,
            occlusion_ratio=occlusion_ratio,
        )

    @staticmethod
    def _compute_uniform_scale(
            image: vtk.vtkImageData, limit: float) -> float:
        x0, x1, y0, y1, z0, z1 = image.GetExtent()
        dims = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
        return min(1.0, limit / dims[0], limit / dims[1], limit / dims[2])

    @staticmethod
    def _ensure_extent_min_thickness(
            extent: Tuple[int, int, int, int, int, int],
            min_thick: int = 3
    ) -> Tuple[int, int, int, int, int, int]:
        x0, x1, y0, y1, z0, z1 = extent
        if (z1 - z0 + 1) < min_thick:
            mid = (z0 + z1) // 2
            half = max(1, min_thick // 2)
            z0 = mid - half
            z1 = z0 + min_thick - 1

        return x0, x1, y0, y1, z0, z1

    def render(
            self,
            image: Union[vtk.vtkImageData, vtk.vtkAlgorithmOutput],
            view: Union[vtk_utils.TypeView, List[vtk_utils.TypeView]],
            color_table: Dict[int, Tuple[float, float, float, float]],
            voi: Optional[Tuple[int, int, int, int, int, int]] = None,
            camera_offset: Union[float, List[float]] = 2500.0,
            camera_zoom: Union[float, List[float]] = 1.6,
            out_size: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
    ):


        voi = self._ensure_extent_min_thickness(voi, 3)
        if voi:
            image = vtk_utils.clip(
                image=image, voi=voi, return_port=False)

        color = vtk.vtkColorTransferFunction()
        scalar_opacity = vtk.vtkPiecewiseFunction()
        for class_id, (r, g, b, a) in color_table.items():
            color.AddRGBPoint(class_id, r, g, b)
            scalar_opacity.AddPoint(class_id, a)

        if isinstance(image, vtk.vtkAlgorithmOutput):
            image = image.GetProducer().GetOutput()
        elif isinstance(image, vtk.vtkImageData):
            pass
        else:
            raise ValueError(f"Unknown image type: {type(image)}")

        volume = vtk_utils.build_volume(
            image=image,
            color=color,
            scalar_opacity=scalar_opacity,
        )

        ren, win = self._renderer, self._window
        ren.RemoveAllViewProps()
        ren.AddVolume(volume)

