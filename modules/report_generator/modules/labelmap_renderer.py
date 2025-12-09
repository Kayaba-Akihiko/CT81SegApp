#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Tuple, Optional, Union, Dict, Any
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
            camera_offset: float = 2500.0,
            tight_crop: bool = False,
            voi_zoom: float = 1.6,
            keep_aspect_for_ppt: bool = True
    ):

        vtk_labelmap = vtk_utils.np_image_to_vtk(
            labelmap, spacing, name='labelmap')
        factor = self._compute_uniform_scale(vtk_labelmap, voxel_limit)
        vtk_labelmap = vtk_utils.resample(
            vtk_labelmap, factor, method='nearest')

    @staticmethod
    def _compute_uniform_scale(
            image: vtk.vtkImageData, limit: float) -> float:
        x0, x1, y0, y1, z0, z1 = image.GetExtent()
        dims = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
        return min(1.0, limit / dims[0], limit / dims[1], limit / dims[2])
