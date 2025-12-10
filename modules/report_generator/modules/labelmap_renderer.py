#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Tuple, Optional, Union, Dict, Any, Sequence, List
import logging

import numpy as np
import numpy.typing as npt
import vtk
import scipy.ndimage as ndi

from xmodules.xutils import vtk_utils

_logger = logging.getLogger(__name__)

class LabelmapRenderer:


    def __init__(
            self,
            labelmap: npt.NDArray[np.integer],  # (N, H, W)
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
            view: Union[vtk_utils.TypeView, List[vtk_utils.TypeView]],
            color_table: Dict[int, Tuple[float, float, float, float]],
            voi: Optional[Tuple[int, int, int, int, int, int]] = None,
            camera_offset: Union[float, List[float]] = 2500.0,
            out_size: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
            device='cpu'
    ):

        image = self._vtk_labelmap
        if voi is not None:
            voi = self._ensure_extent_min_thickness(voi, 3)
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
            device=device,
        )

        ren, win = self._renderer, self._window
        ren.Clear()
        ren.RemoveAllViewProps()
        ren.AddVolume(volume)
        res = vtk_utils.render_view_as_np_image(
            ren=ren,
            win=win,
            view=view,
            view_camera_position=image.GetCenter(),
            view_camera_offset=camera_offset,
            out_size=out_size,
        )
        ren.Clear()
        ren.RemoveAllViewProps()
        return res

    @staticmethod
    def _longest_true_run(bounds: np.ndarray):
        x = bounds.astype(int)
        diff = np.diff(np.r_[0, x, 0])
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        if len(starts) == 0:
            return None
        lengths = ends - starts + 1
        idx = np.argmax(lengths)
        return starts[idx], ends[idx]

    @classmethod
    def _compute_labelmap_z_extent(
            cls,
            labelmap: vtk.vtkImageData,
            class_id: Union[int, Sequence[int]],
            min_area: int = 200,
            min_voxels: int = 500,
    ) -> Tuple[int, int]:
        np_array, _ = vtk_utils.vtk_image_to_np(labelmap)
        if np_array.ndim != 3:
            raise ValueError("Labelmap must be 3D.")

        (x0, x1, y0, y1, z0, z1) = labelmap.GetExtent()
        nz = np_array.shape[0]

        if isinstance(class_id, int):
            mask = np_array == class_id
        elif isinstance(class_id, Sequence):
            mask = np.isin(np_array, class_id)
        else:
            raise ValueError(f"Invalid class_id: {class_id}")
        total_voxels = int(mask.sum())

        # If label is too small overall → no confidence, return full range
        if total_voxels < min_voxels:
            return z0, z1

        # 1. Slice area: count label pixels per Z
        slice_areas = mask.reshape(nz, -1).sum(axis=1)
        good = slice_areas >= min_area

        longest = cls._longest_true_run(good)
        if longest is not None:
            k0, k1 = longest
            return int(k0 + z0), int(k1 + z0)

        # ----------------------------------------------------------------------
        # 2. Fallback: downsample + connected components (robust to noise)
        # ----------------------------------------------------------------------
        s = 4
        down = mask[::s, ::s, ::s]
        lbl, n = ndi.label(down)

        if n == 0:
            return z0, z1

        # Find largest connected component in downsampled mask
        sizes = ndi.sum(down, lbl, index=range(1, n + 1))
        best_label = 1 + int(np.argmax(sizes))

        zz = np.where(lbl == best_label)[0]
        if zz.size == 0:
            return z0, z1

        # Map back to original Z coordinates
        zmin = int(zz.min()) * s + z0
        zmax = int(zz.max()) * s + z0

        # Clamp to valid original bounds
        zmin = max(z0, min(z1, zmin))
        zmax = max(z0, min(z1, zmax))

        if zmax < zmin:
            zmin, zmax = zmax, zmin

        return zmin, zmax
