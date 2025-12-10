#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Tuple, Optional, Union, Dict, Any, TypeAlias, Literal, Type, TypeVar, Protocol, List, Sequence
from pathlib import Path
from collections import OrderedDict
import json
import copy

import numpy as np
import numpy.typing as npt
import polars as pl

from .modules.report_ppt import ReportPPT, PPTXPresentation
from .modules.labelmap_renderer import LabelmapRenderer
from .modules.utils import figure_utils



class ReportGenerator:

    def __init__(
            self,
            template_ppt: Union[PPTXPresentation, Path],
            hu_statistics_table: Union[pl.DataFrame, Path],
            rendering_config: Union[Dict[str, Any], Path],

    ):
        self._report_ppt = ReportPPT(template_ppt)
        if isinstance(hu_statistics_table, Path):
            if hu_statistics_table.suffix == '.csv':
                self._hu_statistics_df = pl.read_csv(hu_statistics_table)
            elif hu_statistics_table.suffix == '.xlsx' or hu_statistics_table.suffix == '.xls':
                self._hu_statistics_df = pl.read_excel(hu_statistics_table)
            else:
                raise ValueError(f'Unsupported file type: {hu_statistics_table.suffix}')
        else:
            self._hu_statistics_df = hu_statistics_table.clone()

        if isinstance(rendering_config, Path):
            with rendering_config.open('rb') as f:
                rendering_config = json.load(f)
        elif isinstance(rendering_config, dict):
            rendering_config = copy.deepcopy(rendering_config)
        else:
            raise ValueError(f'Invalid rendering config: {rendering_config}')
        self._rendering_config = rendering_config

        self._report_pptx_canvas_px = self._report_ppt.collect_shape_sizes(dpi=96)

        color_table = {}
        for class_name, (class_id, r, g, b, a) in self._rendering_config['color'].items():
            color_table[class_id] = (r, g, b, a)
        self._class_color_table = color_table

        self._n_groups = 10

        img_keys = [f'{i:02d}' for i in range(1, self._n_groups + 1)]
        canvas_heights = [self._report_pptx_canvas_px[k][1] for k in img_keys if k in img_keys]
        min_canvas_h = min(canvas_heights) if canvas_heights else 400
        self._min_canvas_h = min_canvas_h


    def generate(
            self,
            labelmap: npt.NDArray[np.integer],
            spacing: npt.NDArray[np.float64],
            class_mean_hus: npt.NDArray[np.float64],
            sex: Literal['male', 'female'],
            age: int,
            groups_classes: OrderedDict[str, Sequence[int]],
            fig_dpi=96
    ):
        if labelmap.ndim != 3:
            raise ValueError('Labelmap must be 3D')
        if spacing.ndim != 1:
            raise ValueError('Spacing must be 1D')
        if len(spacing) != labelmap.ndim:
            raise ValueError('Spacing must have the same length as labelmap')

        if sex not in {'male', 'female'}:
            raise ValueError(f'Invalid sex: {sex}')

        if age < 0:
            raise ValueError(f'Invalid age: {age}')

        if len(groups_classes) != 10:
            raise ValueError(
                f'Expected 10 groups in groups_classes, got {len(groups_classes)}'
            )

        age_sex_hu_df = self._hu_statistics_df.filter(
            (pl.col('sex') == sex)
            & (pl.col('age_group_low') <= age)
            & (pl.col('age_group_high') >= age)
        )

        max_rows = max(len(v) for v in groups_classes.values())
        row_gap_px = 8
        common_box_height_px = max(10, int((self._min_canvas_h - max_rows * row_gap_px) / max_rows))
        common_box_height_px = min(common_box_height_px, 20)
        s_star_common = figure_utils.star_size_points2_from_box_height(
            common_box_height_px, dpi=fig_dpi, scale=1.5
        )

        labelmap_renderer = LabelmapRenderer(
            labelmap=labelmap, spacing=spacing,
            voxel_limit=2048,
            image_size=(1500, 2000),
            alpha_bit_planes=self._rendering_config.get('alpha_bit_planes', None),
            multi_samples=self._rendering_config.get('multi_samples', None),
            use_depth_peeling=self._rendering_config.get('use_depth_peeling', None),
            use_depth_peeling_for_volumes=self._rendering_config.get('use_depth_peeling_for_volumes', None),
            maximum_number_of_peels=self._rendering_config.get('maximum_number_of_peels', None),
            occlusion_ratio=self._rendering_config.get('occlusion_ratio', None),
        )

        ppt_image_dict = {}
        # '1' to '10'
        for idx, group_name in enumerate(groups_classes.keys(), start=1):
            ppt_image_key = f'{idx:02d}'

            box_drawing_data = []
            group_classes = groups_classes[group_name]
            for class_id in group_classes:
                class_df = age_sex_hu_df.filter(pl.col('class_id') == class_id)
                if len(class_df) != 1:
                    raise ValueError(
                        f'Unexpected number of rows {sex=} {age=} {class_id=}: {len(class_df)}'
                    )
                row = class_df.row(0)
                mean, std = float(row['mean']), float(row['std'])
                del row, class_df

                class_mean_hu = float(class_mean_hus[class_id])
                box_drawing_data.append(figure_utils.BoxDrawingData(
                    target=class_mean_hu,
                    mean=mean, std=std,
                    color=self._class_color_table[class_id][:3],
                ))
            box_figure = figure_utils.draw_hu_boxes(
                boxes=box_drawing_data,
                canvas_size_px=self._report_pptx_canvas_px[ppt_image_key],
                s_star=s_star_common,
                box_height_px=common_box_height_px,
                row_gap_px=row_gap_px,
                dpi=fig_dpi,
            )
            ppt_image_dict[ppt_image_key] = box_figure
            del box_drawing_data, box_figure

        front_view, back_view = labelmap_renderer.render(
            view=['front', 'back'],
            color_table=self._class_color_table,
            camera_offset=2500.0 / 2.3,
        )

        # '11' '12'
        ppt_image_dict['11'] = front_view
        ppt_image_dict['12'] = back_view