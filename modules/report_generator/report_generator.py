#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Tuple, Optional, Union, Dict, Any, TypeAlias, Literal, Type, TypeVar, Protocol, List, Sequence
from pathlib import Path
from collections import OrderedDict, defaultdict
import json
import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from xmodules.xutils import vtk_utils

from .modules.report_ppt import ReportPPT, PPTXPresentation, FillImageData as ReportFillImageData, TypeFitMode as TypeReportFitMode
from .modules.labelmap_renderer import LabelmapRenderer
from .modules.utils import figure_utils


@dataclass
class ClassGroupData:
    name: str
    class_ids:List[int]
    rendering_view: vtk_utils.TypeView

class ReportGenerator:

    def __init__(
            self,
            template_ppt: Union[PPTXPresentation, Path],
            hu_statistics_table: Union[pl.DataFrame, Path],
            rendering_config: Union[Dict[str, Any], Path],
            class_info_table: Union[pl.DataFrame, Path],
            class_groups: Union[
                List[Union[ClassGroupData, Dict[str, Any]]],
                Tuple[Union[ClassGroupData, Dict[str, Any]], ...],
                Path,
            ]
    ):
        self._report_ppt = ReportPPT(template_ppt)
        if isinstance(hu_statistics_table, Path):
            self._hu_statistics_df = self._read_dataframe(hu_statistics_table)
        else:
            if not isinstance(hu_statistics_table, pl.DataFrame):
                raise ValueError(f'Invalid hu statistics table: {hu_statistics_table}')
            self._hu_statistics_df = hu_statistics_table.clone()

        if isinstance(rendering_config, Path):
            with rendering_config.open('rb') as f:
                rendering_config = json.load(f)
        elif isinstance(rendering_config, dict):
            rendering_config = copy.deepcopy(rendering_config)
        else:
            raise ValueError(f'Invalid rendering config: {rendering_config}')
        self._skin_class_id = rendering_config.pop('skin_class_id')
        self._rendering_config = rendering_config

        if isinstance(class_info_table, Path):
            self._class_info_df = self._read_dataframe(class_info_table)
        else:
            if not isinstance(class_info_table, pl.DataFrame):
                raise ValueError(f'Invalid class info table: {class_info_table}')
            self._class_info_df = class_info_table.clone()

        color_table = OrderedDict()
        class_name_to_id_map = {}
        class_id_to_name_map = {}
        for row in self._class_info_df.iter_rows(named=True):
            class_id, class_name = row['class_id'], row['class_name']
            class_id = int(class_id)
            r, g, b, a = row['r'], row['g'], row['b'], row['a']
            r, g, b, a = float(r), float(g), float(b), float(a)
            class_name_to_id_map[class_name] = class_id
            class_id_to_name_map[class_id] = class_name
            color_table[class_id] = (r, g, b, a)
        self._class_name_to_id_map = class_name_to_id_map
        self._class_id_to_name_map = class_id_to_name_map
        self._class_color_table = color_table

        def _create_group_data_from_dict(_source_dict: Dict[str, Any]):
            if 'class_ids' in _source_dict:
                class_ids = _source_dict['class_ids']
            elif 'class_names' in _source_dict:
                class_ids = [
                    self._class_name_to_id_map[name]
                    for name in _source_dict['class_names']
                ]
            else:
                raise ValueError(f'Invalid group data: {_source_dict}')
            return ClassGroupData(
                name=_source_dict['group_name'],
                class_ids=class_ids,
                rendering_view=_source_dict['rendering_view'],
            )

        class_groups_ = OrderedDict()
        if isinstance(class_groups, Path):
            with class_groups.open('rb') as f:
                json_data = json.load(f)
            for line_dict in json_data:
                group_data = _create_group_data_from_dict(line_dict)
                if group_data.name in class_groups_:
                    raise ValueError(f'Duplicate group name: {group_data.name}')
                class_groups_[group_data.name] = group_data
            del json_data
        elif isinstance(class_groups, (list, tuple)):
            for group_data in class_groups:
                if isinstance(group_data, dict):
                    group_data = _create_group_data_from_dict(group_data)
                elif isinstance(group_data, ClassGroupData):
                    pass
                else:
                    raise ValueError(f'Invalid group data: {group_data}')
                if group_data.name in class_groups_:
                    raise ValueError(f'Duplicate group name: {group_data.name}')
                class_groups_[group_data.name] = group_data
        else:
            raise ValueError(f'Invalid class groups: {class_groups}')
        self._class_groups = class_groups_
        del class_groups_
        self._n_groups = 10

        if len(self._class_groups) != self._n_groups:
            raise ValueError(f'Expected {self._n_groups} groups, got {len(self._class_groups)}')

        self._report_pptx_canvas_px = self._report_ppt.collect_shape_sizes(dpi=96)

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
            save_path: Path,
            fig_dpi=96,
            device='cpu'
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

        age_sex_hu_df = self._hu_statistics_df.filter(
            (pl.col('sex') == sex)
            & (pl.col('age_group_low') <= age)
            & (pl.col('age_group_high') >= age)
        )

        max_rows = max(len(v.class_ids) for v in self._class_groups.values() if v is not None)
        row_gap_px = 8
        common_box_height_px = max(10, int((self._min_canvas_h - max_rows * row_gap_px) / max_rows))
        common_box_height_px = min(common_box_height_px, 20)
        s_star_common = figure_utils.star_size_points2_from_box_height(
            common_box_height_px, dpi=fig_dpi, scale=1.5
        )

        report_ppt = self._report_ppt.copy()
        ppt_image_dict = {}
        # '1' to '10' box plots
        for idx, group_name in enumerate(self._class_groups.keys(), start=1):
            class_group_data = self._class_groups[group_name]
            if class_group_data is None:
                continue
            ppt_image_key = f'{idx:02d}'
            box_drawing_data = []
            for class_id in class_group_data.class_ids:
                class_df = age_sex_hu_df.filter(pl.col('class_id') == class_id)
                if len(class_df) != 1:
                    raise ValueError(
                        f'Unexpected number of rows {sex=} {age=} {class_id=}: {len(class_df)}'
                    )
                row = class_df.row(0, named=True)
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

        labelmap_renderer = LabelmapRenderer(
            labelmap=labelmap.astype(np.int16, copy=False), spacing=spacing,
            voxel_limit=2048,
            image_size=(1500, 2000),
            alpha_bit_planes=self._rendering_config.get('alpha_bit_planes', None),
            multi_samples=self._rendering_config.get('multi_samples', None),
            use_depth_peeling=self._rendering_config.get('use_depth_peeling', None),
            use_depth_peeling_for_volumes=self._rendering_config.get('use_depth_peeling_for_volumes', None),
            maximum_number_of_peels=self._rendering_config.get('maximum_number_of_peels', None),
            occlusion_ratio=self._rendering_config.get('occlusion_ratio', None),
        )

        front_view = labelmap_renderer.render(
            view="front",
            class_color_table=self._class_color_table,
            camera_offset=2500.0,
            shade=self._rendering_config.get('shade', None),
            specular=self._rendering_config.get('specular', None),
            specular_power=self._rendering_config.get('specular_power', None),
            ambient=self._rendering_config.get('ambient', None),
            diffuse=self._rendering_config.get('diffuse', None),
            scalar_opacity_unit_distance=self._rendering_config.get('scalar_opacity_unit_distance', None),
            blend_mode=self._rendering_config.get('blend_mode', None),
            device=device,
        )
        back_view = front_view
        # '11' '12' full lable view
        ppt_image_dict['11'] = front_view
        ppt_image_dict['12'] = back_view
        del front_view, back_view

        for insert_idx, group_name in enumerate(self._class_groups, start=15):
            class_group_data = self._class_groups[group_name]
            if class_group_data is None:
                continue
            #
            class_color_table = copy.deepcopy(self._class_color_table)
            visible_class_ids = set(class_group_data.class_ids)
            visible_class_ids.add(self._skin_class_id)
            for class_id, (r, g, b, a) in class_color_table.items():
                if class_id not in visible_class_ids:
                    class_color_table[class_id] = (r, g, b, 0)

            rendered_image = labelmap_renderer.render(
                view=class_group_data.rendering_view,
                class_color_table=class_color_table,
                bound_z_class_ids=class_group_data.class_ids,
                camera_offset=2500.0 / 2.3,
                shade=self._rendering_config.get('shade', None),
                specular=self._rendering_config.get('specular', None),
                specular_power=self._rendering_config.get('specular_power', None),
                ambient=self._rendering_config.get('ambient', None),
                diffuse=self._rendering_config.get('diffuse', None),
                scalar_opacity_unit_distance=self._rendering_config.get('scalar_opacity_unit_distance', None),
                blend_mode=self._rendering_config.get('blend_mode', None),
                out_size=(1500, 2000),
                device=device,
            )
            ppt_image_dict[f'{insert_idx:02d}'] = rendered_image
            del rendered_image

        ppt_image_dict['13'] = ppt_image_dict['19'].copy()
        ppt_image_dict['14'] = ppt_image_dict['20'].copy()

        fit_mode: Dict[str, TypeReportFitMode] = {}
        fit_mode.update({f'{i:02d}': 'match_height' for i in range(1, 13)})
        fit_mode.update({'15': 'match_width', '21': 'match_width'})

        fill_images = {}
        for image_key in ppt_image_dict:
            fill_data = ReportFillImageData(
                name=image_key,
                image=ppt_image_dict[image_key],
                fill_mode=fit_mode[image_key] if image_key in fit_mode else 'contain',
                send_to_back=image_key in {'11', '12'},
            )
            fill_images[image_key] = fill_data
            del fill_data
        del ppt_image_dict, fit_mode
        report_ppt.fill_images(fill_images)
        report_ppt.save(save_path)


    @staticmethod
    def _read_dataframe(read_path: Path) -> pl.DataFrame:
        if read_path.suffix == '.csv':
            return pl.read_csv(read_path)
        elif read_path.suffix == '.xlsx' or read_path.suffix == '.xls':
            return pl.read_excel(read_path)
        else:
            raise ValueError(f'Unsupported file type: {read_path.suffix}')