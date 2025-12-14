#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import IO, Self, Tuple, Optional, Union, Dict, Any, TypeAlias, Literal, Type, TypeVar, Protocol, List, Sequence
from pathlib import Path
from collections import OrderedDict
import json
import copy
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt
import polars as pl

from xmodules.xutils import vtk_utils, os_utils
from xmodules.typing import TypePathLike
from xmodules.xdistributor.protocol import DistributorProtocol

from .modules.report_ppt import ReportPPT, PPTXPresentation, FillImageData as ReportFillImageData, TypeFitMode as TypeReportFitMode
from .modules.labelmap_renderer import LabelmapRenderer
from .modules.utils import figure_utils

TypeSex: TypeAlias = Literal['male', 'female']

@dataclass
class ClassGroupData:
    name: str
    class_ids:List[int]
    rendering_view: vtk_utils.TypeView

@dataclass
class PatientInfoData:
    name: Optional[str] = None
    sex: Optional[TypeSex] = None
    age: Optional[Union[float, int]] = None
    birth_year: Optional[int] = None
    birth_month: Optional[int] = None
    birth_day: Optional[int] = None
    height: Optional[Union[float, int]] = None
    weight: Optional[Union[float, int]] = None
    shooting_year: Optional[int] = None
    shooting_month: Optional[int] = None
    shooting_day: Optional[int] = None

    @classmethod
    def from_dict(cls, source_dict: Dict[str, Any]) -> Self:
        # valid field names (lowercase → actual name)
        field_map = {f.name.lower(): f.name for f in fields(cls)}

        build_dict: Dict[str, Any] = {}

        for key, val in source_dict.items():
            key_l = key.lower()

            if key_l not in field_map:
                continue  # ignore unknown keys

            field_name = field_map[key_l]

            if field_name == "sex" and val is not None:
                try:
                    val = cls._format_sex(val)
                except (TypeError, ValueError):
                    val = None
                build_dict[field_name] = val
            elif field_name == "age":
                try:
                    val = cls._format_age(val)
                except (TypeError, ValueError, OverflowError):
                    val = None
                build_dict[field_name] = val
            else:
                # ... lots to check ...
                build_dict[field_name] = val

        return cls(**build_dict)

    @staticmethod
    def _format_sex(sex: str) -> TypeSex:
        if not isinstance(sex, str):
            raise TypeError(f'Invalid type {type(sex)=}')
        sex = sex.lower()
        if sex in {'m', 'male', '男性', '男'}:
            return 'male'
        elif sex in {'f', 'female', '女性', '女'}:
            return 'female'
        else:
            raise ValueError(f'Invalid sex {sex=}')

    @staticmethod
    def _format_age(age: Union[int, float, str]) -> int:
        if isinstance(age, str):
            age = age.lower()
            if age.endswith('y'):
                age = age[:-1]
        elif isinstance(age, (int, float)):
            pass
        else:
            raise TypeError(f'Invalid type {type(age)=}')
        return int(age)


class ReportGenerator:

    def __init__(
            self,
            distributor: DistributorProtocol,
            template_ppt: Union[PPTXPresentation, TypePathLike],
            hu_statistics_table: Union[pl.DataFrame, TypePathLike],
            rendering_config: Union[Dict[str, Any], TypePathLike],
            class_info_table: Union[pl.DataFrame, TypePathLike],
            class_groups: Union[
                List[Union[ClassGroupData, Dict[str, Any]]],
                Tuple[Union[ClassGroupData, Dict[str, Any]], ...],
                TypePathLike,
            ],
            observation_messages: Union[Dict[int, str], TypePathLike],
    ):
        self._distributor = distributor
        if os_utils.is_path_like(template_ppt):
            template_ppt = os_utils.format_path_string(template_ppt)
        self._report_ppt = ReportPPT(template_ppt)

        self._hu_statistics_df = None
        if self._distributor.is_main_process():
            if os_utils.is_path_like(hu_statistics_table):
                self._hu_statistics_df = self._read_dataframe(
                    os_utils.format_path_string(hu_statistics_table))
            elif isinstance(hu_statistics_table, pl.DataFrame):
                self._hu_statistics_df = hu_statistics_table.clone()
            else:
                raise TypeError(f'Invalid hu_statistics_table type: {type(hu_statistics_table)=}')
        if self._distributor.is_distributed():
            self._hu_statistics_df = distributor.broadcast_object(
                self._hu_statistics_df)

        if self._distributor.is_main_process():
            if os_utils.is_path_like(rendering_config):
                with os_utils.format_path_string(rendering_config).open('rb') as f:
                    rendering_config = json.load(f)
            elif isinstance(rendering_config, dict):
                rendering_config = copy.deepcopy(rendering_config)
            else:
                raise TypeError(f'Invalid rendering_config type: {type(rendering_config)=}')
        if self._distributor.is_distributed():
            rendering_config = distributor.broadcast_object(rendering_config)

        self._skin_class_id = rendering_config.pop('skin_class_id')
        self._rendering_config = rendering_config

        self._class_info_df = None
        if self._distributor.is_main_process():
            if os_utils.is_path_like(class_info_table):
                self._class_info_df = self._read_dataframe(
                    os_utils.format_path_string(class_info_table))
            elif isinstance(class_info_table, pl.DataFrame):
                self._class_info_df = class_info_table.clone()
            else:
                raise TypeError(f'Invalid class_info_table type: {type(class_info_table)=}')
        if self._distributor.is_distributed():
            self._class_info_df = self._distributor.broadcast_object(self._class_info_df)

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
            if len(class_ids) <= 0:
                raise ValueError(f'Empty class IDs: {_source_dict}')
            return ClassGroupData(
                name=_source_dict['group_name'],
                class_ids=class_ids,
                rendering_view=_source_dict['rendering_view'],
            )

        class_groups_ = None
        if self._distributor.is_main_process():
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
        if self._distributor.is_distributed():
            class_groups_ = self._distributor.broadcast_object(class_groups_)
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

        if self._distributor.is_main_process():
            if os_utils.is_path_like(observation_messages):
                with os_utils.format_path_string(observation_messages).open('rb') as f:
                    observation_messages = json.load(f)
            elif isinstance(observation_messages, dict):
                observation_messages = copy.deepcopy(observation_messages)
            else:
                raise TypeError(f'Invalid observation_message type: {type(observation_messages)=}')

            cleaned_observation_messages = {}
            for k, v in observation_messages.items():
                k = int(k)
                if k not in range(1, 4):
                    raise ValueError(f'Invalid observation message key: {k}')
                if not isinstance(v, str):
                    raise TypeError(f'Invalid observation message value: {v}')
                cleaned_observation_messages[k] = v
            observation_messages = cleaned_observation_messages
            del cleaned_observation_messages
        if self._distributor.is_distributed():
            observation_messages = self._distributor.broadcast_object(observation_messages)
        assert isinstance(observation_messages, dict)
        self._observation_messages = observation_messages
        #
        # self._observation_messages = {
        #     1: 'あなたの筋肉の質は、同性・同年代と比べて良好あるいは標準的な範囲にあります。'
        #        'ただし、高齢になると平均的な値でも筋力低下や転倒リスクが高まることがありますので、'
        #        '今後もバランスの良い食事と（できるだけお医者様や専門家指導の下で）適度な運動を続けることが大切です。',
        #     2: 'あなたの筋肉の質は、同性・同年代と比べるとやや低めの傾向があります。'
        #        '高齢期には筋肉の質のわずかな低下でも生活機能に影響が出ることがありますので、'
        #        '（できるだけお医者様や専門家指導の下で）食事・運動を工夫し、'
        #        '必要に応じて専門家のアドバイスを受けることをおすすめします。',
        #     3: 'あなたの筋肉の質は、同性・同年代と比べて低い傾向が見られます。'
        #        '高齢期では平均的な水準でも転倒や要介護のリスクが高まることが知られています。'
        #        '早めに医療・リハビリ専門家にご相談されることをおすすめします。',
        # }

    def generate_hu_table(
            self,
            class_mean_hus: npt.NDArray[np.float64],
            class_volumes: npt.NDArray[np.float64],
    ) -> pl.DataFrame:
        def _check(_array: npt.NDArray):
            if not isinstance(_array, np.ndarray):
                raise TypeError(f'Invalid array type: {type(_array)}')
            if not _array.ndim == 1:
                raise ValueError(f'Invalid array shape: {_array.shape}')
            if not len(_array) == len(self._class_id_to_name_map):
                raise ValueError(f'Invalid array length: {_array.shape}')
        _check(class_mean_hus)
        _check(class_volumes)

        # Build data
        res_df_data = []
        for class_id, class_name in self._class_id_to_name_map.items():
            if class_id == 0:
                continue
            res_df_data.append({
                'Structure ID': class_id,
                'Structure name': class_name,
                'Mean HU': class_mean_hus[class_id],
                'Volume [cm^3]': class_volumes[class_id],
            })
        res_df = pl.DataFrame(res_df_data).sort(by='Structure ID')
        return res_df

    def generate_report(
            self,
            patient_info: Union[PatientInfoData, Dict[str, Any]],
            labelmap: npt.NDArray[np.integer],
            spacing: npt.NDArray[np.float64],
            class_mean_hus: npt.NDArray[np.float64],
            fig_dpi=96,
            device='cpu',
    ) -> ReportPPT:

        if labelmap.ndim != 3:
            raise ValueError('Labelmap must be 3D')
        if spacing.ndim != 1:
            raise ValueError('Spacing must be 1D')
        if len(spacing) != labelmap.ndim:
            raise ValueError('Spacing must have the same length as labelmap')

        if isinstance(patient_info, dict):
            patient_info = PatientInfoData.from_dict(patient_info)
        elif isinstance(patient_info, PatientInfoData):
            pass
        else:
            raise TypeError(f'Invalid patient info type: {type(patient_info)}')

        sex = patient_info.sex
        age = patient_info.age

        age_sex_hu_df = self._hu_statistics_df
        if sex is not None:
            age_sex_hu_df = age_sex_hu_df.filter(
                pl.col('sex') == sex)
        if age is not None:
            age_sex_hu_df = age_sex_hu_df.filter(
                (pl.col('age_group_low') <= age)
                & (pl.col('age_group_high') >= age)
            )
        if class_mean_hus is None:
            raise ValueError(f'class_mean_hus is required in patient_info')
        class_mean_hus: npt.NDArray[np.float64]
        all_visual_class_ids = set()
        for class_group_data in self._class_groups.values():
            if class_group_data is None:
                continue
            all_visual_class_ids.update(set(class_group_data.class_ids))
        age_sex_hu_df = age_sex_hu_df.filter(
            pl.col('class_id').is_in(all_visual_class_ids))

        max_rows = max(len(v.class_ids) for v in self._class_groups.values() if v is not None)
        row_gap_px = 8
        common_box_height_px = max(10, int((self._min_canvas_h - max_rows * row_gap_px) / max_rows))
        common_box_height_px = min(common_box_height_px, 20)
        s_star_common = figure_utils.star_size_points2_from_box_height(
            common_box_height_px, dpi=fig_dpi, scale=1.5
        )
        means, stds = age_sex_hu_df['mean', 'std'].to_numpy().astype(np.float64).T
        xlim = figure_utils.compute_lim(
            np.concatenate([
                means,
                means - stds,
                means + stds,
                class_mean_hus[
                    np.asarray(list(all_visual_class_ids), dtype=np.int64)
                ]
            ]),
            pad_ratio=0.12
        )
        del means, stds, all_visual_class_ids

        def _mix_gaussians(_means, _vars, _weights=None):
            _means = np.asarray(_means, dtype=np.float64)
            _vars = np.asarray(_vars, dtype=np.float64)

            if _means.shape != _vars.shape:
                raise ValueError("means and vars must have the same shape")

            _n = _means.size
            if _weights is None:
                _weights = np.full(_n, 1.0 / _n)
            else:
                _weights = np.asarray(_weights, dtype=np.float64)
                _weights = _weights / _weights.sum()

            _mu = np.sum(_weights * _means)
            _var = np.sum(_weights * (_vars + _means ** 2)) - _mu ** 2
            return _mu, _var

        report_ppt = self._report_ppt.copy()

        # '1' to '10' box plots
        box_jobs = list(enumerate(self._class_groups.keys(), start=1))
        if self._distributor.is_distributed():
            if len(box_jobs) < self._distributor.world_size:
                raise ValueError(f'')
            n_jobs_per_rank = (len(box_jobs) + self._distributor.world_size - 1) // self._distributor.world_size
            start = self._distributor.global_rank * n_jobs_per_rank
            end = start + n_jobs_per_rank
            box_jobs = box_jobs[start: end]

        # Use dict to avoid duplicated class counts
        class_low_target_table = {}
        ppt_image_dict = {}
        for idx, group_name in box_jobs:
            class_group_data = self._class_groups[group_name]
            if class_group_data is None:
                continue
            ppt_image_key = f'{idx:02d}'
            box_drawing_data = []
            for class_id in class_group_data.class_ids:
                class_df = age_sex_hu_df.filter(pl.col('class_id') == class_id)
                if len(class_df) <= 0:
                    raise ValueError(f'No data for class {class_id=}')
                elif len(class_df) > 1:
                    # merge distributions
                    mean, std = np.einsum(
                        'ij->ji', class_df['mean', 'std'].to_numpy()
                    ).astype(np.float64)
                    # validity: finite mean/std and std > 0
                    valid = np.isfinite(mean) & np.isfinite(std) & (std > 0)
                    if np.any(valid):
                        mean, var = _mix_gaussians(mean, np.square(std))
                        if var > 0:
                            std = np.sqrt(var)
                        else:
                            std = np.nan
                        del var
                    else:
                        mean, std = np.nan, np.nan
                else:
                    row = class_df.row(0, named=True)
                    mean, std = float(row['mean']), float(row['std'])
                    del row

                del class_df

                target = float(class_mean_hus[class_id])
                if np.isfinite(target):
                    if np.isfinite(mean) and np.isfinite(std) and target < mean - std:
                        class_low_target_table[class_id] = 1
                    else:
                        class_low_target_table[class_id] = 0
                box_drawing_data.append(figure_utils.BoxDrawingData(
                    target=target,
                    mean=mean, std=std,
                    color=self._class_color_table[class_id][:3],
                ))
                del target
            box_figure = figure_utils.draw_hu_boxes(
                boxes=box_drawing_data,
                canvas_size_px=self._report_pptx_canvas_px[ppt_image_key],
                xlim=xlim,
                s_star=s_star_common,
                box_height_px=common_box_height_px,
                row_gap_px=row_gap_px,
                dpi=fig_dpi,
            )
            ppt_image_dict[ppt_image_key] = box_figure
            del box_drawing_data, box_figure
        if self._distributor.is_distributed():
            for rank_dict in self._distributor.all_gather_object(ppt_image_dict):
                ppt_image_dict.update(rank_dict)

            for rank_dict in self._distributor.all_gather_object(class_low_target_table):
                class_low_target_table.update(rank_dict)

        low_target_ratio = sum(class_low_target_table.values()) / len(class_low_target_table)
        # Determine observation
        if low_target_ratio >= 0.5:
            observation = self._observation_messages[3]
        elif low_target_ratio >= 0.2:
            observation = self._observation_messages[2]
        else:
            observation = self._observation_messages[1]
        del low_target_ratio, class_low_target_table
        report_ppt.fill_texts(
            self._build_text_placeholders(
                patient_info, observation, language='jp')
        )
        del observation

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

        front_view, back_view = None, None
        if self._distributor.is_distributed():
            front_view, back_view = labelmap_renderer.render(
                view=['front', 'back'],
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
        if self._distributor.is_distributed():
            front_view, back_view = self._distributor.broadcast_object(front_view, back_view)
        # '11' '12' full label view
        ppt_image_dict['11'] = front_view
        ppt_image_dict['12'] = back_view
        del front_view, back_view

        rendering_jobs = list(enumerate(self._class_groups, start=15))
        if self._distributor.is_distributed():
            if len(rendering_jobs) < self._distributor.world_size:
                raise ValueError(f'')
            n_jobs_per_rank = (len(rendering_jobs) + self._distributor.world_size - 1) // self._distributor.world_size
            start = self._distributor.global_rank * n_jobs_per_rank
            end = start + n_jobs_per_rank
            rendering_jobs = rendering_jobs[start: end]
        ppt_image_dict_ = {}
        for insert_idx, group_name in rendering_jobs:
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
            ppt_image_dict_[f'{insert_idx:02d}'] = rendered_image
            del rendered_image
        if self._distributor.is_distributed():
            for rank_dict in self._distributor.all_gather_object(ppt_image_dict_):
                ppt_image_dict_.update(rank_dict)

        ppt_image_dict.update(ppt_image_dict_)
        del ppt_image_dict_

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
        return report_ppt

    @staticmethod
    def _build_text_placeholders(
            patient_info: Optional[PatientInfoData] = None,
            observation_message: Optional[str] = None,
            default_str: str = '-',
            language: str = 'en'
    ) -> Dict[str, str]:
        if patient_info is None:
            patient_info = PatientInfoData()
        language = language.lower()

        placeholders = {
            'NAME': patient_info.name,
            'BIRTH_YEAR': patient_info.birth_year,
            'BIRTH_MONTH': patient_info.birth_month,
            'BIRTH_DAY': patient_info.birth_day,
            'HEIGHT': patient_info.height,
            'WEIGHT': patient_info.weight,
            'AGE': patient_info.age,
            'SHOOTING_YEAR': patient_info.shooting_year,
            'SHOOTING_MONTH': patient_info.shooting_month,
            'SHOOTING_DAY': patient_info.shooting_day,
            'OBSERVATIONS': observation_message,
        }

        # Sex mapping
        sex = patient_info.sex
        if sex is not None:
            sex = sex.lower()
            if sex in {'m', 'male', '男'}:
                if language == 'jp':
                    sex = '男性'
                elif language == 'en':
                    sex = 'Male'
                else:
                    raise ValueError(f'Unsupported language: {language}')
            elif sex in {'f', 'female', '女'}:
                if language == 'jp':
                    sex = '女性'
                elif language == 'en':
                    sex = 'Female'
                else:
                    raise ValueError(f'Unsupported language: {language}')
        placeholders['SEX'] = sex

        for k, v in placeholders.items():
            if v is None:
                placeholders[k] = default_str
            else:
                placeholders[k] = str(v)
        return placeholders

    @staticmethod
    def _read_dataframe(read_path: Path) -> pl.DataFrame:
        if read_path.suffix == '.csv':
            return pl.read_csv(read_path)
        elif read_path.suffix == '.xlsx' or read_path.suffix == '.xls':
            return pl.read_excel(read_path)
        else:
            raise ValueError(f'Unsupported file type: {read_path.suffix}')