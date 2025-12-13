#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from modules.report_generator.report_generator import ReportGenerator, ClassGroupData
from xmodules.xutils import metaimage_utils, dicom_utils

def main():
    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    resource_root = this_dir / '..' / '..' / '..' / 'resources'
    sex: Literal['male', 'female'] = 'female'
    age = 56
    labelmap_path = this_dir / 'pred_label.mha'
    ct_path = this_dir / 'UZU00001_CT1'

    labelmap, spacing, _ = metaimage_utils.read(labelmap_path)
    labelmap = labelmap.astype(np.uint8)
    ct_image, s, _ = dicom_utils.read_dicom_folder(
        ct_path, name_regex=".*", n_workers=8, progress_bar=True)
    assert np.allclose(spacing, s)
    mean_hus = _calculate_mean_hu(ct_image, labelmap)

    template_path = resource_root / 'MICBON_AI_report_template_p3.pptx'
    hu_statistics_table_path = resource_root / 'hu_statistics.xlsx'
    rendering_config = resource_root / 'rendering_config.json'
    class_table_path = resource_root / 'class_table.csv'
    class_groups_path = resource_root / 'class_groups.json'

    report_generator = ReportGenerator(
        template_ppt=template_path,
        hu_statistics_table=hu_statistics_table_path,
        rendering_config=rendering_config,
        class_info_table=class_table_path,
        class_groups=class_groups_path,
    )
    report_generator.generate(
        patient_info={
            'name': 'Taro',
            'sex': sex,
            'age': age,
        },
        labelmap=labelmap,
        spacing=spacing,
        class_mean_hus=mean_hus,
        pptx_save_path=output_dir / 'report.pptx',
        pdf_save_path=output_dir / 'report.pdf',
        image_save_path=output_dir / 'report.png',
        device='cuda',
    )


def _calculate_mean_hu(
        image: npt.NDArray, labelmap: npt.NDArray[np.integer],
        n_classes: int = 81,
):
    res = []
    image = image.astype(np.float32)
    for class_id in range(n_classes):
        mask = labelmap == class_id
        if not np.any(mask):
            res.append(None)
            continue
        res.append(np.mean(image[mask]))

    return np.asarray(res, dtype=np.float64)


if __name__ == '__main__':
    main()