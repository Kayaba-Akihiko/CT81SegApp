#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from pathlib import Path
import json
import polars as pl
from collections import OrderedDict


def main():
    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    resource_root = this_dir / '..' / 'resources'
    class_table_path = resource_root / 'class_table.csv'

    muscle_groups = OrderedDict()
    muscle_groups.update({
        '1': ['pectoralis_major', 'pectoralis_minor', 'serratus_anterior', 'intercostal_muscles',
              'rectus_abdominis', 'internal_oblique', 'external_oblique', 'transversus_abdominis'],
        '2': ['adductor_muscles', 'pectineus_muscle', 'gracilis_muscle', 'sartorius_muscle'],
        '3': ['rectus_femoris_muscle',
              'vastus_lateralis_muscle_and_vastus_intermedius_muscle',
              'vastus_medialis_muscle'],
        '4': ['anterior_compartment_muscles', 'lateral_compartment_muscles'],
        '5': ['psoas_major_muscle', 'iliacus_muscle', 'quadratus_lumborum'],
        '6': ['obturator_internus_muscle', 'obturator_externus_muscle', 'piriformis_muscle'],
        '7': ['erector_spinae', 'latissimus_dorsi', 'trapezius', 'supraspinatus',
              'infraspinatus', 'serratus_anterior', 'subscapularis',
              'teres_minor_muscle', 'teres_minor'],
        '8': ['gluteus_maximus_muscle', 'gluteus_medius_muscle',
              'gluteus_minimus_muscle', 'tensor_fasciae_latae_muscle'],
        '9': ['biceps_femoris_muscle', 'semitendinosus_muscle', 'semimembranosus_muscle'],
        '10': ['superficial_posterior_compartment_muscles', 'deep_posterior_compartment_muscles'],
    })

    excluding_muscles = {
                'intercostal_muscles',  'pectoralis_major', 'pectoralis_minor', 'serratus_anterior',
                'trapezius', 'supraspinatus', 'infraspinatus',
                'subscapularis', 'teres_minor_muscle', 'teres_minor'
            }

    front_groups = {'1', '2', '3', '4', '5'}

    save_path = output_dir / 'class_groups.json'
    class_table_df = pl.read_csv(class_table_path)
    class_name_to_id_map = {}
    class_id_to_name_map = {}
    for row in class_table_df.iter_rows(named=True):
        class_name_to_id_map[row['class_name']] = row['class_id']
        class_id_to_name_map[row['class_id']] = row['class_name']

    json_data = []
    for group_name, class_names in muscle_groups.items():
        class_names = [name for name in class_names if name not in excluding_muscles]
        class_ids = [class_name_to_id_map[name] for name in class_names]
        if group_name in front_groups:
            view = 'front'
        else:
            view = 'back'
        line_data = {
            'group_name': group_name,
            'class_ids': class_ids,
            'class_names': class_names,
            'rendering_view': view,
        }
        json_data.append(line_data)
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)

if __name__ == '__main__':
    main()