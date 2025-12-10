#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from pathlib import Path

import polars as pl
import json


def main():
    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    config_path = this_dir / '..' / 'resources' / 'legacy' / 'naist_totalsegmentator_81.json'

    res_df_data = []
    with open(config_path, 'r') as f:
        config = json.load(f)
        for class_name, (class_id, r, g, b, a) in config['color'].items():
            res_df_data.append({
                'class_id': class_id,
                'class_name': class_name,
                'r': r, 'g': g, 'b': b, 'a': a
            })
    res_df = pl.DataFrame(res_df_data).sort('class_id')
    res_df.write_csv(output_dir / 'class_table.csv')


if __name__ == '__main__':
    main()