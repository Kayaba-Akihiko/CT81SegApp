

from pathlib import Path
import numpy as np

import polars as pl


def main():

    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)


    hu_statistics_df = pl.read_excel(this_dir / 'hu_statistics.xlsx')
    hu_statistics_df = hu_statistics_df.sort(by='age_group_low')
    class_ids = hu_statistics_df['class_id'].unique().to_numpy()
    sexes = hu_statistics_df['sex'].unique().to_numpy()
    additional_hu_statistics_data = []
    for class_id in class_ids:
        class_hu_statistics_df = hu_statistics_df.filter(pl.col('class_id') == class_id)
        for sex in sexes:
            sex_hu_statistics_df = class_hu_statistics_df.filter(pl.col('sex') == sex)
            if len(sex_hu_statistics_df) == 0:
                continue
            row = sex_hu_statistics_df.row(0, named=True)
            additional_hu_statistics_data.append({
                "class_name": row['class_name'],
                "class_id": class_id,
                "sex": sex,
                "age_group_low": 0,
                'age_group_high': row['age_group_low'] - 1,
                'mean': row['mean'],
                'std': row['std'],
            })
            row = sex_hu_statistics_df.row(-1, named=True)
            additional_hu_statistics_data.append({
                "class_name": row['class_name'],
                "class_id": class_id,
                "sex": sex,
                "age_group_low": row['age_group_high'] + 1,
                'age_group_high': 200,
                'mean': row['mean'],
                'std': row['std'],
            })
    hu_statistics_df = pl.concat(
        [hu_statistics_df,
         pl.DataFrame(additional_hu_statistics_data)]
    )
    hu_statistics_df = hu_statistics_df.sort(by=['class_id', 'sex', 'age_group_low'])
    hu_statistics_df.write_excel(output_dir / 'hu_statistics_extended.xlsx')



if __name__ == '__main__':
    main()