#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from pathlib import Path
import polars as pl

def main():
    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    statistics_dfs = pl.read_excel(
        this_dir / '..' /'resources/legacy/mean_std_hu.xlsx',
        sheet_name=['男性_mean', '男性_std', '女性_mean', '女性_std']
    )
    male_mean_df = statistics_dfs['男性_mean'].rename({'__UNNAMED__0': 'age_group'})
    male_std_df = statistics_dfs['男性_std'].rename({'__UNNAMED__0': 'age_group'})
    female_mean_df = statistics_dfs['女性_mean'].rename({'__UNNAMED__0': 'age_group'})
    female_std_df = statistics_dfs['女性_std'].rename({'__UNNAMED__0': 'age_group'})

    # check columns
    columns = tuple(male_mean_df.columns)
    assert len(columns) == 81
    assert columns == tuple(male_mean_df.columns)
    assert columns == tuple(male_std_df.columns)
    assert columns == tuple(female_mean_df.columns)
    assert columns == tuple(female_std_df.columns)

    final_df_data = []
    for sex in ['male', 'female']:
        if sex == 'male':
            mean_df = male_mean_df
            std_df = male_std_df
        elif sex == 'female':
            mean_df = female_mean_df
            std_df = female_std_df
        else:
            raise ValueError(f'{sex=}')
        columns = mean_df.columns
        columns = [c for c in columns if c != 'age_group']
        for mean_row, std_row in zip(
                mean_df.iter_rows(named=True), std_df.iter_rows(named=True)
        ):
            age_group = mean_row['age_group']
            age_low, age_high = age_group.split('-')
            age_low = int(age_low)
            age_high = int(age_high)

            for i, column_name in enumerate(columns):
                # HU_pelvis -> pelvis
                structure_name = column_name[3:]

                mean_val = mean_row[column_name]
                std_val = std_row[column_name]

                # if mean_val is None and std_val is None:
                #     continue
                #
                # if mean_val is None and std_val is not None:
                #     raise ValueError(f'{column_name=} {mean_val=}')
                # if mean_val is not None and std_val is None:
                #     raise ValueError(f'{column_name=} {std_val=}')

                final_df_data.append({
                    'class_name': structure_name,
                    'class_id': i + 1,
                    'sex': sex,
                    'age_group_low': age_low,
                    'age_group_high': age_high,
                    'mean': mean_val,
                    'std': std_val,
                })
    final_df = pl.DataFrame(final_df_data)
    assert len(final_df.unique(['class_id', 'sex', 'age_group_low'])) == len(final_df)
    final_df = final_df.sort(
        by=['class_id', 'sex', 'age_group_low'])
    final_df.write_excel(output_dir / 'hu_statistics.xlsx')



if __name__ == '__main__':
    main()