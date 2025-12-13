#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path

from xmodules.xutils import os_utils, dicom_utils

def main():
    this_file = Path(__file__)
    this_dir = this_file.parent

    ct_path = this_dir / 'report_gen_test' / 'UZU00001_CT1'

    ct_image, s, p, tags = dicom_utils.read_dicom_folder(
        ct_path,
        name_regex=".*",
        n_workers=8,
        progress_bar=True,
        required_tag=['PatientsName', 'PatientsAge']
    )
    print(tags)


if __name__ == '__main__':
    main()