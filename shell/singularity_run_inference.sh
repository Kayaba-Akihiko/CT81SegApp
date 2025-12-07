#!/bin/bash
#
# Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
# Biomedical Imaging Intelligence Laboratory,
# Nara Institute of Science and Technology.
# All rights reserved.
# This file can not be copied and/or distributed
# without the express permission of Yi GU.
#

script_dir="$(cd "$(dirname "$0")" && pwd)"
ct_input=${script_dir}/input/UZU00001_CT1
output_dir=${script_dir}/output
batch_size=4
num_core=8

simg_path="${script_dir}"/src/py3.12-torch2.8-cu12.8_latest.sif

singularity exec --nv \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env NUMEXPR_MAX_THREADS=${num_core} \
    "${simg_path}" \
    python "${script_dir}"/src/basic_2d.py \
    -i "${ct_input}" -o "${output_dir}" \
    -n "${num_core}" -b "${batch_size}" \
    --dicom_name_regex ".*"