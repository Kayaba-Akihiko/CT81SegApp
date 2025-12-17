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
batch_size=2
num_core=8
num_gpus=1

simg_path="${script_dir}"/src/resources/py3.12-torch2.8-cu12.8_latest.sif
#simg_path=/win/flounder/user/koku/sif/py3.12-torch2.8-cu12.8_latest.sif

mkdir -p "${output_dir}"
echo "[Script start] $(date)" 2>&1 | tee -a "${output_dir}"/inference.log
start=$(date +%s)
singularity exec --nv --nvccli \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env NUMEXPR_MAX_THREADS=${num_core} \
    "${simg_path}" \
    python "${script_dir}"/src/main.py \
    --image_path "${ct_input}" --output_dir "${output_dir}" \
    --n_workers "${num_core}" --batch_size "${batch_size}" \
    --dist_devices "${num_gpus}" \
    --dicom_name_regex ".*"
end=$(date +%s)
echo "[Script end] $(date)" 2>&1 | tee -a "${output_dir}"/inference.log
echo "Script elapsed: $((end - start)) sec" 2>&1 | tee -a "${output_dir}"/inference.log
