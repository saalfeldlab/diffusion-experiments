#!/bin/bash
cp inference_config_a100.yaml inference_configs/inference_config_a100_$LSB_JOBID.yaml
sed -i -E "s/^(\s*dir:\s*)(.*)$/\1samples_${LSB_JOBID:=\2}/" inference_configs/inference_config_a100_$LSB_JOBID.yaml
python src/exp08/inference.py inference_configs/inference_config_a100_$LSB_JOBID.yaml
