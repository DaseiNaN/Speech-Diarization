#!/bin/bash

exp_dir=/home/dasein/Projects/Speech-Diarization/logs/experiments/runs

model_name=sa_eend_spk_embed
model_time=2022-03-24_18-18-19


# /home/dasein/Projects/Speech-Diarization/logs/experiments/runs/sa_eend_spk_embed/2022-03-24_18-18-19
avg_model=$exp_dir/$model_name/$model_time/avg/avg.ckpt

ifiles=`eval echo $exp_dir/$model_name/$model_time/checkpoints/epoch_0{90..99}.ckpt`
python src/vendor/model_averaging.py $avg_model $ifiles
