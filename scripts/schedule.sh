#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# kaldi env
# sa_eend basic
python train.py experiment=sa_eend

# sa_eend_spk_embed
python train.py experiment=sa_eend_spk_embed
