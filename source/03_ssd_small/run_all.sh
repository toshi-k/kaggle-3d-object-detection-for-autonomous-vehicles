#!/bin/bash

python3 01_train.py --dataset ds_range50 --debug False

python3 02_detect.py --dataset ds_range50 --debug False

python3 03_make_sub.py --debug False
