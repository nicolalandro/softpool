#!/bin/bash

GPU=0
BATH=16
N_REG=8
N_POINTS=24ì048

CUDA_VISIBLE_DEVICES=${GPU} python3 train.py --batch ${BATCH} \
                                        --n_regions ${N_REG} \
			                            --num_points ${N_POINTS} \
                                        --dataset shapenet \
                                        --savepath ijcv_shapenet_softpool \
                                        --methods softpool > train.log
