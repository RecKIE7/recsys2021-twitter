#!/bin/bash

read target
NOW=$(date +"%Y-%m-%d_%H:%M:%S")
nohup python train.py --target $1 train > ./log/log_$1_train_$NOW &


