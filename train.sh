#!/bin/bash

NOW=$(date +"%Y-%m-%d_%H:%M:%S")
nohup python train.py train --target 0 > ./log/log_train_$NOW &


