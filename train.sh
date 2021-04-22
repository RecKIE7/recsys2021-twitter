#!/bin/bash

NOW=$(date +"%Y-%m-%d_%H:%M:%S")
nohup python train.py train > ./log/log_train_$NOW &

