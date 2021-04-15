#!/bin/bash

target=""

if [ "$1" = "" ]; then
echo "Please enter the target name."

else 

target=$1
NOW=$(date +"%Y-%m-%d_%H:%M:%S")
nohup python train.py --target "$target" train > ./log/log_"$target"_train_$NOW &

fi



