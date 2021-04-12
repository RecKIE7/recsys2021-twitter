#!/bin/bash


# for i in `seq 0 290`; 
# do
for entry in `ls /hdd/twitter/raw_lzo/`; 
do
    echo "seed ${entry}";
	python load_all.py --file ${entry} > /hdd/log/load_all.py_s${entry}.txt;
done

