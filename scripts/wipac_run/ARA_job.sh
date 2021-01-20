#!/bin/bash

# load in variables
data=$1
ped=$2
station=$3
run=$4
out=$5
mode=$6

# run the reconstruction script
ulimit -s 131072; /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode}

if [ $? -ne 0 ] #error handle if something has gone wrong
then
	echo ${data}_${ped}_${station}_${run} >> /data/user/mkim/OMF_filter/ARA0${station}/problems_A${station}_R${run}.txt
