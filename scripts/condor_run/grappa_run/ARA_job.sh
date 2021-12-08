#!/bin/bash

# load in variables
data=$1
ped=$2
station=$3
out=/misc/disk19/users/mkim/OMF_filter/ARA0${station}/PPS_Miss/

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/AraSoft/for_local_araroot.sh
cd /home/mkim/analysis/MF_filters/scripts/

#ulimit -s 131072; python3 /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode}
python3 /home/mkim/analysis/MF_filters/scripts/pps_miss_collector.py ${data} ${ped} ${out}

