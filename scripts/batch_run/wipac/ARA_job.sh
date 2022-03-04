#!/bin/bash

# load in variables
key=qual_cut
station=$1
run=$2

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/AraSoft/for_local_araroot.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/script_executor.py ${key} ${station} ${run}

