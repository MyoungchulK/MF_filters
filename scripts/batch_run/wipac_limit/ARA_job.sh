#!/bin/bash

# load in variables
station=$1
trig=$2
run=$3
run_w=$4
ant_c=1
smear_l=20

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /home/mkim/.bashrc
source /cvmfs/ara.opensciencegrid.org/trunk/RHEL_8_x86_64/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/upper_limit_summary.py ${station} ${run}

