#!/bin/bash

# load in variables
data=$1
evt=$2

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py mf_noise 2 2015 ${data} ${evt}

