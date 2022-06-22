#!/bin/bash

# load in variables
st=$1
data=$2
key=rayl

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py ${key} ${st} 2015 ${data}

