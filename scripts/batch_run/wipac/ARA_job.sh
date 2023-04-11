#!/bin/bash

# load in variables
key=cw_band
station=$1
run=$2
blind=0
condor_run=0
not_override=0
qual_type=1
l2_data=0
no_tqdm=1

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/script_executor.py -k ${key} -s ${station} -r ${run} -b ${blind} -c ${condor_run} -n ${not_override} -q ${qual_type} -t ${no_tqdm} -l ${l2_data}

