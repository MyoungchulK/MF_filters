#!/bin/bash

# load in variables
key=snr
sim_type=noise
st=$1
run=$2
year=2015
not_override=0
user_path=/misc/disk19/users/
#user_path=/data/user/
data=${user_path}mkim/OMF_filter/ARA0${st}/sim_${sim_type}/AraOut.${sim_type}_A${st}_R${run}.txt.run0.root

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key} -s ${st} -y ${year} -d ${data} -n ${not_override}

