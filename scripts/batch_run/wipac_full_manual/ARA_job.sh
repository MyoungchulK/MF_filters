#!/bin/bash

# load in variables
key=reco_ele_lite_q1
station=$1
run=$2
blind=1
condor_run=0
not_override=1
qual_type=3
no_tqdm=1

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /home/mkim/.bashrc
source /cvmfs/ara.opensciencegrid.org/trunk/RHEL_8_x86_64/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

data=/misc/disk19/users/mkim/OMF_filter/ARA0${station}/temp/event00${run}.root
ped=/misc/disk19/users/mkim/OMF_filter/ARA0${station}/ped_full/ped_full_values_A${station}_R${run}.dat
if [ -f "$data" ]; then
    echo "PASS!"
else
    data=/misc/disk19/users/mkim/OMF_filter/ARA0${station}/temp/event${run}.root
fi

if [ -f "$data" ]; then
    echo "PASS2!"
else
    data=/misc/disk19/users/mkim/OMF_filter/ARA0${station}/temp/event0${run}.root
fi

python3 /home/mkim/analysis/MF_filters/scripts/script_executor.py -k ${key} -t ${no_tqdm} -b ${blind} -dd ${data} -pp ${ped} -s ${station} -r ${run} -n ${not_override}

