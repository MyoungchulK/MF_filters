#!/bin/bash

# load in variables
key=snr
sim_key=signal
st=$1
run=$2
en=$3
fla=$4
sim_run=$5
year=2015
not_override=0
user_path=/misc/disk19/users/mkim/OMF_filter/
if [ -d "$user_path" ]; then
    echo "There is ${user_path}"
else
    #echo "Switch to /data/user/"
    #user_path=/data/user/
    echo "Switch to /data/ana/ARA/"
    user_path=/data/ana/ARA/
fi
data=${user_path}ARA0${st}/sim_${sim_key}_full/AraOut.${sim_key}_E${en}_F${fla}_A${st}_R${run}.txt.run${sim_run}.root


# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key} -s ${st} -y ${year} -d ${data} -n ${not_override}

