#!/bin/bash

# load in variables
key=temp
st=$1
run=$2
sim_run=$3
setup=/home/mkim/analysis/MF_filters/sim/ARA0${st}/sim_${key}_setup_full/${key}_A${st}_R${run}.txt
evt_setup=/home/mkim/analysis/MF_filters/sim/ARA0${st}/sim_${key}_setup_full/${key}_A${st}_R${run}_setup.txt
user_path=/misc/disk19/users/mkim/OMF_filter/
if [ -d "$user_path" ]; then
    echo "There is ${user_path}"
else
    #echo "Switch to /data/user/"
    #user_path=/data/user/
    echo "Switch to /data/ana/ARA/"
    user_path=/data/ana/ARA/
fi
result=${user_path}ARA0${st}/sim_${key}_full

if [ -d "$result" ]; then
    echo "There is ${result}"
else
    echo "Make ${result}"
    mkdir ${result}
fi

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/AraSoft/AraSim/

./AraSim ${setup} ${sim_run} $TMPDIR ${evt_setup}

# at the end, move the results back
mv $TMPDIR/*.root ${result}

