#!/bin/bash

# load in variables
key=noise
st=$1
run=$2
sim_run=0
user_path=/misc/disk19/users/
#user_path=/data/user/
setup=${user_path}mkim/OMF_filter/ARA0${st}/sim_${key}_setup/${key}_A${st}_R${run}.txt
result=${user_path}mkim/OMF_filter/ARA0${st}/sim_${key}

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/AraSoft/AraSim/

./AraSim ${setup} ${sim_run} $TMPDIR

# at the end, move the results back
mv $TMPDIR/*.root ${result}

#FILE=${result}/AraOut.${key}_A${st}_R${run}.txt.run${sim_run}.root
#if [ -f "$FILE" ]; then
#    echo "$FILE exists."
#else
#    echo "$FILE does not exist."
#    ./AraSim ${setup} ${sim_run} ${result}
#fi
