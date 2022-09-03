#!/bin/bash

# load in variables
setup=$1
run=$2
st=$3
result=/misc/disk19/users/mkim/OMF_filter/ARA0${st}/sim_temp
#result=/data/user/mkim/OMF_filter/ARA0${st}/sim_signal
evt_file=/home/mkim/analysis/MF_filters/sim/sim_temp/temp_A${st}_setup.txt

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/AraSoft/AraSim/

./AraSim ${setup} ${run} $TMPDIR ${evt_file}

# at the end, move the results back
mv $TMPDIR/*.root ${result}

