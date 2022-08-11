#!/bin/bash

# load in variables
data=$1
output=$2

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
#source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/AraSoft/mwx2root/

./mwx2root -o ${output} ${data}

