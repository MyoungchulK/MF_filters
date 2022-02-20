#!/bin/bash

# load in variables
data=$1
station=$2
run=$3
ped=pedestalValues.run${run}.dat
out=/data/user/mkim/OMF_filter/ARA0${station}/Ped/${ped}

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
cd /cvmfs/ara.opensciencegrid.org/trunk/centos7/ara_build/bin/

#./repeder ${data} ${out}
./repeder -d -m 0 -M 4096 ${data} ${out}


