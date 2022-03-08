#!/bin/bash

# load in variables
data=$1
station=$2
run=$3
out=/data/user/mkim/OMF_filter/ARA0${station}/ped/ped_values_A${station}_R${run}.dat
qual=/data/user/mkim/OMF_filter/ARA0${station}/ped/ped_qualities_A${station}_R${run}.dat

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/AraSoft/for_local_araroot.sh
cd /home/mkim/analysis/AraSoft/AraUtil/bin/

./repeder -d -m 0 -M 4096 -q ${qual} ${data} ${out}

