#!/bin/bash

# load in variables
data=$1
ped=$2
out=/data/user/mkim/Rayl_test/real_norm


# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
cd /home/mkim/analysis/AraSim_lastest/AraSim/

./AraSim ${data} ${ped} ${out}






#if [ $? -ne 0 ] #error handle if something has gone wrong
#then
#	echo python3 /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode} >> /data/user/mkim/OMF_filter/problems_A${station}_R${run}.txt

#else

#    mv ${temp_dir}* ${out}

#fi 
