#!/bin/bash

# load in variables
data=$1
ped=$2
out=$3
#station=$4
#run=$5
#mode=$6

#temp_dir=/scratch/mkim/MF_filters_out/
#script_dir=/home/mkim/analysis/MF_filters/scripts/

# run the reconstruction script
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/
#cd ${script_dir}

#ulimit -s 131072; python3 mf_filter.py ${data} ${ped} ${station} ${run} ${temp_dir} ${mode}
#ulimit -s 131072; python3 /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode}
python3 /home/mkim/analysis/MF_filters/scripts/mf_filter_coherent.py ${data} ${ped} ${out}

#if [ $? -ne 0 ] #error handle if something has gone wrong
#then
#	echo python3 /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode} >> /data/user/mkim/OMF_filter/problems_A${station}_R${run}.txt

#else

#    mv ${temp_dir}* ${out}

#fi 
