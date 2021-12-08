#!/bin/bash

# load in variables
data=$1
ped=$2
station=$3
#out=/data/user/mkim/OMF_filter/ARA0${station}/Rayl/
#out=/data/user/mkim/OMF_filter/ARA0${station}/Info/
#out=/data/user/mkim/OMF_filter/ARA0${station}/Offset/
#out=/data/user/mkim/OMF_filter/ARA0${station}/Medi_Tilt_repeder/
out=/data/user/mkim/OMF_filter/ARA0${station}/RMS_Peak_Old/
#out=/data/user/mkim/OMF_filter/ARA0${station}/Medi_Tilt_kJustPed_repeder/
#out=/data/user/mkim/OMF_filter/ARA0${station}/Medi_Tilt_kLatestCalib_repeder/
#out=/home/mkim/test_new/

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/AraSoft/for_local_araroot.sh
cd /home/mkim/analysis/MF_filters/scripts/

#ulimit -s 131072; python3 /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode}
#python3 /home/mkim/analysis/MF_filters/scripts/info_collector.py ${data} ${ped} ${out}
#python3 /home/mkim/analysis/MF_filters/scripts/wf_collector.py ${data} ${ped} ${out}
#python3 /home/mkim/analysis/MF_filters/scripts/rayleigh_params.py ${data} ${ped} ${out}
#python3 /home/mkim/analysis/MF_filters/scripts/offset_collector.py ${data} ${ped} ${out}
#python3 /home/mkim/analysis/MF_filters/scripts/qual_debug_collector.py ${data} ${ped} ${out}
python3 /home/mkim/analysis/MF_filters/scripts/rms_peak_collector.py ${data} ${ped} ${out}






#if [ $? -ne 0 ] #error handle if something has gone wrong
#then
#	echo python3 /home/mkim/analysis/MF_filters/scripts/mf_filter.py ${data} ${ped} ${station} ${run} ${out} ${mode} >> /data/user/mkim/OMF_filter/problems_A${station}_R${run}.txt

#else

#    mv ${temp_dir}* ${out}

#fi 
