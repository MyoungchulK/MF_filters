#!/bin/bash

# load in variables
key=sub_info
key1=baseline
#key15=baseline_merge
key2=phase
key3=cw_flag
key4=cw_ratio
key5=rms
#key55=rms_merge
#key555=snr_maker
key6=reco
key7=qual_cut
sim_key=noise
st=$1
run=$2
sim_run=$3
year=2015
not_override=0
user_path=/misc/disk19/users/mkim/OMF_filter/
sim_path=/misc/disk20/users/mkim/OMF_filter/
if [ -d "$user_path" ]; then
    echo "There is ${user_path}"
else
    #echo "Switch to /data/user/"
    #user_path=/data/user/
    echo "Switch to /data/ana/ARA/"
    user_path=/data/ana/ARA/
fi
data=${sim_path}ARA0${st}/sim_${sim_key}_full/AraOut.${sim_key}_A${st}_R${run}.txt.run${sim_run}.root
rms_path=${user_path}ARA0${st}/rms_sim/rms_AraOut.${sim_key}_A${st}_R${run}.txt.run${sim_run}.h5

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key1} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key2} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key3} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key4} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key5} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/snr_maker.py ${st} ${rms_path}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key6} -s ${st} -y ${year} -d ${data} -n ${not_override}
python3 /home/mkim/analysis/MF_filters/scripts/sim_script_executor.py -k ${key7} -s ${st} -y ${year} -d ${data} -n ${not_override}

