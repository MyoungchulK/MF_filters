#!/bin/bash

# load in variables
key1=baseline
key2=cw_flag
key3=cw_ratio
station=$1
run=$2
blind=1
condor_run=0
not_override=1
qual_2nd=0
no_tqdm=0

source ../setup.sh

python3 /home/mkim/analysis/MF_filters/scripts/script_executor.py -k ${key1} -s ${station} -r ${run} -b ${blind} -c ${condor_run} -n ${not_override} -q ${qual_2nd} -t ${no_tqdm}
python3 /home/mkim/analysis/MF_filters/scripts/script_executor.py -k ${key2} -s ${station} -r ${run} -b ${blind} -c ${condor_run} -n ${not_override} -q ${qual_2nd} -t ${no_tqdm}
python3 /home/mkim/analysis/MF_filters/scripts/script_executor.py -k ${key3} -s ${station} -r ${run} -b ${blind} -c ${condor_run} -n ${not_override} -q ${qual_2nd} -t ${no_tqdm}
