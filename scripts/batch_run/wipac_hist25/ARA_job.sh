#!/bin/bash

# load in variables
station=$1
trig=$2
run=$3
run_w=$4
ant_c=1
smear_l=25

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

#python3 /home/mkim/analysis/MF_filters/scripts/cw_hist_livetime_set_cut_rf_sec.py ${station} ${trig} ${run} ${run_w}
#python3 /home/mkim/analysis/MF_filters/scripts/cw_hist_livetime_set_cut.py ${station} ${trig} ${run} ${run_w}
python3 /home/mkim/analysis/MF_filters/scripts/cw_hist_livetime_test_smear_combine_cut.py ${station} ${trig} ${run} ${run_w} ${ant_c} ${smear_l}

