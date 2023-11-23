#!/bin/bash

# load in variables
station=$1
pol=$2
slo=$3
frac=$4

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /home/mkim/.bashrc
source /cvmfs/ara.opensciencegrid.org/trunk/RHEL_8_x86_64/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/MF_filters/scripts/

python3 /home/mkim/analysis/MF_filters/scripts/back_est_gof_ell.py ${station} ${pol} ${slo} ${frac}

