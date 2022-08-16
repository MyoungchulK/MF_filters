#!/bin/bash

# load in variables
data=$1
station=$2
run=$3
ped_path=/misc/disk19/users/mkim/OMF_filter/ARA0${station}/ped_full/
qual=${ped_path}ped_full_qualities_A${station}_R${run}.dat
out_file=ped_full_values_A${station}_R${run}.dat
out=${ped_path}${out_file}

# run the reconstruction script
export HDF5_USE_FILE_LOCKING='FALSE'
source /cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh
source /home/mkim/analysis/MF_filters/setup.sh
cd /home/mkim/analysis/AraSoft/AraUtil/bin/

# before start the code copy data file into local path
cp -r ${data} $TMPDIR
new_data=$TMPDIR/event*.root
new_out=$TMPDIR/${out_file}
echo ${data}" is copied into "${new_data}

./repeder -d -m 0 -M 4096 -q ${qual} ${new_data} ${new_out}
#./repeder -d -m 0 -M 4096 -q ${qual} ${data} ${out}

# at the end, move the results back
mv ${new_out} ${out}
echo ${new_out}" is copied into "${out}
