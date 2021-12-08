import numpy as np
import os, sys
import re
from glob import glob
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run

Station = int(sys.argv[1])
if Station != 5:
    bad_run_list = bad_run(Station)
    bad_sur_run_list = bad_surface_run(Station)
    bad_runs = np.append(bad_run_list, bad_sur_run_list)
    print(bad_runs.shape)
    del bad_run_list, bad_sur_run_list
else:
    bad_runs = np.array([])

d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Rayl/*'

d_list = sorted(glob(d_path))
d_len = len(d_list)
print(d_len)
run_tot=np.full((d_len),np.nan)

aa = 0
for d in d_list:

    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    aa += 1

run_range = np.arange(np.nanmin(run_tot),np.nanmax(run_tot)+1).astype(int)
print(len(run_range))
good_runs_index=np.full((len(run_range)),np.nan)
loc_w_sc_all = np.full((1024,16,len(run_range)), np.nan)
scale_w_sc_all = np.full((1024,16,len(run_range)), np.nan)
loc_wo_sc_all = np.full((1024,16,len(run_range)), np.nan)
scale_wo_sc_all = np.full((1024,16,len(run_range)), np.nan)

aa = 0
for d in d_list:

    run_loc = np.where(run_range == int(re.sub("\D", "", d[-8:-1])))[0][0]

    h5_file = h5py.File(d, 'r')
    if aa == 0:
        freq = h5_file['freq'][:]
    else:
        pass

    loc_w_sc_all[:,:,run_loc] = h5_file['loc_w_sc'][:]
    scale_w_sc_all[:,:,run_loc] = h5_file['scale_w_sc'][:]
    loc_wo_sc_all[:,:,run_loc] = h5_file['loc_wo_sc'][:]
    scale_wo_sc_all[:,:,run_loc] = h5_file['scale_wo_sc'][:]

    try:
        a = np.where(bad_runs == run_range[run_loc])[0][0]
        good_runs_index[run_loc] = np.nan
    except IndexError:
        good_runs_index[run_loc] = 1.
    
    percen = np.round((aa/d_len)*100,2)
    print(f'{percen} %')
    aa += 1 
    
    del h5_file, percen, run_loc

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Rayl_tot/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Rayleigh_Fit_A{Station}_tot.h5', 'w')
hf.create_dataset('run_range', data=run_range, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('good_runs_index', data=good_runs_index, compression="gzip", compression_opts=9)
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('loc_w_sc', data=loc_w_sc_all, compression="gzip", compression_opts=9)
hf.create_dataset('scale_w_sc', data=scale_w_sc_all, compression="gzip", compression_opts=9)
hf.create_dataset('loc_wo_sc', data=loc_wo_sc_all, compression="gzip", compression_opts=9)
hf.create_dataset('scale_wo_sc', data=scale_wo_sc_all, compression="gzip", compression_opts=9)
hf.close()

print('Done!!')








