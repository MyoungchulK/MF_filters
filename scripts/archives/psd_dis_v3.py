import numpy as np
import os, sys
import re
from glob import glob
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import config_checker
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

#d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/PSD/*'
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/PSD_old/*'

d_list = sorted(glob(d_path))
d_len = len(d_list)
print(d_len)
run_tot=np.full((d_len),np.nan)

aa = 0
for d in d_list:

    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    aa += 1

if Station == 2:
        x_min = 1400
        x_max = 13000
if Station == 5:
        x_min = 1400
        x_max = 4000
if Station == 3:
        x_min = 400
        x_max = 8000

#run_range = np.arange(np.nanmin(run_tot),np.nanmax(run_tot)+1).astype(int)
run_range = np.arange(x_min,x_max+1).astype(int)
print(len(run_range))
config_tot=np.full((len(run_range)),-1)
good_runs_index=np.full((len(run_range)),-1)
psd_tot = np.full((2048,16,len(run_range)),np.nan)
freq = np.fft.fftfreq(2048,0.5/1e9)
print(psd_tot.shape)

aa = 0
for d in d_list:

    run_loc = np.where(run_range == int(re.sub("\D", "", d[-8:-1])))[0][0]

    h5_file = h5py.File(d, 'r')
    #config_tot[run_loc] = int(h5_file['Config'][:][2])
    config_tot[run_loc] = int(h5_file['Config'][:][0])
    psd_tot[:,:,run_loc] = h5_file['soft_psd'][:]

    try:
        a = np.where(bad_runs == run_range[run_loc])[0][0]
        good_runs_index[run_loc] = 0
    except IndexError:
        good_runs_index[run_loc] = 1
    
    percen = np.round((aa/d_len)*100,2)
    print(f'{percen} %')
    aa += 1 
    
    del h5_file, percen, run_loc

path = '/data/user/mkim/Sim_Data_Diff/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
#hf = h5py.File(f'psd_tot_A{Station}.h5', 'w')
hf = h5py.File(f'psd_tot_A{Station}_old.h5', 'w')
hf.create_dataset('config_tot', data=config_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_range', data=run_range, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('good_runs_index', data=good_runs_index, compression="gzip", compression_opts=9)
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('psd_tot', data=psd_tot, compression="gzip", compression_opts=9)
hf.close()

print('Done!!')








