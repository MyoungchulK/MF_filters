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

d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/PSD/*'

d_list = sorted(glob(d_path))
d_len = len(d_list)
print(d_len)

config_tot=np.full((d_len),np.nan)
run_tot=np.full((d_len),np.nan)
good_runs_index=np.full((d_len),np.nan)
psd_tot = np.full((2048,16,d_len),np.nan)
freq = np.fft.fftfreq(2048,0.5/1e9)
print(psd_tot.shape)

aa = 0
for d in d_list:

    h5_file = h5py.File(d, 'r')

    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    config_tot[aa] = int(h5_file['Config'][:][0])
    psd_tot[:,:,aa] = h5_file['soft_psd'][:]

    try:
        a = np.where(bad_runs == run_tot[aa])[0][0]
        good_runs_index[aa] = 0
    except IndexError:
        good_runs_index[aa] = 1
    
    percen = np.round((aa/d_len)*100,2)
    print(f'{percen} %')
    aa += 1 
    
    del h5_file, percen

path = '/data/user/mkim/Sim_Data_Diff/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'psd_tot_A{Station}.h5', 'w')
hf.create_dataset('config_tot', data=config_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('good_runs_index', data=good_runs_index, compression="gzip", compression_opts=9)
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('psd_tot', data=psd_tot, compression="gzip", compression_opts=9)
hf.close()

print('Done!!')








