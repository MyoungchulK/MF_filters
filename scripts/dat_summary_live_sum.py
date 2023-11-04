import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Live_v6_A{Station}_R*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

hf = h5py.File(d_list1[0], 'r')
year = hf['year'][:]
month = hf['month'][:]
u_bins = hf['u_bins'][:]
u_bin_center = hf['u_bin_center'][:]
y_bins = hf['y_bins'][:]
y_bin_center = hf['y_bin_center'][:]
configs = hf['configs'][:]
runs = hf['runs'][:]
livetime = hf['livetime'][:]
livetime_plot = hf['livetime_plot'][:]

run_ep = np.full((0), 0, dtype = int)
con_ep = np.copy(run_ep)
sec_ep = np.copy(run_ep)
live_ep = np.full((0), 0, dtype = float)
live_good_ep = np.copy(live_ep)
live_bad_ep = np.copy(live_ep)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    if r != 0:
        configs += hf['configs'][:]
        livetime += hf['livetime'][:]
        livetime_plots = hf['livetime_plot'][:]
        livetime_plot[:, 1:] += livetime_plots[:, 1:]

    run_ep1 = hf['run_ep'][:]
    con_ep1 = hf['con_ep'][:]
    sec_ep1 = hf['sec_ep'][:]
    live_ep1 = hf['live_ep'][:]
    live_good_ep1 = hf['live_good_ep'][:]
    live_bad_ep1 = hf['live_bad_ep'][:]
    run_ep = np.concatenate((run_ep, run_ep1))
    con_ep = np.concatenate((con_ep, con_ep1))
    sec_ep = np.concatenate((sec_ep, sec_ep1))
    live_ep = np.concatenate((live_ep, live_ep1))
    live_good_ep = np.concatenate((live_good_ep, live_good_ep1))
    live_bad_ep = np.concatenate((live_bad_ep, live_bad_ep1))

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Live_v6_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('year', data=year, compression="gzip", compression_opts=9)
hf.create_dataset('month', data=month, compression="gzip", compression_opts=9)
hf.create_dataset('u_bins', data=u_bins, compression="gzip", compression_opts=9)
hf.create_dataset('u_bin_center', data=u_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('y_bins', data=y_bins, compression="gzip", compression_opts=9)
hf.create_dataset('y_bin_center', data=y_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('livetime_plot', data=livetime_plot, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('sec_ep', data=sec_ep, compression="gzip", compression_opts=9)
hf.create_dataset('live_ep', data=live_ep, compression="gzip", compression_opts=9)
hf.create_dataset('live_good_ep', data=live_good_ep, compression="gzip", compression_opts=9)
hf.create_dataset('live_bad_ep', data=live_bad_ep, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






