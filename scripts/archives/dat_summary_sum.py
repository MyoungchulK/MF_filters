import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Data_Summary_Signal_A*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
livetime = np.full((d_len, 3), 0, dtype = float)
nan_counts = np.full((d_len), 0, dtype = int)

z_bins = np.linspace(-90, 90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
a_bins = np.linspace(-180, 180, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
c_bins = np.linspace(0, 1.2, 1200 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bin_len = len(a_bin_center)
c_bin_len = len(c_bin_center)

map_r_z = np.full((d_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int) # run, z or a, trig, pol, rad, sol
map_r_z_cut = np.copy(map_r_z)
map_r_z_rf = np.full((d_len, z_bin_len, 3, 2), 0, dtype = int)
map_r_a = np.full((d_len, a_bin_len, 3, 2, 2, 2), 0, dtype = int)
map_r_a_cut = np.copy(map_r_a)
map_r_a_rf = np.full((d_len, a_bin_len, 3, 2), 0, dtype = int)
map_r_c = np.full((d_len, c_bin_len, 3, 2, 2, 2), 0, dtype = int) # run, z or a or c, trig, pol, rad, sol
map_r_c_cut = np.copy(map_r_c)
map_r_c_rf = np.full((d_len, c_bin_len, 3, 2), 0, dtype = int)
map_az = np.full((num_configs, a_bin_len, z_bin_len, 3, 2, 2, 2), 0, dtype = int) # a, z, trig, pol, rad, sol, config
map_az_cut = np.copy(map_az)
map_az_rf = np.full((num_configs, a_bin_len, z_bin_len, 3, 2), 0, dtype = int)
del bad_runs, z_bin_len, a_bin_len, c_bin_len

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    configs += hf['configs'][:]
    livetime += hf['livetime'][:]
    nan_counts += hf['nan_counts'][:]

    map_r_z += hf['map_r_z'][:]
    map_r_z_cut += hf['map_r_z_cut'][:]
    map_r_z_rf += hf['map_r_z_rf'][:]
    map_r_a += hf['map_r_a'][:]
    map_r_a_cut += hf['map_r_a_cut'][:]
    map_r_a_rf += hf['map_r_a_rf'][:]
    map_r_c += hf['map_r_c'][:]
    map_r_c_cut += hf['map_r_c_cut'][:]
    map_r_c_rf += hf['map_r_c_rf'][:]
    map_az += hf['map_az'][:]
    map_az_cut += hf['map_az_cut'][:]
    map_az_rf += hf['map_az_rf'][:]
    del hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_Signal_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('livetime', data=livetime, compression="gzip", compression_opts=9)
hf.create_dataset('nan_counts', data=nan_counts, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z', data=map_r_z, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_cut', data=map_r_z_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_z_rf', data=map_r_z_rf, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a', data=map_r_a, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_cut', data=map_r_a_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_a_rf', data=map_r_a_rf, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c', data=map_r_c, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_cut', data=map_r_c_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_r_c_rf', data=map_r_c_rf, compression="gzip", compression_opts=9)
hf.create_dataset('map_az', data=map_az, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_cut', data=map_az_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_rf', data=map_az_rf, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






