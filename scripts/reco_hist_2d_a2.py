import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco_a2/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

z_bins = np.linspace(0, 180, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)
a_bins = np.linspace(0, 360, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
a_bin_len = len(a_bin_center)
c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
c_bin_len = len(c_bin_center)

map_az_cut = np.full((a_bin_len, z_bin_len, 2, num_configs), 0, dtype = int) # a, z, pol, config
map_ac_cut = np.full((a_bin_len, c_bin_len, 2, num_configs), 0, dtype = int)
map_zc_cut = np.full((z_bin_len, c_bin_len, 2, num_configs), 0, dtype = int)
del z_bin_len, a_bin_len, c_bin_len

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    bad_run = d_run_tot[r] in bad_runs
    if bad_run: continue

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue

    g_idx = int(hf['config'][2] - 1)
    coord = hf['coord'][:] # pol, thephi
    coef = hf['coef'][:]
    del hf

    for pol in range(2):
        map_az_cut[:, :, pol, g_idx] += np.histogram2d(coord[pol, 1], coord[pol, 0], bins = (a_bins, z_bins))[0].astype(int)
        map_ac_cut[:, :, pol, g_idx] += np.histogram2d(coord[pol, 1], coef[pol], bins = (a_bins, c_bins))[0].astype(int)
        map_zc_cut[:, :, pol, g_idx] += np.histogram2d(coord[pol, 0], coef[pol], bins = (z_bins, c_bins))[0].astype(int)
    del g_idx, bad_run, coef, coord

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Map_New_2d_a2_v2_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_az_cut', data=map_az_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_ac_cut', data=map_ac_cut, compression="gzip", compression_opts=9)
hf.create_dataset('map_zc_cut', data=map_zc_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






