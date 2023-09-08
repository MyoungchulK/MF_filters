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

d_path1 = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/Upper_Limit_A{Station}_R*'
d_list1, d_run_tot1, d_run_range1, d_len1 = file_sorter(d_path1)
del d_run_range1

num_pols = 2
map_d_len = 400
num_slos = 180
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
bin_width = 100

bin_edges_o = np.full((2, num_pols, map_d_len, num_slos, num_configs), np.nan, dtype = float)
bin_edges_u = np.copy(bin_edges_o)
nobs_hist = np.full((bin_width, num_pols, map_d_len, num_slos, num_configs), np.nan, dtype = int)
upl_hist = np.copy(nobs_hist)
upl_mean = np.full((num_pols, map_d_len, num_slos, num_configs), np.nan, dtype = float)
s_up_s = np.copy(upl_mean)

for r in tqdm(range(len(d_run_tot1))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list1[r], 'r')
    except OSError: 
        print(d_list1[r])
        continue
    if r == 0:
        s_ang = hf['s_ang'][:]
        bins_d = hf['bins_d'][:]
        bin_center_d = hf['bin_center_d'][:]
        bins_s = hf['bins_s'][:]
        bin_center_s = hf['bin_center_s'][:] 
        map_d_bins = hf['map_d_bins'][:]
        map_d_bin_center = hf['map_d_bin_center'][:]
        slope_m = hf['slope_m'][:]
        intercept_d = hf['intercept_d'][:]
        back_medi = hf['back_medi'][:]
        back_err = hf['back_err'][:]
        map_s_pass = hf['map_s_pass'][:]
        map_s_cut = hf['map_s_cut'][:]
    bin_edges_o[:, :, :, :, r] = hf['bin_edges_o'][:, :, :, :, r]
    bin_edges_u[:, :, :, :, r] = hf['bin_edges_u'][:, :, :, :, r]
    nobs_hist[:, :, :, :, r] = hf['nobs_hist'][:, :, :, :, r]
    upl_hist[:, :, :, :, r] = hf['upl_hist'][:, :, :, :, r]
    upl_mean[:, :, :, r] = hf['upl_mean'][:, :, :, r]
    s_up_s[:, :, :, r] = hf['s_up_s'][:, :, :, r]

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Upper_Limit_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('s_ang', data=s_ang, compression="gzip", compression_opts=9)
hf.create_dataset('bins_d', data=bins_d, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_d', data=bin_center_d, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('slope_m', data=slope_m, compression="gzip", compression_opts=9)
hf.create_dataset('intercept_d', data=intercept_d, compression="gzip", compression_opts=9)
hf.create_dataset('back_medi', data=back_medi, compression="gzip", compression_opts=9)
hf.create_dataset('back_err', data=back_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass', data=map_s_pass, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut', data=map_s_cut, compression="gzip", compression_opts=9)
hf.create_dataset('bin_edges_o', data=bin_edges_o, compression="gzip", compression_opts=9)
hf.create_dataset('bin_edges_u', data=bin_edges_u, compression="gzip", compression_opts=9)
hf.create_dataset('nobs_hist', data=nobs_hist, compression="gzip", compression_opts=9)
hf.create_dataset('upl_hist', data=upl_hist, compression="gzip", compression_opts=9)
hf.create_dataset('upl_mean', data=upl_mean, compression="gzip", compression_opts=9)
hf.create_dataset('s_up_s', data=s_up_s, compression="gzip", compression_opts=9)
hf.close()
print('done! file is in:',path+file_name, size_checker(path+file_name))






