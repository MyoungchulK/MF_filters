import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm
from datetime import datetime

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run
from tools.run import file_sorter
from tools.run import bin_range_maker

Station = int(sys.argv[1])

# bad runs
if Station != 5:
    bad_run_list = bad_run(Station)
    bad_sur_run_list = bad_surface_run(Station)
    bad_runs = np.append(bad_run_list, bad_sur_run_list)
    print(bad_runs.shape)
    del bad_run_list, bad_sur_run_list
else:
    bad_runs = np.array([])

# sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Off_Blk/*'
#d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Off_Blk_Cal/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
run_arr = []

# off blk hist
sig_range = np.arange(0,1000,0.25)
sig_bins, sig_bin_center = bin_range_maker(sig_range, len(sig_range))
print(sig_bin_center.shape)

sig_tot = np.full((len(sig_bin_center)), 0, dtype = int)
print(sig_tot.shape)
sig_tot_list = []

sig_fit_tot = np.copy(sig_tot)
print(sig_fit_tot.shape)
sig_fit_tot_list = []

sig_max_tot = np.copy(sig_tot)
print(sig_max_tot.shape)
sig_max_tot_list = []

sig_max_ex_tot = np.copy(sig_tot)
print(sig_max_ex_tot.shape)
sig_max_ex_tot_list = []

ratio_tot = np.full((101), 0, dtype = int)
print(ratio_tot.shape)
ratio_tot_list=[]

ant_idx = np.arange(15)
min_mu_tot = np.full((1000,15), 0, dtype = int)
min_mu_offset = min_mu_tot.shape[0]//2
print(min_mu_tot.shape)
min_mu_tot_list = []

cov_idx = np.arange(225)
min_cov_mtx_tot = np.full((1000,225), 0, dtype = int)
min_cov_mtx_offset = min_cov_mtx_tot.shape[0]//2
print(min_cov_mtx_tot.shape)
min_cov_mtx_tot_list = []

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    run_arr.append(d_run_tot[r])

    file_name = f'Off_Blk_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)

    ex_flag = hf['ex_flag'][:]
    sig_ori = hf['sig_wo_freq_arr'][:]

    sig_max = np.nanmax(sig_ori, axis=0)
    
    sig_max_copy = np.copy(sig_max)
    sig_max_copy = sig_max_copy[~np.isnan(sig_max_copy)]
    sig_max_hist = np.histogram(sig_max_copy, bins = sig_bins)[0]
    sig_max_tot += sig_max_hist
    sig_max_tot_list.append(sig_max_hist)
    del sig_max_copy

    sig_max_ex = sig_max * ex_flag
    sig_max_ex = sig_max_ex[~np.isnan(sig_max_ex)]
    sig_max_ex_hist = np.histogram(sig_max_ex, bins = sig_bins)[0]
    sig_max_ex_tot += sig_max_ex_hist
    sig_max_ex_tot_list.append(sig_max_ex_hist)
    del sig_max, sig_max_ex, ex_flag

    sig = sig_ori.flatten()
    sig = sig[~np.isnan(sig)]
    sig_hist = np.histogram(sig, bins = sig_bins)[0]
    sig_tot += sig_hist
    sig_tot_list.append(sig_hist)
    del sig_ori

    ratio = hf['ratio_wo_freq'][:]
    ratio_tot[int(ratio)] += 1
    ratio_tot_list.append(ratio)

    min_mu = hf['min_mu_wo_freq'][:]
    min_mu_int = min_mu.astype(int)
    try:
        min_mu_tot[min_mu_int+min_mu_offset, ant_idx] += 1
    except IndexError:
        pass
    min_mu_tot_list.append(min_mu)
    del min_mu_int

    min_cov_mtx = hf['min_cov_mtx_wo_freq'][:]
    min_cov_mtx_fla = min_cov_mtx.flatten()
    min_cov_mtx_int = min_cov_mtx_fla.astype(int)
    try:
        min_cov_mtx_tot[min_cov_mtx_int+min_cov_mtx_offset, cov_idx] += 1
    except IndexError:
        pass
    min_cov_mtx_tot_list.append(min_cov_mtx_fla)
    del min_cov_mtx_int

    sig_fit_ran = np.random.multivariate_normal(min_mu, min_cov_mtx, len(sig))
    sig_fit_inv = np.linalg.inv(min_cov_mtx)
    sig_fit_arr = np.sqrt(np.einsum('...k,kl,...l->...', sig_fit_ran-min_mu, sig_fit_inv, sig_fit_ran-min_mu)) 
    sig_fit_hist = np.histogram(sig_fit_arr, bins = sig_bins)[0]   
    sig_fit_tot += sig_fit_hist
    sig_fit_tot_list.append(sig_fit_hist)
 
    del sig, min_cov_mtx, sig_fit_ran, sig_fit_inv, sig_fit_arr

    del hf, file_name

run_arr = np.asarray(run_arr)
config_arr = np.asarray(config_arr)
print(run_arr.shape)
print(config_arr.shape)

print(sig_tot.shape)
print(sig_fit_tot.shape)
print(sig_max_tot.shape)
print(sig_max_ex_tot.shape)
print(ratio_tot.shape)
print(min_mu_tot.shape)
print(min_cov_mtx_tot.shape)

sig_tot_list = np.transpose(np.asarray(sig_tot_list),(1,0))
print(sig_tot_list.shape)
sig_fit_tot_list = np.transpose(np.asarray(sig_fit_tot_list),(1,0))
print(sig_fit_tot_list.shape)
sig_max_tot_list = np.transpose(np.asarray(sig_max_tot_list),(1,0))
print(sig_max_tot_list.shape)
sig_max_ex_tot_list = np.transpose(np.asarray(sig_max_ex_tot_list),(1,0))
print(sig_max_ex_tot_list.shape)
ratio_tot_list = np.asarray(ratio_tot_list)
print(ratio_tot_list.shape)
min_mu_tot_list = np.transpose(np.asarray(min_mu_tot_list),(1,0))
print(min_mu_tot_list.shape)
min_cov_mtx_tot_list = np.transpose(np.asarray(min_cov_mtx_tot_list),(1,0))
print(min_cov_mtx_tot_list.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Off_Blk_MDG_A{Station}.h5'
#file_name = f'Off_Blk_MDG_Cal_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)

hf.create_dataset('sig_range', data=sig_range, compression="gzip", compression_opts=9)
hf.create_dataset('sig_bins', data=sig_bins, compression="gzip", compression_opts=9)
hf.create_dataset('sig_bin_center', data=sig_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('sig_tot', data=sig_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_tot_list', data=sig_tot_list, compression="gzip", compression_opts=9)

hf.create_dataset('sig_fit_tot', data=sig_fit_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_fit_tot_list', data=sig_fit_tot_list, compression="gzip", compression_opts=9)

hf.create_dataset('sig_max_tot', data=sig_max_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_max_tot_list', data=sig_max_tot_list, compression="gzip", compression_opts=9)

hf.create_dataset('sig_max_ex_tot', data=sig_max_ex_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_max_ex_tot_list', data=sig_max_ex_tot_list, compression="gzip", compression_opts=9)

hf.create_dataset('ratio_tot', data=ratio_tot, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_tot_list', data=ratio_tot_list, compression="gzip", compression_opts=9)

hf.create_dataset('min_mu_tot', data=min_mu_tot, compression="gzip", compression_opts=9)
hf.create_dataset('min_mu_tot_list', data=min_mu_tot_list, compression="gzip", compression_opts=9)

hf.create_dataset('min_cov_mtx_tot', data=min_cov_mtx_tot, compression="gzip", compression_opts=9)
hf.create_dataset('min_cov_mtx_tot_list', data=min_cov_mtx_tot_list, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')




















