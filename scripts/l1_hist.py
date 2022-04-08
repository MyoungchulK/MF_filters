import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/l1_full/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

l1_range = np.arange(0,100000,100)
l1_bins = np.linspace(0,100000,1000+1)
l1_bin_center = (l1_bins[1:] + l1_bins[:-1]) / 2
min_range = np.arange(0, 360)
min_bins = np.linspace(0, 360, 360 + 1)
min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2

#l1_rate_hist = []
#l1_thres_hist = []
#l1_rate_cut_hist = []
#l1_thres_cut_hist = []

#l1_rate_hist2d = np.full((len(min_bin_center), len(l1_bin_center), 16), 0, dtype = int)
l1_thres_hist2d = np.full((len(min_bin_center), len(l1_bin_center), 16), 0, dtype = int)
#l1_rate_cut_hist2d = np.copy(l1_rate_hist2d)
l1_thres_cut_hist2d = np.copy(l1_thres_hist2d)

#l1_rate_hist2d_max = []
l1_thres_hist2d_max = []
#l1_rate_cut_hist2d_max = []
l1_thres_cut_hist2d_max = []
#l1_rate_hist2d_min = []
#l1_thres_hist2d_min = []
#l1_rate_cut_hist2d_min = []
#l1_thres_cut_hist2d_min = []

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    hf = h5py.File(d_list[r], 'r')
    config = hf['config'][2]
    config_arr.append(config)
    run_arr.append(d_run_tot[r])
    
    trig_ch = hf['trig_ch'][:]
    #l1_rate = hf['l1_rate_hist'][:]
    #l1_rate = l1_rate[:, trig_ch]
    #l1_thres = hf['l1_thres_hist'][:]
    #l1_thres = l1_thres[:, trig_ch]
    #l1_rate_hist.append(l1_rate)
    #l1_thres_hist.append(l1_thres)

    #l1_rate_2d = hf['l1_rate_hist2d'][:]
    #l1_rate_2d = l1_rate_2d[:, :, trig_ch]
    l1_thres_2d = hf['l1_thres_hist2d'][:]
    l1_thres_2d = l1_thres_2d[:, :, trig_ch]
    #l1_rate_hist2d += l1_rate_2d
    l1_thres_hist2d += l1_thres_2d
    del l1_thres_2d

    #l1_rate_max = hf['l1_rate_hist2d_max'][:]
    #l1_rate_max = l1_rate_max[:, trig_ch]
    l1_thres_max = hf['l1_thres_hist2d_max'][:]
    l1_thres_max = l1_thres_max[:, trig_ch]
    #l1_rate_hist2d_max.append(l1_rate_max)
    l1_thres_hist2d_max.append(l1_thres_max)

    #l1_rate_min = hf['l1_rate_hist2d_min'][:]
    #l1_rate_min = l1_rate_min[:, trig_ch]
    #l1_thres_min = hf['l1_thres_hist2d_min'][:]
    #l1_thres_min = l1_thres_min[:, trig_ch]
    #l1_rate_hist2d_min.append(l1_rate_min)
    #l1_thres_hist2d_min.append(l1_thres_min)

    if d_run_tot[r] in bad_runs:
        #print('bad run:', d_list[r], d_run_tot[r])
        continue

    config_arr_cut.append(config)
    run_arr_cut.append(d_run_tot[r])

    #l1_rate_cut = hf['l1_rate_cut_hist'][:]
    #l1_rate_cut = l1_rate_cut[:, trig_ch]
    #l1_thres_cut = hf['l1_thres_cut_hist'][:]
    #l1_thres_cut = l1_thres_cut[:, trig_ch]
    #l1_rate_cut_hist.append(l1_rate_cut)
    #l1_thres_cut_hist.append(l1_thres_cut)

    #l1_rate_cut_2d = hf['l1_rate_cut_hist2d'][:]
    #l1_rate_cut_2d = l1_rate_cut_2d[:, :, trig_ch]
    l1_thres_cut_2d = hf['l1_thres_cut_hist2d'][:]
    l1_thres_cut_2d = l1_thres_cut_2d[:, :, trig_ch]
    #l1_rate_cut_hist2d += l1_rate_cut_2d
    l1_thres_cut_hist2d += l1_thres_cut_2d
    del l1_thres_cut_2d

    #l1_rate_cut_max = hf['l1_rate_cut_hist2d_max'][:]
    #l1_rate_cut_max = l1_rate_cut_max[:, trig_ch]
    l1_thres_cut_max = hf['l1_thres_cut_hist2d_max'][:]
    l1_thres_cut_max = l1_thres_cut_max[:, trig_ch]
    #l1_rate_cut_hist2d_max.append(l1_rate_cut_max)
    l1_thres_cut_hist2d_max.append(l1_thres_cut_max)

    #l1_rate_cut_min = hf['l1_rate_cut_hist2d_min'][:]
    #l1_rate_cut_min = l1_rate_cut_min[:, trig_ch]
    #l1_thres_cut_min = hf['l1_thres_cut_hist2d_min'][:]
    #l1_thres_cut_min = l1_thres_cut_min[:, trig_ch]
    #l1_rate_cut_hist2d_min.append(l1_rate_cut_min)
    #l1_thres_cut_hist2d_min.append(l1_thres_cut_min)
    del hf, trig_ch

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'L1_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('l1_range', data=l1_range, compression="gzip", compression_opts=9)
hf.create_dataset('l1_bins', data=l1_bins, compression="gzip", compression_opts=9)
hf.create_dataset('l1_bin_center', data=l1_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('min_range', data=min_range, compression="gzip", compression_opts=9)
hf.create_dataset('min_bins', data=min_bins, compression="gzip", compression_opts=9)
hf.create_dataset('min_bin_center', data=min_bin_center, compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_hist2d', data=l1_rate_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_thres_hist2d', data=l1_thres_hist2d, compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_cut_hist2d', data=l1_rate_cut_hist2d, compression="gzip", compression_opts=9)
hf.create_dataset('l1_thres_cut_hist2d', data=l1_thres_cut_hist2d, compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_hist', data=np.asarray(l1_rate_hist), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_thres_hist', data=np.asarray(l1_thres_hist), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_cut_hist', data=np.asarray(l1_rate_cut_hist), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_thres_cut_hist', data=np.asarray(l1_thres_cut_hist), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_hist2d_max', data=np.asarray(l1_rate_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('l1_thres_hist2d_max', data=np.asarray(l1_thres_hist2d_max), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_cut_hist2d_max', data=np.asarray(l1_rate_cut_hist2d_max), compression="gzip", compression_opts=9)
hf.create_dataset('l1_thres_cut_hist2d_max', data=np.asarray(l1_thres_cut_hist2d_max), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_hist2d_min', data=np.asarray(l1_rate_hist2d_min), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_thres_hist2d_min', data=np.asarray(l1_thres_hist2d_min), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_rate_cut_hist2d_min', data=np.asarray(l1_rate_cut_hist2d_min), compression="gzip", compression_opts=9)
#hf.create_dataset('l1_thres_cut_hist2d_min', data=np.asarray(l1_thres_cut_hist2d_min), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






