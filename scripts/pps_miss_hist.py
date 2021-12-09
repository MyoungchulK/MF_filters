import numpy as np
import os, sys
import re
from glob import glob
import h5py
from tqdm import tqdm

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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/PPS_Miss/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = np.array([], dtype = int)
run_arr = np.copy(config_arr)
evt_num = np.copy(config_arr)
pps_miss = np.copy(config_arr)
raw_std = []
bp_std = []

evt_range = np.arange(1000)
evt_bins, evt_bin_center = bin_range_maker(evt_range, len(evt_range))
evt_pps_before = np.full((len(evt_bin_center)), 0, dtype = int)
evt_pps_after = np.copy(evt_pps_before)

std_range = np.arange(200)
std_bins, std_bin_center = bin_range_maker(std_range, len(std_range))
std_pps_before = np.full((len(std_bin_center), 16), 0, dtype = int)
std_pps_after = np.copy(std_pps_before)
bp_std_pps_before = np.copy(std_pps_before)
bp_std_pps_after = np.copy(std_pps_before)

evt_limit = 100

for r in tqdm(range(len(d_run_tot))):
  if r > len(d_run_tot)-100:
    if d_run_tot[r] in bad_runs:
        #print('bad run:',d_list[r],d_run_tot[r])
        continue

    hf = h5py.File(d_list[r], 'r')

    config_arr = np.append(config_arr, hf['config'][2])
    run_arr = np.append(run_arr, d_run_tot[r])
    evt_arr = hf['rf_soft_evt_num'][:evt_limit - 1]
    evt_num = np.append(evt_num, evt_arr)

    pps_num = hf['rf_soft_pps_number'][:evt_limit]
    unix_t = hf['rf_soft_unix_time'][:evt_limit]
    diff = np.diff(pps_num) - np.diff(unix_t)
    pps_miss = np.append(pps_miss, diff)

    raw_arr = hf['wf_std'][:,:evt_limit - 1]
    bp_arr = hf['bp_wf_std'][:,:evt_limit - 1]

    raw_std.append(raw_arr)
    bp_std.append(bp_arr)

    pps_cut = np.where(diff > 1)[0]
    if len(pps_cut) != 0:
        evt_pps_before += np.histogram(evt_arr[:pps_cut[-1]+1], bins = evt_bins)[0]
        evt_pps_after += np.histogram(evt_arr[pps_cut[-1]+1:], bins = evt_bins)[0]

        for a in range(16):
            std_pps_before[:,a] += np.histogram(raw_arr[a,:pps_cut[-1]+1], bins = std_bins)[0]
            std_pps_after[:,a] += np.histogram(raw_arr[a,pps_cut[-1]+1:], bins = std_bins)[0]

            bp_std_pps_before[:,a] += np.histogram(bp_arr[a,:pps_cut[-1]+1], bins = std_bins)[0]
            bp_std_pps_after[:,a] += np.histogram(bp_arr[a,pps_cut[-1]+1:], bins = std_bins)[0]
    else:
        evt_pps_after += np.histogram(evt_arr, bins = evt_bins)[0]
        for a in range(16):
            std_pps_after[:,a] += np.histogram(raw_arr[a], bins = std_bins)[0]
            bp_std_pps_after[:,a] += np.histogram(bp_arr[a], bins = std_bins)[0]    
    del hf, pps_num, unix_t

raw_std = np.asarray(raw_std)
bp_std = np.asarray(bp_std)

print(run_arr.shape)
print(config_arr.shape)
print(evt_num.shape)
print(pps_miss.shape)
print(raw_std.shape)
print(bp_std.shape)
print(evt_pps_before.shape)
print(evt_pps_after.shape)
print(std_pps_before.shape)
print(std_pps_after.shape)
print(bp_std_pps_before.shape)
print(bp_std_pps_after.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'PPS_Miss_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_arr', data=run_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
hf.create_dataset('pps_miss', data=pps_miss, compression="gzip", compression_opts=9)
hf.create_dataset('raw_std', data=raw_std, compression="gzip", compression_opts=9)
hf.create_dataset('bp_std', data=bp_std, compression="gzip", compression_opts=9)

hf.create_dataset('evt_range', data=evt_range, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bins', data=evt_bins, compression="gzip", compression_opts=9)
hf.create_dataset('evt_bin_center', data=evt_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('evt_pps_before', data=evt_pps_before, compression="gzip", compression_opts=9)
hf.create_dataset('evt_pps_after', data=evt_pps_after, compression="gzip", compression_opts=9)

hf.create_dataset('std_range', data=std_range, compression="gzip", compression_opts=9)
hf.create_dataset('std_bins', data=std_bins, compression="gzip", compression_opts=9)
hf.create_dataset('std_bin_center', data=std_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('std_pps_before', data=std_pps_before, compression="gzip", compression_opts=9)
hf.create_dataset('std_pps_after', data=std_pps_after, compression="gzip", compression_opts=9)
hf.create_dataset('bp_std_pps_before', data=bp_std_pps_before, compression="gzip", compression_opts=9)
hf.create_dataset('bp_std_pps_after', data=bp_std_pps_after, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')

from matplotlib import pyplot as plt

p_path = f'{path}PPS_Miss_Sctt/'
if not os.path.exists(p_path):
    os.makedirs(p_path)
os.chdir(p_path)
print(p_path)

for a in tqdm(range(16)):
  #if a == ant:    

    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    ax.set_title(f'First Few Evts & PPS Miss Correlation A{Station} Ch{a}', y=1.06,fontsize=20)
    ax.set_ylabel(r'Event #', labelpad = 20,fontsize=20)
    ax.set_xlabel(r'PPS Miss', labelpad = 20,fontsize=20)
    ax.set_zlabel(r'RMS [ $mV$ ]', labelpad = 10,fontsize=20)
    ax.grid()
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)
    ax.view_init(30, 130)
    #ax.view_init(90, 0)
    
    std_arr = raw_std[:,a,:].flatten()

    sctt = ax.scatter3D(pps_miss, evt_num, std_arr, c = std_arr, cmap='jet', s = 100, alpha = 0.8, marker='o')  
    cbar1 = plt.colorbar(sctt, ax=ax, shrink = 0.5, aspect = 10, pad = 0.1,location = 'right')
    cbar1.ax.tick_params(axis='y', labelsize=15)
    cbar1.ax.set_ylabel(r'RMS [ $mV$ ]', fontsize=15)
    
    fig.savefig(f'{p_path}First_Few_Evts_PPS_Miss_Correlation_A{Station}_Ch{a}_3d.png',bbox_inches='tight')
    #plt.show()
    plt.close()















