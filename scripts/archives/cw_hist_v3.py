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
dtype = '_all_002'
#dtype = '_wb_002'

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run(use_qual = True)

# sort
#d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw{dtype}/*'
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
config_arr = []
config_arr_cut = []
run_arr = []
run_arr_cut = []

hf = h5py.File(d_list[0], 'r')
freq_range = hf['freq_range'][:]
freq_bins = hf['freq_bins'][:]
freq_bin_center = hf['freq_bin_center'][:]
amp_range = hf['amp_range'][:]
amp_bins = hf['amp_bins'][:]
amp_bin_center = hf['amp_bin_center'][:]
power_range = hf['power_range'][:]
power_bins = hf['power_bins'][:]
power_bin_center = hf['power_bin_center'][:]
ratio_range = hf['ratio_range'][:]
ratio_bins = hf['ratio_bins'][:]
ratio_bin_center = hf['ratio_bin_center'][:]
amp_err_range = hf['amp_err_range'][:]
amp_err_bins = hf['amp_err_bins'][:]
amp_err_bin_center = hf['amp_err_bin_center'][:]
phase_err_range = hf['phase_err_range'][:]
phase_err_bins = hf['phase_err_bins'][:]
phase_err_bin_center = hf['phase_err_bin_center'][:]
del hf

freq_len = len(freq_bins) - 1
amp_len = len(amp_bins) - 1
ratio_len = len(ratio_bins) - 1
amp_err_len = len(amp_err_bins) - 1
phase_err_len = len(phase_err_bins) - 1

fft_rf_map = np.full((freq_len, amp_len, 16), 0, dtype = float)
fft_rf_cut_map = np.copy(fft_rf_map)
sub_rf_map = np.copy(fft_rf_map)
sub_rf_cut_map = np.copy(fft_rf_map)
amp_ratio_rf_map = np.full((amp_len, ratio_len, 16), 0, dtype = float)
amp_ratio_rf_cut_map = np.copy(amp_ratio_rf_map)
amp_err_ratio_rf_map = np.full((amp_err_len, ratio_len, 16), 0, dtype = float)
amp_err_ratio_rf_cut_map = np.copy(amp_err_ratio_rf_map)
phase_err_ratio_rf_map = np.full((phase_err_len, ratio_len, 16), 0, dtype = float)
phase_err_ratio_rf_cut_map = np.copy(phase_err_ratio_rf_map)
amp_err_phase_err_rf_map = np.full((amp_err_len, phase_err_len, 16), 0, dtype = float)
amp_err_phase_err_rf_cut_map = np.copy(amp_err_phase_err_rf_map)
sub_rf_map_w = np.full((freq_len, amp_len, 16), 0, dtype = float)
sub_rf_cut_map_w = np.copy(sub_rf_map_w)
amp_ratio_rf_map_w = np.full((amp_len, ratio_len, 16), 0, dtype = float)
amp_ratio_rf_cut_map_w = np.copy(amp_ratio_rf_map_w)
amp_err_ratio_rf_map_w = np.full((amp_err_len, ratio_len, 16), 0, dtype = float)
amp_err_ratio_rf_cut_map_w = np.copy(amp_err_ratio_rf_map_w)
phase_err_ratio_rf_map_w = np.full((phase_err_len, ratio_len, 16), 0, dtype = float)
phase_err_ratio_rf_cut_map_w = np.copy(phase_err_ratio_rf_map_w)
amp_err_phase_err_rf_map_w = np.full((amp_err_len, phase_err_len, 16), 0, dtype = float)
amp_err_phase_err_rf_cut_map_w = np.copy(amp_err_phase_err_rf_map_w)

power_rf_hist = []
power_rf_cut_hist = []
ratio_rf_hist = []
ratio_rf_cut_hist = []
amp_err_rf_hist = []
amp_err_rf_cut_hist = []
phase_err_rf_hist = []
phase_err_rf_cut_hist = []
power_rf_hist_w = []
power_rf_cut_hist_w = []
ratio_rf_hist_w = []
ratio_rf_cut_hist_w = []
amp_err_rf_hist_w = []
amp_err_rf_cut_hist_w = []
phase_err_rf_hist_w = []
phase_err_rf_cut_hist_w = []

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
    #if d_run_tot[r] == 4434 or d_run_tot[r] == 12237:
    #    print(d_run_tot[r])
    #    continue

    hf = h5py.File(d_list[r], 'r')

    try:    
        config = hf['config'][2]
        config_arr.append(config)
        run_arr.append(d_run_tot[r])
        
        fft_rf_map += hf['fft_rf_map'][:]
        sub_rf_map += hf['sub_rf_map'][:]
        amp_ratio_rf_map += hf['amp_ratio_rf_map'][:]
        amp_err_ratio_rf_map += hf['amp_err_ratio_rf_map'][:]
        phase_err_ratio_rf_map += hf['phase_err_ratio_rf_map'][:]
        amp_err_phase_err_rf_map += hf['amp_err_phase_err_rf_map'][:]
        sub_rf_map_w += hf['sub_rf_map_w'][:]
        amp_ratio_rf_map_w += hf['amp_ratio_rf_map_w'][:]
        amp_err_ratio_rf_map_w += hf['amp_err_ratio_rf_map_w'][:]
        phase_err_ratio_rf_map_w += hf['phase_err_ratio_rf_map_w'][:]
        amp_err_phase_err_rf_map_w += hf['amp_err_phase_err_rf_map_w'][:]

        power = hf['power_rf_hist'][:]
        ratio = hf['ratio_rf_hist'][:]
        amp_err = hf['amp_err_rf_hist'][:]
        phase_err = hf['phase_err_rf_hist'][:]
        power_rf_hist.append(power)
        ratio_rf_hist.append(ratio)
        amp_err_rf_hist.append(amp_err)
        phase_err_rf_hist.append(phase_err)
        power_w = hf['power_rf_hist_w'][:]
        ratio_w = hf['ratio_rf_hist_w'][:]
        amp_err_w = hf['amp_err_rf_hist_w'][:]
        phase_err_w = hf['phase_err_rf_hist_w'][:]
        power_rf_hist_w.append(power_w)
        ratio_rf_hist_w.append(ratio_w)
        amp_err_rf_hist_w.append(amp_err_w)
        phase_err_rf_hist_w.append(phase_err_w)

        if d_run_tot[r] in bad_runs:
            #print('bad run:', d_list[r], d_run_tot[r])
            continue

        config_arr_cut.append(config)
        run_arr_cut.append(d_run_tot[r])
        
        fft_rf_cut_map += hf['fft_rf_cut_map'][:]
        sub_rf_cut_map += hf['sub_rf_cut_map'][:]
        amp_ratio_rf_cut_map += hf['amp_ratio_rf_cut_map'][:]
        amp_err_ratio_rf_cut_map += hf['amp_err_ratio_rf_cut_map'][:]
        phase_err_ratio_rf_cut_map += hf['phase_err_ratio_rf_cut_map'][:]
        amp_err_phase_err_rf_cut_map += hf['amp_err_phase_err_rf_cut_map'][:]
        sub_rf_cut_map_w += hf['sub_rf_cut_map_w'][:]
        amp_ratio_rf_cut_map_w += hf['amp_ratio_rf_cut_map_w'][:]
        amp_err_ratio_rf_cut_map_w += hf['amp_err_ratio_rf_cut_map_w'][:]
        phase_err_ratio_rf_cut_map_w += hf['phase_err_ratio_rf_cut_map_w'][:]
        amp_err_phase_err_rf_cut_map_w += hf['amp_err_phase_err_rf_cut_map_w'][:]

        power_cut = hf['power_rf_cut_hist'][:]
        ratio_cut = hf['ratio_rf_cut_hist'][:]
        amp_err_cut = hf['amp_err_rf_cut_hist'][:]
        phase_err_cut = hf['phase_err_rf_cut_hist'][:]
        power_rf_cut_hist.append(power_cut)
        ratio_rf_cut_hist.append(ratio_cut)
        amp_err_rf_cut_hist.append(amp_err_cut)
        phase_err_rf_cut_hist.append(phase_err_cut)
        power_cut_w = hf['power_rf_cut_hist_w'][:]
        ratio_cut_w = hf['ratio_rf_cut_hist_w'][:]
        amp_err_cut_w = hf['amp_err_rf_cut_hist_w'][:]
        phase_err_cut_w = hf['phase_err_rf_cut_hist_w'][:]
        power_rf_cut_hist_w.append(power_cut_w)
        ratio_rf_cut_hist_w.append(ratio_cut_w)
        amp_err_rf_cut_hist_w.append(amp_err_cut_w)
        phase_err_rf_cut_hist_w.append(phase_err_cut_w)
        del hf
    except KeyError:
        continue

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'CW_A{Station}{dtype}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config_arr', data=np.asarray(config_arr), compression="gzip", compression_opts=9)
hf.create_dataset('config_arr_cut', data=np.asarray(config_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr', data=np.asarray(run_arr), compression="gzip", compression_opts=9)
hf.create_dataset('run_arr_cut', data=np.asarray(run_arr_cut), compression="gzip", compression_opts=9)
hf.create_dataset('freq_range', data=freq_range, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_range', data=amp_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bins', data=amp_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_bin_center', data=amp_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_range', data=amp_err_range, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bins', data=amp_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_bin_center', data=amp_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_range', data=phase_err_range, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_bins', data=phase_err_bins, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_bin_center', data=phase_err_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_range', data=ratio_range, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bins', data=ratio_bins, compression="gzip", compression_opts=9)
hf.create_dataset('ratio_bin_center', data=ratio_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('power_range', data=power_range, compression="gzip", compression_opts=9)
hf.create_dataset('power_bins', data=power_bins, compression="gzip", compression_opts=9)
hf.create_dataset('power_bin_center', data=power_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_map', data=fft_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('fft_rf_cut_map', data=fft_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_map', data=sub_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_map', data=sub_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_ratio_rf_map', data=amp_ratio_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_ratio_rf_cut_map', data=amp_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_rf_map', data=amp_err_ratio_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_rf_cut_map', data=amp_err_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_ratio_rf_map', data=phase_err_ratio_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_ratio_rf_cut_map', data=phase_err_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_phase_err_rf_map', data=amp_err_phase_err_rf_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_phase_err_rf_cut_map', data=amp_err_phase_err_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('power_rf_hist', data=np.asarray(power_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('power_rf_cut_hist', data=np.asarray(power_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('ratio_rf_hist', data=np.asarray(ratio_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('ratio_rf_cut_hist', data=np.asarray(ratio_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_rf_hist', data=np.asarray(amp_err_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_rf_cut_hist', data=np.asarray(amp_err_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_rf_hist', data=np.asarray(phase_err_rf_hist), compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_rf_cut_hist', data=np.asarray(phase_err_rf_cut_hist), compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_map_w', data=sub_rf_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('sub_rf_cut_map_w', data=sub_rf_cut_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('amp_ratio_rf_map_w', data=amp_ratio_rf_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('amp_ratio_rf_cut_map_w', data=amp_ratio_rf_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_rf_map_w', data=amp_err_ratio_rf_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_ratio_rf_cut_map_w', data=amp_err_ratio_rf_cut_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_ratio_rf_map_w', data=phase_err_ratio_rf_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_ratio_rf_cut_map_w', data=phase_err_ratio_rf_cut_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_phase_err_rf_map_w', data=amp_err_phase_err_rf_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_phase_err_rf_cut_map_w', data=amp_err_phase_err_rf_cut_map_w, compression="gzip", compression_opts=9)
hf.create_dataset('power_rf_hist_w', data=np.asarray(power_rf_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('power_rf_cut_hist_w', data=np.asarray(power_rf_cut_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('ratio_rf_hist_w', data=np.asarray(ratio_rf_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('ratio_rf_cut_hist_w', data=np.asarray(ratio_rf_cut_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_rf_hist_w', data=np.asarray(amp_err_rf_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('amp_err_rf_cut_hist_w', data=np.asarray(amp_err_rf_cut_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_rf_hist_w', data=np.asarray(phase_err_rf_hist_w), compression="gzip", compression_opts=9)
hf.create_dataset('phase_err_rf_cut_hist_w', data=np.asarray(phase_err_rf_cut_hist_w), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






