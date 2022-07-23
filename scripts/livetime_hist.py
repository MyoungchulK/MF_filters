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

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/daq_cut/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

num_cuts = 23
sec_to_min = 60

tot_time = np.full((len(d_run_tot)), np.nan, dtype = float)
tot_num_evts = np.copy(tot_time)
tot_evt_per_min = np.full((1000, len(d_run_tot)), np.nan, dtype = float)
tot_sec_per_min = np.copy(tot_evt_per_min)
tot_time_bins = np.copy(tot_evt_per_min)
tot_sum_cut_evt_per_min = np.copy(tot_evt_per_min)
tot_cut_evt_per_min = np.full((num_cuts, 1000, len(d_run_tot)), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
   
  #if r < 10:
 
    hf = h5py.File(d_list[r], 'r')
    unix_time = hf['unix_time'][:]
    daq_cut = hf['total_daq_cut'][:]
    trig_type = hf['trig_type'][:]
    del hf

    rc_trig = trig_type != 2
    num_evts = np.count_nonzero(rc_trig)
    tot_num_evts[r] = num_evts
    tot_t = np.abs(unix_time[-1] - unix_time[0])
    tot_time[r] = tot_t    
    del trig_type, num_evts, tot_t

    time_bins = np.arange(np.nanmin(unix_time), np.nanmax(unix_time)+1, sec_to_min, dtype = int)
    time_bins = time_bins.astype(float)
    time_bins -= 0.5
    time_bins = np.append(time_bins, np.nanmax(unix_time) + 0.5)
    tot_time_bins[:len(time_bins), r] = time_bins

    sec_per_min = np.diff(time_bins)
    tot_sec_per_min[:len(sec_per_min), r] = sec_per_min
    del sec_per_min

    rc_trig_evt = rc_trig.astype(int)
    rc_trig_evt = rc_trig_evt.astype(float)
    rc_trig_evt[rc_trig_evt < 0.5] = np.nan
    rc_trig_evt *= unix_time
    evt_per_min = np.histogram(rc_trig_evt, bins = time_bins)[0]
    tot_evt_per_min[:len(evt_per_min), r] = evt_per_min
    del rc_trig_evt, evt_per_min

    daq_cut_flag = (daq_cut != 0).astype(int)
    daq_cut_flag = daq_cut_flag.astype(float)
    daq_cut_flag[daq_cut_flag < 0.5] = np.nan
    daq_cut_flag[~rc_trig] = np.nan
    del daq_cut, rc_trig

    #tot_copy = np.copy(daq_cut_flag)
    #tot_copy[:,-1] = np.nan 
    #tot_daq_cut_flag = np.nansum(tot_copy, axis = 1)
    tot_daq_cut_flag = np.nansum(daq_cut_flag, axis = 1)
    tot_idx = tot_daq_cut_flag < 0.5
    tot_daq_cut_flag[~tot_idx] = 1
    tot_daq_cut_flag[tot_idx] = np.nan
    tot_daq_cut_flag *= unix_time
    sum_cut_evt_per_min = np.histogram(tot_daq_cut_flag, bins = time_bins)[0]
    tot_sum_cut_evt_per_min[:len(sum_cut_evt_per_min), r] = sum_cut_evt_per_min
    del tot_daq_cut_flag, sum_cut_evt_per_min 

    daq_cut_flag *= unix_time[:, np.newaxis]
    for t in range(num_cuts):
        cut_evt_per_min = np.histogram(daq_cut_flag[:, t], bins = time_bins)[0] 
        tot_cut_evt_per_min[t, :len(cut_evt_per_min), r] = cut_evt_per_min
        del cut_evt_per_min
    del unix_time, time_bins, daq_cut_flag

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Livetime_daq_cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('tot_time', data=tot_time, compression="gzip", compression_opts=9)
hf.create_dataset('tot_num_evts', data=tot_num_evts, compression="gzip", compression_opts=9)
hf.create_dataset('tot_evt_per_min', data=tot_evt_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_sec_per_min', data=tot_sec_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_time_bins', data=tot_time_bins, compression="gzip", compression_opts=9)
hf.create_dataset('tot_sum_cut_evt_per_min', data=tot_sum_cut_evt_per_min, compression="gzip", compression_opts=9)
hf.create_dataset('tot_cut_evt_per_min', data=tot_cut_evt_per_min, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






