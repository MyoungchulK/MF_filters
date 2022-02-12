import numpy as np
import os, sys
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import file_sorter
from tools.utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

knwon_issue = known_issue_loader(Station)
bad_runs = knwon_issue.get_knwon_bad_run()

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_temp/*'
print(d_path)
d_list, d_run_tot, d_run_range = file_sorter(d_path)
del d_run_range

# config array
read_err = np.full((3), 0, dtype = int)
zero_err = np.copy(read_err)
blk_gap_err = np.copy(read_err)
evt_type = np.copy(read_err)
rf_evt_type = np.copy(read_err)
bias_rf_evt_type = np.copy(read_err)
bias_only_rf_evt_type = np.copy(read_err)
evt_num_runs = np.copy(read_err)
evt_num_rf_evts = np.copy(read_err)
timing_err_in_gap = np.copy(read_err)
timing_err_num = np.full((16), 0, dtype = int)
tot_qual_cut = np.full((11), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):

 #if r <10:

    hf = h5py.File(d_list[r], 'r')
    qual_cut = hf['pre_qual_cut'][:]
    trig_type = hf['trig_type'][:]
    timing_err = hf['timing_err'][:]
    evt_num = hf['evt_num'][:]
    del hf

    timing_sum = np.nansum(timing_err, axis = 0, dtype = int)
    timing_idx = np.where(timing_sum != 0)[0]
    if len(timing_idx) > 0:
        print(Station, d_run_tot[r], evt_num[timing_idx])
        print(timing_err[:, timing_idx])
        print(timing_idx)

    tot_qual_cut += np.count_nonzero(qual_cut != 0, axis = 0)

    blk_gap = qual_cut[:,6].astype(int)
    timing_flag = np.nansum(timing_err, axis = 0, dtype = int) != 0  
    timing_gap = blk_gap[timing_flag]
    timing_err_in_gap[0] += len(timing_gap)
    timing_err_in_gap[1] += np.count_nonzero(timing_gap != 0)
    timing_err_in_gap[2] += np.count_nonzero(timing_gap == 0) 
    del timing_flag, timing_gap

    timing_err_num += np.count_nonzero(timing_err, axis = 1) 
    del timing_err

    rf_count = np.count_nonzero(trig_type == 0)
    evt_type[0] += rf_count
    evt_type[1] += np.count_nonzero(trig_type == 1)
    evt_type[2] += np.count_nonzero(trig_type == 2)
    rf_evt_type[0] += rf_count

    bias_volt = (qual_cut[:,-3] + qual_cut[:,-2]).astype(int)
    bias_rf_evt_type[0] += rf_count
    bias_rf_evt_type[1] += np.count_nonzero(np.logical_and(bias_volt == 0, trig_type == 0))
    bias_rf_evt_type[2] += np.count_nonzero(np.logical_and(bias_volt != 0, trig_type == 0))

    bias_only_qual_cut = (qual_cut[:,-3]).astype(int)
    bias_only_rf_evt_type[0] += rf_count
    bias_only_rf_evt_type[1] += np.count_nonzero(np.logical_and(bias_only_qual_cut == 0, trig_type == 0))
    bias_only_rf_evt_type[2] += np.count_nonzero(np.logical_and(bias_only_qual_cut != 0, trig_type == 0))
    del bias_only_qual_cut

    evt_num_issue = (qual_cut[:,0]).astype(int)
    evt_num_runs[0] += 1
    if np.any(evt_num_issue != 0): 
        evt_num_runs[2] += 1
    else:
        evt_num_runs[1] += 1 
    
    evt_num_rf_evts[0] += rf_count
    evt_num_rf_evts[1] += np.count_nonzero(np.logical_and(evt_num_issue == 0, trig_type == 0))
    evt_num_rf_evts[2] += np.count_nonzero(np.logical_and(evt_num_issue != 0, trig_type == 0))
    del evt_num_issue

    if d_run_tot[r] in bad_runs:
        rf_evt_type[2] += rf_count
        #print('bad run:', d_list[r], d_run_tot[r])
        continue
    del rf_count       
 
    read_flag = (qual_cut[:,2] + qual_cut[:,3] + qual_cut[:,4]).astype(int) != 0
    zero_flag = qual_cut[:,5].astype(int) != 0
    blk_gap_flag = blk_gap != 0
    del blk_gap

    qual_cut_temp = np.copy(qual_cut)
    qual_cut_temp[:, -1] = 0
    qual_cut_sum = np.nansum(qual_cut_temp, axis = 1)
    del qual_cut_temp, qual_cut

    rf_evt_type[1] += np.count_nonzero(np.logical_and(qual_cut_sum == 0, trig_type == 0))
    rf_evt_type[2] += np.count_nonzero(np.logical_and(qual_cut_sum != 0, trig_type == 0))
    del trig_type

    read_bias = bias_volt[read_flag]
    read_err[0] += len(read_bias)
    read_err[1] += np.count_nonzero(read_bias != 0)
    read_err[2] += np.count_nonzero(read_bias == 0)

    zero_bias = bias_volt[zero_flag]
    zero_err[0] += len(zero_bias)
    zero_err[1] += np.count_nonzero(zero_bias != 0)
    zero_err[2] += np.count_nonzero(zero_bias == 0)

    blk_gap_bias = bias_volt[blk_gap_flag]
    blk_gap_err[0] += len(blk_gap_bias)
    blk_gap_err[1] += np.count_nonzero(blk_gap_bias != 0)
    blk_gap_err[2] += np.count_nonzero(blk_gap_bias == 0)

    del bias_volt, read_flag, zero_flag, blk_gap_flag, read_bias, zero_bias, blk_gap_bias

print('read_err:',read_err)
print('zero_err:',zero_err)
print('blk_gap_err:',blk_gap_err)
print('evt_type:',evt_type)
print('rf_evt_type:',rf_evt_type)
print('bias_rf_evt_type:',bias_rf_evt_type)
print('bias_only_rf_evt_type:',bias_only_rf_evt_type)
print('evt_num_runs:',evt_num_runs)
print('evt_num_rf_evts:',evt_num_rf_evts)
print('timing_err_in_gap:',timing_err_in_gap)
print('timing_err_num:',timing_err_num)
print('tot_qual_cut:',tot_qual_cut)

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'qual_cut_v1_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('read_err', data=read_err, compression="gzip", compression_opts=9)
hf.create_dataset('zero_err', data=zero_err, compression="gzip", compression_opts=9)
hf.create_dataset('blk_gap_err', data=blk_gap_err, compression="gzip", compression_opts=9)
hf.create_dataset('evt_type', data=evt_type, compression="gzip", compression_opts=9)
hf.create_dataset('rf_evt_type', data=rf_evt_type, compression="gzip", compression_opts=9)
hf.create_dataset('bias_rf_evt_type', data=bias_rf_evt_type, compression="gzip", compression_opts=9)
hf.create_dataset('bias_only_rf_evt_type', data=bias_only_rf_evt_type, compression="gzip", compression_opts=9)
hf.create_dataset('evt_num_runs', data=evt_num_runs, compression="gzip", compression_opts=9)
hf.create_dataset('evt_num_rf_evts', data=evt_num_rf_evts, compression="gzip", compression_opts=9)
hf.create_dataset('timing_err_in_gap', data=timing_err_in_gap, compression="gzip", compression_opts=9)
hf.create_dataset('timing_err_num', data=timing_err_num, compression="gzip", compression_opts=9)
hf.create_dataset('tot_qual_cut', data=tot_qual_cut, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)

# quick size check
size_checker(path+file_name)







