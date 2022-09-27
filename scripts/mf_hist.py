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
if Station == 2:
    config_len = 6
if Station == 3:
    config_len = 7

d_type = ''
#d_type = '_sim'
s_type = ''
#s_type = 'noise_'
#s_type = 'signal_F1_'

bad_path = f'../data/rayl_runs/rayl_run_A{Station}.txt'
bad_run_arr = []
with open(bad_path, 'r') as f:
    for lines in f:
        run_num = int(lines)
        bad_run_arr.append(run_num)
bad_run_arr = np.asarray(bad_run_arr, dtype = int)
del bad_path

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mf{d_type}/*{s_type}*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
if d_type != '_sim':
    q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut/'
    c_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_cut/'

mf_bins = np.linspace(0,15,1500+1)
mf_bin_center = (mf_bins[1:] + mf_bins[:-1]) / 2
mf_bin_len = len(mf_bin_center)
mf_hist = np.full((2, mf_bin_len, config_len), 0, dtype = int)
if d_type != '_sim':
    mf_hist_rf = np.copy(mf_hist)
    mf_hist_cal = np.copy(mf_hist)
    mf_hist_soft = np.copy(mf_hist)
    mf_hist_rf_clean = np.copy(mf_hist)
    mf_hist_cal_clean = np.copy(mf_hist)
    mf_hist_soft_clean = np.copy(mf_hist)
    mf_list = []
    mf_list_rf = []
    mf_list_cal = []
    mf_list_soft = []
    mf_list_rf_clean = []
    mf_list_cal_clean = []
    mf_list_soft_clean = [] 

num_evts = np.full((config_len), 0, dtype = float)

for r in tqdm(range(len(d_run_tot))):

    if d_run_tot[r] in bad_run_arr:
        print('bad run:', d_list[r], d_run_tot[r])
        continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    c_idx = ara_run.get_config_number() - 1
    del ara_run

    if d_type != '_sim':
        q_name = f'{q_path}qual_cut_A{Station}_R{d_run_tot[r]}.h5'
        c_name = f'{c_path}cw_cut_A{Station}_R{d_run_tot[r]}.h5'
        hf_q = h5py.File(q_name, 'r')
        q_cut_tot = hf_q['tot_qual_cut'][:]
        q_cut_tot[:, 10] = 0 # disable bad unix time
        q_cut_tot[:, 21] = 0 # disable known bad run
        q_cut = np.nansum(q_cut_tot, axis = 1) 
        trig_type = hf_q['trig_type'][:]
        hf_c = h5py.File(c_name, 'r')
        q_cut += hf_c['cw_qual_cut_sum'][:]
        q_cut = q_cut.astype(int)
        del q_name, c_name, hf_q, hf_c, q_cut_tot
        rf_idx = trig_type == 0
        cal_idx = trig_type == 1
        soft_idx = trig_type == 2
        rf_c_idx = np.logical_and(rf_idx, q_cut == 0)
        cal_c_idx = np.logical_and(cal_idx, q_cut == 0)
        soft_c_idx = np.logical_and(soft_idx, q_cut == 0)
        del trig_type, q_cut

    hf = h5py.File(d_list[r], 'r')
    evt_wise = hf['evt_wise'][:]
    del hf
    if d_type != '_sim':
            evt_w_rf = np.copy(evt_wise)
            evt_w_rf[:, ~rf_idx] = np.nan
            evt_w_cal = np.copy(evt_wise)
            evt_w_cal[:, ~cal_idx] = np.nan
            evt_w_soft = np.copy(evt_wise)
            evt_w_soft[:, ~soft_idx] = np.nan
            evt_w_rf_c = np.copy(evt_wise)
            evt_w_rf_c[:, ~rf_c_idx] = np.nan
            evt_w_cal_c = np.copy(evt_wise)
            evt_w_cal_c[:, ~cal_c_idx] = np.nan
            evt_w_soft_c = np.copy(evt_wise)
            evt_w_soft_c[:, ~soft_c_idx] = np.nan

            mf_h_rf = np.full((2, mf_bin_len), 0, dtype = int)
            mf_h_cal = np.full((2, mf_bin_len), 0, dtype = int)
            mf_h_soft = np.full((2, mf_bin_len), 0, dtype = int)
            mf_h_rf_c = np.full((2, mf_bin_len), 0, dtype = int)
            mf_h_cal_c = np.full((2, mf_bin_len), 0, dtype = int)
            mf_h_soft_c = np.full((2, mf_bin_len), 0, dtype = int)
            for a in range(2):
                mf_h_rf[a] = np.histogram(evt_w_rf[a], bins = mf_bins)[0]
                mf_h_cal[a] = np.histogram(evt_w_cal[a], bins = mf_bins)[0]
                mf_h_soft[a] = np.histogram(evt_w_soft[a], bins = mf_bins)[0]
                mf_h_rf_c[a] = np.histogram(evt_w_rf_c[a], bins = mf_bins)[0]
                mf_h_cal_c[a] = np.histogram(evt_w_cal_c[a], bins = mf_bins)[0]
                mf_h_soft_c[a] = np.histogram(evt_w_soft_c[a], bins = mf_bins)[0]
            mf_hist_rf[:, :, c_idx] += mf_h_rf
            mf_hist_cal[:, :, c_idx] += mf_h_cal
            mf_hist_soft[:, :, c_idx] += mf_h_soft
            mf_hist_rf_clean[:, :, c_idx] += mf_h_rf_c
            mf_hist_cal_clean[:, :, c_idx] += mf_h_cal_c
            mf_hist_soft_clean[:, :, c_idx] += mf_h_soft_c
            mf_list_rf.append(mf_h_rf)
            mf_list_cal.append(mf_h_cal)
            mf_list_soft.append(mf_h_soft)
            mf_list_rf_clean.append(mf_h_rf_c)
            mf_list_cal_clean.append(mf_h_cal_c)
            mf_list_soft_clean.append(mf_h_soft_c)
        
            num_evts[c_idx] += np.count_nonzero(~np.isnan(evt_w_rf_c[0]))
            del evt_w_rf, evt_w_cal, evt_w_soft, evt_w_rf_c, evt_w_cal_c, evt_w_soft_c
            del rf_idx, cal_idx, soft_idx, rf_c_idx, cal_c_idx, soft_c_idx
    else:
            num_evts[c_idx] += np.count_nonzero(~np.isnan(evt_wise[0]))

    mf_h = np.full((2, mf_bin_len), 0, dtype = int)
    for a in range(2):
        mf_h[a] = np.histogram(evt_wise[a], bins = mf_bins)[0]
    mf_hist[:, :, c_idx] += mf_h
    if d_type != '_sim':
        mf_list.append(mf_h)
    del evt_wise, c_idx

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'MF{d_type}_{s_type}A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('mf_bins', data=mf_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mf_bin_center', data=mf_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('mf_hist', data=mf_hist, compression="gzip", compression_opts=9)
if d_type != '_sim':
    hf.create_dataset('mf_hist_rf', data=mf_hist_rf, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_hist_cal', data=mf_hist_cal, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_hist_soft', data=mf_hist_soft, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_hist_rf_clean', data=mf_hist_rf_clean, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_hist_cal_clean', data=mf_hist_cal_clean, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_hist_soft_clean', data=mf_hist_soft_clean, compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list', data=np.asarray(mf_list), compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list_rf', data=np.asarray(mf_list_rf), compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list_cal', data=np.asarray(mf_list_cal), compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list_soft', data=np.asarray(mf_list_soft), compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list_rf_clean', data=np.asarray(mf_list_rf_clean), compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list_cal_clean', data=np.asarray(mf_list_cal_clean), compression="gzip", compression_opts=9)
    hf.create_dataset('mf_list_soft_clean', data=np.asarray(mf_list_soft_clean), compression="gzip", compression_opts=9)
hf.create_dataset('num_evts', data=num_evts, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
