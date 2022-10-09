import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
#from tools.ara_run_manager import run_info_loader
#from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/medi/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

g_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/'

run_num = []

medi_bins = np.linspace(0, 2500, 250 + 1)
medi_bin_center = (medi_bins[1:] + medi_bins[:-1]) / 2
medi_len = len(medi_bin_center)
sen_bins = np.linspace(0, 10, 50 + 1)
sen_bin_center = (sen_bins[1:] + sen_bins[:-1]) / 2
sen_len = len(sen_bin_center)
cal_bins = np.linspace(0, 5, 50 + 1)
cal_bin_center = (cal_bins[1:] + cal_bins[:-1]) / 2

medi = []
medi_sen = []
medi_cal = []
medi_tot = []
sen = []
cal = []

q_idx = np.array([10, 12], dtype = int)

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    qual_dat = f'{g_path}qual_cut/qual_cut_A{Station}_R{d_run_tot[r]}.h5'
    qual_f = h5py.File(qual_dat, 'r')
    qual = qual_f['tot_qual_cut'][:, q_idx]
    tot_q = np.nansum(qual, axis = 1) != 0
    sen_q = qual[:, 0] != 0
    cal_q = qual[:, 1] != 0
    del qual_dat, qual_f 

    sub_dat = f'{g_path}sub_info_full/sub_info_full_A{Station}_R{d_run_tot[r]}.h5'
    sub_f = h5py.File(sub_dat, 'r')
    sen_v = sub_f['dda_volt'][:]
    cal_r_min = sub_f['cal_min_rate_pps'][:]
    del sub_dat, sub_f

    cal_hist = np.histogram(cal_r_min, bins = cal_bins)[0].astype(int)
    sen_hist = np.full((sen_len, 4), 0, dtype = int)
    for d in range(4):
        sen_hist[:, d] = np.histogram(sen_v[:, d], bins = sen_bins)[0].astype(int)
    del sen_v, cal_r_min

    cal.append(cal_hist)
    sen.append(sen_hist)
    run_num.append(d_run_tot[r])

    medi_f = h5py.File(d_list[r], 'r') 
    medi_rf = medi_f['medi'][:]
    trig = medi_f['trig_type'][:]
    trig = trig != 0
    medi_rf[:, trig] = np.nan
    del medi_f, trig

    medi_sen_r = np.copy(medi_rf)
    medi_sen_r[:, sen_q] = np.nan
    medi_cal_r = np.copy(medi_rf)
    medi_cal_r[:, cal_q] = np.nan
    medi_tot_r = np.copy(medi_rf)
    medi_tot_r[:, tot_q] = np.nan
    del qual, tot_q, sen_q, cal_q

    medi_hist = np.full((medi_len, 16), 0, dtype = int)
    medi_hist_sen = np.copy(medi_hist)
    medi_hist_cal = np.copy(medi_hist)
    medi_hist_tot = np.copy(medi_hist)
    for m in range(16):    
        medi_hist[:, m] = np.histogram(medi_rf[m], bins = medi_bins)[0].astype(int)
        medi_hist_sen[:, m] = np.histogram(medi_sen_r[m], bins = medi_bins)[0].astype(int)
        medi_hist_cal[:, m] = np.histogram(medi_cal_r[m], bins = medi_bins)[0].astype(int)
        medi_hist_tot[:, m] = np.histogram(medi_tot_r[m], bins = medi_bins)[0].astype(int)
    del medi_rf, medi_sen_r, medi_cal_r, medi_tot_r
    medi.append(medi_hist)
    medi_sen.append(medi_hist_sen)
    medi_cal.append(medi_hist_cal)
    medi_tot.append(medi_hist_tot)


path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Medi_Cut_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('medi_bins', data=medi_bins, compression="gzip", compression_opts=9)
hf.create_dataset('medi_bin_center', data=medi_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sen_bins', data=sen_bins, compression="gzip", compression_opts=9)
hf.create_dataset('sen_bin_center', data=sen_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('cal_bins', data=cal_bins, compression="gzip", compression_opts=9)
hf.create_dataset('cal_bin_center', data=cal_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('run_num', data=np.asarray(run_num), compression="gzip", compression_opts=9)
hf.create_dataset('medi', data=np.asarray(medi), compression="gzip", compression_opts=9)
hf.create_dataset('medi_sen', data=np.asarray(medi_sen), compression="gzip", compression_opts=9)
hf.create_dataset('medi_cal', data=np.asarray(medi_cal), compression="gzip", compression_opts=9)
hf.create_dataset('medi_tot', data=np.asarray(medi_tot), compression="gzip", compression_opts=9)
hf.create_dataset('sen', data=np.asarray(sen), compression="gzip", compression_opts=9)
hf.create_dataset('cal', data=np.asarray(cal), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
