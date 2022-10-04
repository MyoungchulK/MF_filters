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
#sensors = []

medi_bins = np.linspace(0, 2500, 250 + 1)
medi_bin_center = (medi_bins[1:] + medi_bins[:-1]) / 2
medi_len = len(medi_bin_center)
#sen_bins = np.linspace(0, 4, 40 + 1)
#sen_bin_center = (sen_bins[1:] + sen_bins[:-1]) / 2
#sen_len = len(sen_bin_center)
#cal_bins = np.linspace(0, 4, 40 + 1)
#cal_bin_center = (cal_bins[1:] + cal_bins[:-1]) / 2

medi = []
#medi_1st = []
#medi_vol = []
#medi_cal = []
#sen = []
#cal_min = []
#cal_sec = []

q_idx = np.array([11, 12, 15], dtype = int)

for r in tqdm(range(len(d_run_tot))):
  #if r < 10:

    qual_dat = f'{g_path}qual_cut/qual_cut_A{Station}_R{d_run_tot[r]}.h5'
    qual_f = h5py.File(qual_dat, 'r')
    qual = qual_f['tot_qual_cut'][:, q_idx]
    del qual_dat, qual_f 

    sub_dat = f'{g_path}sub_info_full/sub_info_full_A{Station}_R{d_run_tot[r]}.h5'
    sub_f = h5py.File(sub_dat, 'r')
    sen_u = np.any(np.isnan(sub_f['sensor_unix_time'][:]))#.astype(int)
    #sen_v = sub_f['dda_volt'][:]
    #cal_r_min = sub_f['cal_min_rate_pps'][:]
    #cal_r_sec = sub_f['cal_sec_rate_pps'][:]
    del sub_dat, sub_f

    if ~sen_u:
        qual[:, -1] = 0
    qual = np.nansum(qual, axis = 1) != 0
    del sen_u

    #cal_m_hist = np.histogram(cal_r_min, bins = cal_bins)[0].astype(int)
    #cal_s_hist = np.histogram(cal_r_sec, bins = cal_bins)[0].astype(int)
    #del cal_r_min, cal_r_sec
    #cal_min.append(cal_m_hist)
    #cal_sec.append(cal_s_hist)

    #sen_hist = np.full((sen_len, 4), 0, dtype = int)
    #for d in range(4):
    #    sen_hist[:, d] = np.histogram(sen_v[:, d], bins = sen_bins)[0].astype(int)
    #del sen_v 
    #sen.append(sen_hist)

    run_num.append(d_run_tot[r])
    #sensors.append(sen_u)

    medi_f = h5py.File(d_list[r], 'r') 
    medi_r = medi_f['medi'][:]
    trig = medi_f['trig_type'][:]
    trig = trig != 0
    medi_r[:, qual] = np.nan
    medi_r[:, trig] = np.nan
    del medi_f

    #medi_r_1st = np.copy(medi_r)
    #medi_r_1st[:, qual[:, 0] != 0] = np.nan
    #medi_r_vol = np.copy(medi_r)
    #medi_r_vol[:, qual[:, 1] != 0] = np.nan
    #medi_r_cal = np.copy(medi_r)
    #medi_r_cal[:, qual[:, 2] != 0] = np.nan
    del qual

    medi_hist = np.full((medi_len, 16), 0, dtype = int)
    #medi_hist_1st = np.copy(medi_hist)
    #medi_hist_vol = np.copy(medi_hist)
    #medi_hist_cal = np.copy(medi_hist)
    for m in range(16):    
        medi_hist[:, m] = np.histogram(medi_r[m], bins = medi_bins)[0].astype(int)
        #medi_hist_1st[:, m] = np.histogram(medi_r_1st[m], bins = medi_bins)[0].astype(int)
        #medi_hist_vol[:, m] = np.histogram(medi_r_vol[m], bins = medi_bins)[0].astype(int)
        #medi_hist_cal[:, m] = np.histogram(medi_r_cal[m], bins = medi_bins)[0].astype(int)
    medi.append(medi_hist)
    #medi_1st.append(medi_hist_1st)
    #medi_vol.append(medi_hist_vol)
    #medi_cal.append(medi_hist_cal)
    del medi_r#, medi_r_1st, medi_r_vol, medi_r_cal


path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Medi_A{Station}_v3.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('medi_bins', data=medi_bins, compression="gzip", compression_opts=9)
hf.create_dataset('medi_bin_center', data=medi_bin_center, compression="gzip", compression_opts=9)
#hf.create_dataset('sen_bins', data=sen_bins, compression="gzip", compression_opts=9)
#hf.create_dataset('sen_bin_center', data=sen_bin_center, compression="gzip", compression_opts=9)
#hf.create_dataset('cal_bins', data=cal_bins, compression="gzip", compression_opts=9)
#hf.create_dataset('cal_bin_center', data=cal_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('run_num', data=np.asarray(run_num), compression="gzip", compression_opts=9)
#hf.create_dataset('sensors', data=np.asarray(sensors), compression="gzip", compression_opts=9)
hf.create_dataset('medi', data=np.asarray(medi), compression="gzip", compression_opts=9)
#hf.create_dataset('medi_1st', data=np.asarray(medi_1st), compression="gzip", compression_opts=9)
#hf.create_dataset('medi_vol', data=np.asarray(medi_vol), compression="gzip", compression_opts=9)
#hf.create_dataset('medi_cal', data=np.asarray(medi_cal), compression="gzip", compression_opts=9)
#hf.create_dataset('sen', data=np.asarray(sen), compression="gzip", compression_opts=9)
#hf.create_dataset('cal_min', data=np.asarray(cal_min), compression="gzip", compression_opts=9)
#hf.create_dataset('cal_sec', data=np.asarray(cal_sec), compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
