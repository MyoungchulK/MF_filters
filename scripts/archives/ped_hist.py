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
from tools.antenna import antenna_info
from tools.run import bin_range_maker
from tools.run import config_checker

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
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)

p_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Medi_Tilt/*'
p_list, p_run_tot, p_run_range = file_sorter(p_path)

pj_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Medi_Tilt_kJustPed/*'
pj_list, pj_run_tot, pj_run_range = file_sorter(pj_path)

# detector config
ant_num = antenna_info()[2]

# config array
run_tot_list = []
config_v2_list = []
date_list = []
unix_list = []

medi_front = []
medi_mid_list = []
medi_back = []

medi_fb_arr = np.arange(-1000,1000)
medi_fb_bins, medi_fb_bin_center = bin_range_maker(medi_fb_arr, len(medi_fb_arr))
medi_fb = np.zeros((2,len(medi_fb_bin_center), ant_num))
print(medi_fb.shape)

medi_mid_arr = np.arange(-1000,1000)
medi_mid_bins, medi_mid_bin_center = bin_range_maker(medi_mid_arr, len(medi_mid_arr))
medi_mid = np.zeros((len(medi_mid_bin_center), ant_num))
print(medi_mid.shape)

r_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Medi_Tilt_repeder/*'
r_list, r_run_tot, r_run_range = file_sorter(r_path)

medi_front_re = []
medi_mid_re_list = []
medi_back_re = []

medi_fb_re_arr = np.arange(-1000,1000)
medi_fb_re_bins, medi_fb_re_bin_center = bin_range_maker(medi_fb_re_arr, len(medi_fb_re_arr))
medi_fb_re = np.zeros((2,len(medi_fb_re_bin_center), ant_num))
print(medi_fb_re.shape)

medi_mid_re_arr = np.arange(-1000,1000)
medi_mid_re_bins, medi_mid_re_bin_center = bin_range_maker(medi_mid_re_arr, len(medi_mid_re_arr))
medi_mid_re = np.zeros((len(medi_mid_re_bin_center), ant_num))
print(medi_mid_re.shape)

for r in tqdm(range(len(d_run_tot))):
  #if r < 50: 
    if d_run_tot[r] in bad_runs:
        print('bad run:',d_list[r],d_run_tot[r])
        continue
    else:

        if d_run_tot[r] == p_run_tot[r] == pj_run_tot[r] == r_run_tot[r]:
            pass
        else:
            print(d_list[r])
            print(p_list[r])
            print(pj_list[r])
            print(r_list[r])
            sys.exit(1)

        hf = h5py.File(d_list[r], 'r')
        config_v2 = config_checker(Station, d_run_tot[r])
        config_v2_list.append(config_v2)
        unix_time = hf['unix_time'][0]
        unix_list.append(unix_time[0])
        date_time = datetime.fromtimestamp(unix_time[0])
        date_time1 = date_time.strftime('%Y%m%d%H%M%S')
        date_list.append(int(date_time1))
        run_tot_list.append(d_run_tot[r])
        trig_num = hf['trig_num'][:]
        rf_trig = np.where(trig_num == 0)[0]
        del trig_num, config_v2, unix_time, date_time, date_time1, hf

        hf = h5py.File(pj_list[r], 'r')
        medi_all = hf['medi_all'][1]
        rf_medi_all = medi_all[:,rf_trig]
        del hf, medi_all
        
        hf = h5py.File(p_list[r], 'r')
        medi_all = hf['medi_all'][:]
        medi_all_f = medi_all[0]
        medi_all_b = medi_all[2]
        rf_medi_front_all = medi_all_f[:,rf_trig]
        rf_medi_back_all = medi_all_b[:,rf_trig]
        del hf, medi_all, medi_all_f, medi_all_b

        temp_medi_f = np.zeros((len(medi_fb_bin_center), ant_num))
        temp_medi_b = np.zeros((len(medi_fb_bin_center), ant_num))
        temp_medi_mid = np.zeros((len(medi_mid_bin_center), ant_num))

        for a in range(ant_num):

            rf_medi_hist = np.histogram(rf_medi_all[a], bins = medi_mid_bins)[0]
            rf_medi_front_hist = np.histogram(rf_medi_front_all[a], bins = medi_fb_bins)[0]
            rf_medi_back_hist = np.histogram(rf_medi_back_all[a], bins = medi_fb_bins)[0]

            medi_mid[:,a] += rf_medi_hist   
            medi_fb[0,:,a] += rf_medi_front_hist
            medi_fb[1,:,a] += rf_medi_back_hist
            
            temp_medi_f[:,a] = rf_medi_front_hist
            temp_medi_b[:,a] = rf_medi_back_hist
            temp_medi_mid[:,a] = rf_medi_hist
            del rf_medi_hist, rf_medi_front_hist, rf_medi_back_hist

        medi_front.append(temp_medi_f)
        medi_back.append(temp_medi_b)
        medi_mid_list.append(temp_medi_mid)
        del temp_medi_f, temp_medi_b, temp_medi_mid 
        del rf_medi_all, rf_medi_front_all, rf_medi_back_all

        hf = h5py.File(r_list[r], 'r')
        medi_all = hf['medi_all'][:]
        medi_all_mid = medi_all[0,1]
        medi_all_f = medi_all[1,0]
        medi_all_b = medi_all[1,2]
        rf_medi_re_all = medi_all_mid[:,rf_trig]
        rf_medi_front_re_all = medi_all_f[:,rf_trig]
        rf_medi_back_re_all = medi_all_b[:,rf_trig]
        del hf, medi_all, medi_all_mid, medi_all_f, medi_all_b
        temp_medi_f_re = np.zeros((len(medi_fb_re_bin_center), ant_num))
        temp_medi_b_re = np.zeros((len(medi_fb_re_bin_center), ant_num))
        temp_medi_mid_re = np.zeros((len(medi_mid_re_bin_center), ant_num))
        for a in range(ant_num):

            rf_medi_re_hist = np.histogram(rf_medi_re_all[a], bins = medi_mid_re_bins)[0]
            rf_medi_front_re_hist = np.histogram(rf_medi_front_re_all[a], bins = medi_fb_re_bins)[0]
            rf_medi_back_re_hist = np.histogram(rf_medi_back_re_all[a], bins = medi_fb_re_bins)[0]

            medi_mid_re[:,a] += rf_medi_re_hist
            medi_fb_re[0,:,a] += rf_medi_front_re_hist
            medi_fb_re[1,:,a] += rf_medi_back_re_hist

            temp_medi_f_re[:,a] = rf_medi_front_re_hist
            temp_medi_b_re[:,a] = rf_medi_back_re_hist
            temp_medi_mid_re[:,a] = rf_medi_re_hist
            del rf_medi_re_hist, rf_medi_front_re_hist, rf_medi_back_re_hist

        medi_front_re.append(temp_medi_f_re)
        medi_back_re.append(temp_medi_b_re)
        medi_mid_re_list.append(temp_medi_mid_re)
        del temp_medi_f_re, temp_medi_b_re, temp_medi_mid_re
        del rf_medi_re_all, rf_medi_front_re_all, rf_medi_back_re_all
        del rf_trig

run_tot_list = np.asarray(run_tot_list)
config_v2_list = np.asarray(config_v2_list)
date_list = np.asarray(date_list)
unix_list = np.asarray(unix_list)

medi_front = np.transpose(np.asarray(medi_front),(1,2,0))
medi_mid_list = np.transpose(np.asarray(medi_mid_list),(1,2,0))
medi_back = np.transpose(np.asarray(medi_back),(1,2,0))
print(run_tot_list.shape)
print(config_v2_list.shape)
print(date_list.shape)
print(unix_list.shape)
print(medi_front.shape)
print(medi_mid_list.shape)
print(medi_back.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Medi_Hist_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_tot_list', data=run_tot_list, compression="gzip", compression_opts=9)
hf.create_dataset('config_v2_list', data=config_v2_list, compression="gzip", compression_opts=9)
hf.create_dataset('date_list', data=date_list, compression="gzip", compression_opts=9)
hf.create_dataset('unix_list', data=unix_list, compression="gzip", compression_opts=9)

hf.create_dataset('medi_fb_arr', data=medi_fb_arr, compression="gzip", compression_opts=9)
hf.create_dataset('medi_fb_bins', data=medi_fb_bins, compression="gzip", compression_opts=9)
hf.create_dataset('medi_fb_bin_center', data=medi_fb_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('medi_fb', data=medi_fb, compression="gzip", compression_opts=9)

hf.create_dataset('medi_mid_arr', data=medi_mid_arr, compression="gzip", compression_opts=9)
hf.create_dataset('medi_mid_bins', data=medi_mid_bins, compression="gzip", compression_opts=9)
hf.create_dataset('medi_mid_bin_center', data=medi_mid_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('medi_mid', data=medi_mid, compression="gzip", compression_opts=9)

hf.create_dataset('medi_mid_list', data=medi_mid_list, compression="gzip", compression_opts=9)
hf.create_dataset('medi_front', data=medi_front, compression="gzip", compression_opts=9)
hf.create_dataset('medi_back', data=medi_back, compression="gzip", compression_opts=9)

medi_front_re = np.transpose(np.asarray(medi_front_re),(1,2,0))
medi_mid_re_list = np.transpose(np.asarray(medi_mid_re_list),(1,2,0))
medi_back_re = np.transpose(np.asarray(medi_back_re),(1,2,0))
print(medi_front_re.shape)
print(medi_mid_re_list.shape)
print(medi_back_re.shape)

hf.create_dataset('medi_fb_re_arr', data=medi_fb_re_arr, compression="gzip", compression_opts=9)
hf.create_dataset('medi_fb_re_bins', data=medi_fb_re_bins, compression="gzip", compression_opts=9)
hf.create_dataset('medi_fb_re_bin_center', data=medi_fb_re_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('medi_fb_re', data=medi_fb_re, compression="gzip", compression_opts=9)

hf.create_dataset('medi_mid_re_arr', data=medi_mid_re_arr, compression="gzip", compression_opts=9)
hf.create_dataset('medi_mid_re_bins', data=medi_mid_re_bins, compression="gzip", compression_opts=9)
hf.create_dataset('medi_mid_re_bin_center', data=medi_mid_re_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('medi_mid_re', data=medi_mid_re, compression="gzip", compression_opts=9)

hf.create_dataset('medi_mid_re_list', data=medi_mid_re_list, compression="gzip", compression_opts=9)
hf.create_dataset('medi_front_re', data=medi_front_re, compression="gzip", compression_opts=9)
hf.create_dataset('medi_back_re', data=medi_back_re, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')





















