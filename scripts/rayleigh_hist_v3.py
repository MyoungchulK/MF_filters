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
from tools.fft import psd_maker
from tools.fft import freq_pad_maker
from tools.fft import db_log_maker
from tools.antenna import antenna_info
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
d_list, run_tot, run_range = file_sorter(d_path)

r_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Rayl/*'
r_list, r_run_tot = file_sorter(r_path)[:2]

# detector config
ant_num = antenna_info()[2]

# freq bin
freq, df = freq_pad_maker(oneside = True, dfreq = True)

freq_bins, freq_bin_center = bin_range_maker(freq, len(freq))
print(freq_bin_center.shape)

# rayl 2d
rayl_2d_run_w_sc = np.full((len(freq_bin_center), len(run_range), ant_num),np.nan)
rayl_2d_run_wo_sc = np.copy(rayl_2d_run_w_sc)
print(rayl_2d_run_w_sc.shape)
print(rayl_2d_run_wo_sc.shape)

rayl_2d_run_w_sc_list = []
rayl_2d_run_wo_sc_list = []

mag_w_sc = np.arange(-125,-40)
mag_bins_w_sc, mag_bin_center_w_sc = bin_range_maker(mag_w_sc, len(mag_w_sc))

mag_wo_sc = np.arange(-180,-110)
mag_bins_wo_sc, mag_bin_center_wo_sc = bin_range_maker(mag_wo_sc, len(mag_wo_sc))

rayl_2d_freq_w_sc = np.zeros((len(freq_bin_center), len(mag_bin_center_w_sc), ant_num))
rayl_2d_freq_wo_sc = np.zeros((len(freq_bin_center), len(mag_bin_center_wo_sc), ant_num))
print(rayl_2d_freq_w_sc.shape)
print(rayl_2d_freq_wo_sc.shape)

sc_range = np.arange(0,135)
sc_bins, sc_bin_center = bin_range_maker(sc_range, len(sc_range))

sc_2d_run = np.copy(rayl_2d_run_w_sc)
sc_2d_freq = np.zeros((len(freq_bin_center), len(sc_bin_center), ant_num))
sc_2d_run_list = [] 
print(sc_2d_run.shape)
print(sc_2d_freq.shape)

config_arr = np.full((len(run_tot)),np.nan)
config_v2_arr = np.copy(config_arr)
date_arr = np.copy(config_arr)
unix_arr = np.copy(config_arr)

run_tot_list = []
config_list = []
config_v2_list = []
date_list = []
unix_list = []
"""
psd_w_max = 0
psd_w_min = 0
psd_wo_max = 0
psd_wo_min = 0
sc_max = 0
sc_min = 0
t = 0
"""
for r in tqdm(range(len(run_tot))):
  #if r <20: 
    hf_config = h5py.File(d_list[r], 'r')
    config = hf_config['config'][2]
    config_arr[r] = config
    config_v2 = config_checker(Station, run_tot[r])
    config_v2_arr[r] = config_v2
    unix_time = hf_config['unix_time'][0]
    unix_arr[r] = unix_time[0]
    date_time = datetime.fromtimestamp(unix_time[0])
    date_time1 = date_time.strftime('%Y%m%d%H%M%S')
    date_arr[r] = int(date_time1)
    del hf_config, date_time   
 
    rayl_run_idx = np.where(r_run_tot == run_tot[r])[0]
    if len(rayl_run_idx) > 0:

        rr = rayl_run_idx[0]
        rrr = np.where(run_range == run_tot[r])[0][0]
    
        if r_run_tot[rr] in bad_runs:
            print('bad run:',r_list[rr],r_run_tot[rr])
            continue
        else:

            config_list.append(config)
            config_v2_list.append(config_v2)
            date_list.append(int(date_time1))
            unix_list.append(unix_time[0])
            run_tot_list.append(r_run_tot[rr])

            hf = h5py.File(r_list[rr], 'r')

            psd_w_sc = hf['psd'][:] 
            mu_wo_sc = hf['mu_wo_sc'][:]
            psd_wo_sc = psd_maker(mu_wo_sc/2, df, oneside = True, symmetry = True, dbm_per_hz = True)
            sc = hf['sc'][:]
            sc_db = db_log_maker(sc**2)
            del sc, mu_wo_sc
            """
            if t == 0:
                psd_w_max = np.nanmax(psd_w_sc)
                psd_w_min = np.nanmin(psd_w_sc)
                psd_wo_max = np.nanmax(psd_wo_sc)
                psd_wo_min = np.nanmin(psd_wo_sc)
                sc_max = np.nanmax(sc_db)
                sc_min = np.nanmin(sc_db)
            else:
                if np.nanmax(psd_w_sc) > psd_w_max:
                    psd_w_max = np.nanmax(psd_w_sc)
                else:
                    pass
                if np.nanmin(psd_w_sc) < psd_w_min:
                    psd_w_min = np.nanmin(psd_w_sc)
                else:
                    pass
                if np.nanmax(psd_wo_sc) > psd_wo_max:
                    psd_wo_max = np.nanmax(psd_wo_sc)
                else:
                    pass
                if np.nanmin(psd_wo_sc) < psd_wo_min:
                    psd_wo_min = np.nanmin(psd_wo_sc)
                else:
                    pass
                if np.nanmax(sc_db) > sc_max:
                    sc_max = np.nanmax(sc_db)
                else:
                    pass
                if np.nanmin(sc_db) < sc_min:
                    sc_min = np.nanmin(sc_db)
                else:
                    pass
   
            print(psd_w_max)
            print(psd_w_min)
            print(psd_wo_max)
            print(psd_wo_min)
            print(sc_max)
            print(sc_min)
            t += 1   
            """
            rayl_2d_run_w_sc[:,rrr] = psd_w_sc
            rayl_2d_run_wo_sc[:,rrr] = psd_wo_sc
            rayl_2d_run_w_sc_list.append(psd_w_sc)
            rayl_2d_run_wo_sc_list.append(psd_wo_sc)
            
            sc_2d_run[:,rrr] = sc_db
            sc_2d_run_list.append(sc_db)

            for a in range(ant_num):
  
                rayl_2d_freq_w_sc[:,:,a] += np.histogram2d(freq, psd_w_sc[:,a], bins = (freq_bins,mag_bins_w_sc))[0]
                rayl_2d_freq_wo_sc[:,:,a] += np.histogram2d(freq, psd_wo_sc[:,a], bins = (freq_bins,mag_bins_wo_sc))[0]
 
                sc_2d_freq[:,:,a] += np.histogram2d(freq, sc_db[:,a], bins = (freq_bins,sc_bins))[0]

            del hf, psd_w_sc, psd_wo_sc, sc_db
        del rr, rrr

    else:
        print('No Rayl runs!!!!!!:', run_tot[r])
    del config, config_v2, unix_time, date_time1, rayl_run_idx 

config_list = np.asarray(config_list)
config_v2_list = np.asarray(config_v2_list)
date_list = np.asarray(date_list)
run_tot_list = np.asarray(run_tot_list)
rayl_2d_run_w_sc_list = np.transpose(np.asarray(rayl_2d_run_w_sc_list),(1,0,2))
rayl_2d_run_wo_sc_list = np.transpose(np.asarray(rayl_2d_run_wo_sc_list),(1,0,2))
sc_2d_run_list = np.transpose(np.asarray(sc_2d_run_list),(1,0,2))
print(rayl_2d_run_w_sc_list.shape)
print(rayl_2d_run_wo_sc_list.shape)
print(sc_2d_run_list.shape)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Rayl_Hist_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_range', data=run_range, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_tot_list', data=run_tot_list, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_list', data=config_list, compression="gzip", compression_opts=9)
hf.create_dataset('config_v2_arr', data=config_v2_arr, compression="gzip", compression_opts=9)
hf.create_dataset('config_v2_list', data=config_v2_list, compression="gzip", compression_opts=9)
hf.create_dataset('date_arr', data=date_arr, compression="gzip", compression_opts=9)
hf.create_dataset('date_list', data=date_list, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_list', data=unix_list, compression="gzip", compression_opts=9)

hf.create_dataset('df', data=np.array([df]), compression="gzip", compression_opts=9)
hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bins', data=freq_bins, compression="gzip", compression_opts=9)
hf.create_dataset('freq_bin_center', data=freq_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('mag_w_sc', data=mag_w_sc, compression="gzip", compression_opts=9)
hf.create_dataset('mag_bins_w_sc', data=mag_bins_w_sc, compression="gzip", compression_opts=9)
hf.create_dataset('mag_bin_center_w_sc', data=mag_bin_center_w_sc, compression="gzip", compression_opts=9)

hf.create_dataset('mag_wo_sc', data=mag_wo_sc, compression="gzip", compression_opts=9)
hf.create_dataset('mag_bins_wo_sc', data=mag_bins_wo_sc, compression="gzip", compression_opts=9)
hf.create_dataset('mag_bin_center_wo_sc', data=mag_bin_center_wo_sc, compression="gzip", compression_opts=9)

hf.create_dataset('rayl_2d_run_w_sc', data=rayl_2d_run_w_sc, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_2d_run_wo_sc', data=rayl_2d_run_wo_sc, compression="gzip", compression_opts=9)

hf.create_dataset('rayl_2d_run_w_sc_list', data=rayl_2d_run_w_sc_list, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_2d_run_wo_sc_list', data=rayl_2d_run_wo_sc_list, compression="gzip", compression_opts=9)

hf.create_dataset('rayl_2d_freq_w_sc', data=rayl_2d_freq_w_sc, compression="gzip", compression_opts=9)
hf.create_dataset('rayl_2d_freq_wo_sc', data=rayl_2d_freq_wo_sc, compression="gzip", compression_opts=9)

hf.create_dataset('sc_range', data=sc_range, compression="gzip", compression_opts=9)
hf.create_dataset('sc_bins', data=sc_bins, compression="gzip", compression_opts=9)
hf.create_dataset('sc_bin_center', data=sc_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('sc_2d_run', data=sc_2d_run, compression="gzip", compression_opts=9)
hf.create_dataset('sc_2d_run_list', data=sc_2d_run_list, compression="gzip", compression_opts=9)
hf.create_dataset('sc_2d_freq', data=sc_2d_freq, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB')
print('Done!!')



