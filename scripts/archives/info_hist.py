import numpy as np
import os, sys
import re
from glob import glob
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import bad_run
from tools.run import bad_surface_run

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

# data sort
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/*'
d_list = glob(d_path)
d_len = len(d_list)
print(d_len)
run_tot=np.full((d_len),np.nan,dtype=int)
aa = 0
for d in d_list:
    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    aa += 1
del aa, d_path

# detector config
ant_num = 16
trig_type = 3
config_type = 6

# wf len
wf_len_bins = np.linspace(0, 3000, 3000+1)
wf_len_bin_center = (wf_len_bins[1:] + wf_len_bins[:-1]) * 0.5
wf_len = np.zeros((len(wf_len_bin_center), ant_num, trig_type, config_type))

# wf if point
wf_if_bins = np.linspace(-1500, 1500, 6000+1)
wf_if_bin_center = (wf_if_bins[1:] + wf_if_bins[:-1]) * 0.5
wf_if = np.zeros((len(wf_if_bin_center), 2, ant_num, trig_type, config_type))

# peak
peak_bins = np.linspace(0, 100, 10000+1)
peak_bin_center = (peak_bins[1:] + peak_bins[:-1]) * 0.5
peak = np.zeros((len(peak_bin_center), ant_num, trig_type, config_type))

# rms
rms_bins = np.linspace(0, 100, 10000+1)
rms_bin_center = (rms_bins[1:] + rms_bins[:-1]) * 0.5
rms = np.zeros((len(rms_bin_center), ant_num, trig_type, config_type))

# read win
read_win_bins = np.linspace(0, 1000, 1000+1)
read_win_bin_center = (read_win_bins[1:] + read_win_bins[:-1]) * 0.5
read_win = np.zeros((len(read_win_bin_center), trig_type, config_type))

dd = 0
ddd = 0
for d in d_list:

    if run_tot[dd] in bad_runs:
        print('bad run:',d,run_tot[dd])
    else:
        #print(d)
        hf = h5py.File(d, 'r')

        config  = hf['config'][2]

        trig_num = hf['trig_num'][:]
        rf_trig = np.where(trig_num == 0)[0]
        cal_trig = np.where(trig_num == 1)[0]
        soft_trig = np.where(trig_num == 2)[0]

        qual_num = hf['qual_num'][:].astype(float)
        qual_num[qual_num == 0] = np.nan

        wf_len_all = hf['wf_len_all'][:] * qual_num[np.newaxis, :]
        wf_if_all = hf['wf_if_all'][:] * qual_num[np.newaxis, np.newaxis, :]
        peak_all = hf['peak_all'][:] * qual_num[np.newaxis, :]
        rms_all = hf['rms_all'][:] * qual_num[np.newaxis, :]
        read_win_all = hf['read_win'][:] * qual_num

        # wf len
        rf_len_arr = wf_len_all[:,rf_trig]
        cal_len_arr = wf_len_all[:,cal_trig]
        soft_len_arr = wf_len_all[:,soft_trig]

        # wf if
        rf_if_arr = wf_if_all[:,:,rf_trig]
        cal_if_arr = wf_if_all[:,:,cal_trig]
        soft_if_arr = wf_if_all[:,:,soft_trig]

        # wf peak
        rf_peak_arr = peak_all[:,rf_trig]
        cal_peak_arr = peak_all[:,cal_trig]
        soft_peak_arr = peak_all[:,soft_trig]

        # rms peak
        rf_rms_arr = rms_all[:,rf_trig]
        cal_rms_arr = rms_all[:,cal_trig]
        soft_rms_arr = rms_all[:,soft_trig]

        # read win
        rf_read_win_arr = read_win_all[rf_trig]
        cal_read_win_arr = read_win_all[cal_trig]
        soft_read_win_arr = read_win_all[soft_trig]        

        for a in range(ant_num):

            wf_len[:,a,0,config] += np.histogram(rf_len_arr[a], bins = wf_len_bins)[0]
            wf_len[:,a,1,config] += np.histogram(cal_len_arr[a], bins = wf_len_bins)[0]
            wf_len[:,a,2,config] += np.histogram(soft_len_arr[a], bins = wf_len_bins)[0] 

            wf_if[:,0,a,0,config] += np.histogram(rf_if_arr[0,a], bins = wf_if_bins)[0]
            wf_if[:,0,a,1,config] += np.histogram(cal_if_arr[0,a], bins = wf_if_bins)[0]
            wf_if[:,0,a,2,config] += np.histogram(soft_if_arr[0,a], bins = wf_if_bins)[0]

            wf_if[:,1,a,0,config] += np.histogram(rf_if_arr[1,a], bins = wf_if_bins)[0]
            wf_if[:,1,a,1,config] += np.histogram(cal_if_arr[1,a], bins = wf_if_bins)[0]
            wf_if[:,1,a,2,config] += np.histogram(soft_if_arr[1,a], bins = wf_if_bins)[0]

            peak[:,a,0,config] += np.histogram(rf_peak_arr[a], bins = peak_bins)[0]
            peak[:,a,1,config] += np.histogram(cal_peak_arr[a], bins = peak_bins)[0]
            peak[:,a,2,config] += np.histogram(soft_peak_arr[a], bins = peak_bins)[0]

            rms[:,a,0,config] += np.histogram(rf_rms_arr[a], bins = rms_bins)[0]
            rms[:,a,1,config] += np.histogram(cal_rms_arr[a], bins = rms_bins)[0]
            rms[:,a,2,config] += np.histogram(soft_rms_arr[a], bins = rms_bins)[0]

        read_win[:,0,config] += np.histogram(rf_read_win_arr, bins = read_win_bins)[0]
        read_win[:,1,config] += np.histogram(cal_read_win_arr, bins = read_win_bins)[0]
        read_win[:,2,config] += np.histogram(soft_read_win_arr, bins = read_win_bins)[0]

        del hf, config, wf_len_all, trig_num, qual_num, rf_trig, cal_trig, soft_trig, rf_len_arr, cal_len_arr, soft_len_arr
        del wf_if_all, peak_all, rms_all, rf_if_arr, cal_if_arr, soft_if_arr, rf_peak_arr, cal_peak_arr, soft_peak_arr
        del rf_rms_arr, cal_rms_arr, soft_rms_arr, read_win_all, rf_read_win_arr, cal_read_win_arr, soft_read_win_arr
        ddd+=1      

    print('Progress:',np.round(dd/d_len*100,2),'%')
    dd+=1

print(dd)
print(ddd)

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Info_Hist_A{Station}.h5', 'w')
hf.create_dataset('tot_run', data=np.array([ddd]), compression="gzip", compression_opts=9)
hf.create_dataset('ant_num', data=np.array([ant_num]), compression="gzip", compression_opts=9)

del trig_type, ant_num

hf.create_dataset('wf_len_bins', data=wf_len_bins * 0.5, compression="gzip", compression_opts=9)
hf.create_dataset('wf_len_bin_center', data=wf_len_bin_center * 0.5, compression="gzip", compression_opts=9)
hf.create_dataset('wf_len', data=wf_len, compression="gzip", compression_opts=9)

hf.create_dataset('wf_if_bins', data=wf_if_bins, compression="gzip", compression_opts=9)
hf.create_dataset('wf_if_bin_center', data=wf_if_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('wf_if', data=wf_if, compression="gzip", compression_opts=9)

hf.create_dataset('peak_bins', data=peak_bins, compression="gzip", compression_opts=9)
hf.create_dataset('peak_bin_center', data=peak_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('peak', data=peak, compression="gzip", compression_opts=9)

hf.create_dataset('rms_bins', data=rms_bins, compression="gzip", compression_opts=9)
hf.create_dataset('rms_bin_center', data=rms_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('rms', data=rms, compression="gzip", compression_opts=9)

hf.create_dataset('read_win_bins', data=read_win_bins, compression="gzip", compression_opts=9)
hf.create_dataset('read_win_bin_center', data=read_win_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('read_win', data=read_win, compression="gzip", compression_opts=9)

hf.close()

print('Done!!')





















