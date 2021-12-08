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
from tools.qual_temp import offset_block_error_check

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

# info data
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Info/*'
d_list_chaos = glob(d_path)
d_len = len(d_list_chaos)
print(d_len)
run_tot=np.full((d_len),np.nan,dtype=int)
aa = 0
for d in d_list_chaos:
    run_tot[aa] = int(re.sub("\D", "", d[-8:-1]))
    aa += 1
del aa, d_path

run_index = np.argsort(run_tot)
run_tot = run_tot[run_index]
d_list = []
for d in range(d_len):
    d_list.append(d_list_chaos[run_index[d]])
print(run_tot)
del d_list_chaos, d_len



# detector config
ant_num = 16
trig_type = 3
qual_type = 3

# run bin
run_bins = np.linspace(0, len(run_tot), len(run_tot)+1)
print(run_bins)
run_bin_center = (run_bins[1:] + run_bins[:-1]) * 0.5
print(run_bin_center)

# roll hist
roll_hist_bins = np.linspace(-750, 750, 1500+1)
roll_hist_bin_center = (roll_hist_bins[1:] + roll_hist_bins[:-1]) * 0.5
roll_hist = np.zeros((len(roll_hist_bin_center), ant_num, trig_type, qual_type))
roll_hist_2d = np.zeros((len(roll_hist_bin_center), len(run_bin_center), ant_num, trig_type, qual_type))
print(roll_hist.shape)
print(roll_hist_2d.shape)

#config
config_arr = np.full((len(run_bin_center)),np.nan)
print(config_arr.shape)

#unix time
unix_arr = np.full((len(run_bin_center)),np.nan)
date_arr = np.full((len(run_bin_center)),np.nan)
print(unix_arr.shape)

for r in tqdm(range(len(run_tot))):

    try:
        # info
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print('OSERROR!',d_list[r])
        continue

    config_arr[r] = hf['config'][2]
    unix_time = hf['unix_time'][0]
    unix_arr[r] = unix_time[0]
    date_time = datetime.fromtimestamp(unix_time[0])
    date_time1 = date_time.strftime('%Y%m%d%H%M%S')
    date_arr[r] = int(date_time1)
    del date_time, date_time1

    if run_tot[r] in bad_runs:
        print('bad run:',d_list[r],run_tot[r])
        del hf
        continue
    else:

        trig_num = hf['trig_num'][:]
        rf_trig = np.where(trig_num == 0)[0]
        cal_trig = np.where(trig_num == 1)[0]
        soft_trig = np.where(trig_num == 2)[0]
        del trig_num

        qual_num = hf['qual_num_pyroot'][:].astype(float)
        qual_pass = np.copy(qual_num)
        qual_pass[qual_pass == 0] = np.nan
        qual_cut = np.copy(qual_num)
        qual_cut[qual_cut == 1] = np.nan
        qual_cut[qual_cut == 0] = 1
        del qual_num

        roll_mm = hf['roll_mm'][:]
        roll_v = offset_block_error_check(Station, unix_time, roll_mm)
        del roll_mm

        # qual
        roll_max_mv_pass = np.copy(roll_v)
        roll_max_mv_pass *= qual_pass[np.newaxis, :]
        roll_max_mv_cut = np.copy(roll_v) 
        roll_max_mv_cut *= qual_cut[np.newaxis, :]
        del hf, qual_pass, qual_cut

        #trig
        rf_roll_max_mv = roll_v[:,rf_trig]
        cal_roll_max_mv = roll_v[:,cal_trig]
        soft_roll_max_mv = roll_v[:,soft_trig]
        del roll_v

        rf_roll_max_mv_pass = roll_max_mv_pass[:,rf_trig]
        cal_roll_max_mv_pass = roll_max_mv_pass[:,cal_trig]
        soft_roll_max_mv_pass = roll_max_mv_pass[:,soft_trig]
        del roll_max_mv_pass

        rf_roll_max_mv_cut = roll_max_mv_cut[:,rf_trig]
        cal_roll_max_mv_cut = roll_max_mv_cut[:,cal_trig]
        soft_roll_max_mv_cut = roll_max_mv_cut[:,soft_trig]
        del roll_max_mv_cut, rf_trig, cal_trig, soft_trig

        for a in range(ant_num):

            rf_hist = np.histogram(rf_roll_max_mv[a], bins = roll_hist_bins)[0]
            rf_hist_pass = np.histogram(rf_roll_max_mv_pass[a], bins = roll_hist_bins)[0]
            rf_hist_cut = np.histogram(rf_roll_max_mv_cut[a], bins = roll_hist_bins)[0]
            cal_hist = np.histogram(cal_roll_max_mv[a], bins = roll_hist_bins)[0]
            cal_hist_pass = np.histogram(cal_roll_max_mv_pass[a], bins = roll_hist_bins)[0]
            cal_hist_cut = np.histogram(cal_roll_max_mv_cut[a], bins = roll_hist_bins)[0]
            soft_hist = np.histogram(soft_roll_max_mv[a], bins = roll_hist_bins)[0]
            soft_hist_pass = np.histogram(soft_roll_max_mv_pass[a], bins = roll_hist_bins)[0]
            soft_hist_cut = np.histogram(soft_roll_max_mv_cut[a], bins = roll_hist_bins)[0]
            
            roll_hist[:,a,0,0] += rf_hist
            roll_hist[:,a,0,1] += rf_hist_pass
            roll_hist[:,a,0,2] += rf_hist_cut
            roll_hist[:,a,1,0] += cal_hist
            roll_hist[:,a,1,1] += cal_hist_pass
            roll_hist[:,a,1,2] += cal_hist_cut
            roll_hist[:,a,2,0] += soft_hist
            roll_hist[:,a,2,1] += soft_hist_pass
            roll_hist[:,a,2,2] += soft_hist_cut
            
            roll_hist_2d[:,r,a,0,0] += rf_hist
            roll_hist_2d[:,r,a,0,1] += rf_hist_pass
            roll_hist_2d[:,r,a,0,2] += rf_hist_cut
            roll_hist_2d[:,r,a,1,0] += cal_hist
            roll_hist_2d[:,r,a,1,1] += cal_hist_pass
            roll_hist_2d[:,r,a,1,2] += cal_hist_cut
            roll_hist_2d[:,r,a,2,0] += soft_hist
            roll_hist_2d[:,r,a,2,1] += soft_hist_pass
            roll_hist_2d[:,r,a,2,2] += soft_hist_cut

            del rf_hist, rf_hist_pass, rf_hist_cut, cal_hist, cal_hist_pass, cal_hist_cut, soft_hist, soft_hist_pass, soft_hist_cut

        del rf_roll_max_mv, rf_roll_max_mv_pass, rf_roll_max_mv_cut, cal_roll_max_mv, cal_roll_max_mv_pass, cal_roll_max_mv_cut, soft_roll_max_mv, soft_roll_max_mv_pass, soft_roll_max_mv_cut


path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Offset_Hist_Test_A{Station}.h5', 'w')
hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)
hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('date_arr', data=date_arr, compression="gzip", compression_opts=9)
hf.create_dataset('run_bins', data=run_bins, compression="gzip", compression_opts=9)
hf.create_dataset('run_bin_center', data=run_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('roll_hist_bins', data=roll_hist_bins, compression="gzip", compression_opts=9)
hf.create_dataset('roll_hist_bin_center', data=roll_hist_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('roll_hist', data=roll_hist, compression="gzip", compression_opts=9)
hf.create_dataset('roll_hist_2d', data=roll_hist_2d, compression="gzip", compression_opts=9)

hf.close()
print('Done!!')





















