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
from tools.antenna import antenna_info
from tools.qual import offset_block_error_check

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

# detector config
ant_num = antenna_info()[2]
trig_type = 3
qual_type = 3
config_type = np.arange(1,7)

# config array
config_arr = np.full((len(run_tot)),np.nan)
unix_arr = np.copy(config_arr)
date_arr = np.copy(config_arr)

# roll hist
mv_range = np.arange(-700,700)
mv_bins, mv_bin_center = bin_range_maker(mv_range, len(mv_range))
print(mv_bin_center.shape)

mv_2d_run = np.full((len(mv_bin_center),len(run_range), ant_num, trig_type, qual_type),np.nan)
mv_2d_vol = np.zeros((len(mv_bin_center),ant_num, trig_type, qual_type, len(config_type)))
print(mv_2d_run.shape)
print(mv_2d_vol.shape)

for r in tqdm(range(len(run_tot))):
  #if r<5: 
    rr = np.where(run_range == run_tot[r])[0][0]
   
    try: 
        # info
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print('OSERROR!',d_list[r])
        continue

    config_arr1 = hf['config'][2]
    config_arr[r] = config_arr1
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
        qual_pass[qual_pass == 1] = np.nan
        qual_pass[qual_pass == 0] = 1
        qual_cut = np.copy(qual_num)
        qual_cut[qual_pass == 0] = np.nan
        del qual_num

        roll_mm = hf['roll_mm'][:]
        roll_mv = offset_block_error_check(Station, unix_time, roll_mm)
        del roll_mm

        # qual
        roll_mv_pass = np.copy(roll_mv)
        roll_mv_pass *= qual_pass[np.newaxis, :]
        roll_mv_cut = np.copy(roll_mv)
        roll_mv_cut *= qual_cut[np.newaxis, :]
        del hf, qual_pass, qual_cut

        #trig
        rf_roll_mv = roll_mv[:,rf_trig]
        cal_roll_mv = roll_mv[:,cal_trig]
        soft_roll_mv = roll_mv[:,soft_trig]
        del roll_mv

        rf_roll_mv_pass = roll_mv_pass[:,rf_trig]
        cal_roll_mv_pass = roll_mv_pass[:,cal_trig]
        soft_roll_mv_pass = roll_mv_pass[:,soft_trig]
        del roll_mv_pass

        rf_roll_mv_cut = roll_mv_cut[:,rf_trig]
        cal_roll_mv_cut = roll_mv_cut[:,cal_trig]
        soft_roll_mv_cut = roll_mv_cut[:,soft_trig]
        del roll_mv_cut, rf_trig, cal_trig, soft_trig         

        for a in range(ant_num):

            rf_hist = np.histogram(rf_roll_mv[a], bins = mv_bins)[0]
            rf_hist_pass = np.histogram(rf_roll_mv_pass[a], bins = mv_bins)[0]
            rf_hist_cut = np.histogram(rf_roll_mv_cut[a], bins = mv_bins)[0]
            cal_hist = np.histogram(cal_roll_mv[a], bins = mv_bins)[0]
            cal_hist_pass = np.histogram(cal_roll_mv_pass[a], bins = mv_bins)[0]
            cal_hist_cut = np.histogram(cal_roll_mv_cut[a], bins = mv_bins)[0]
            soft_hist = np.histogram(soft_roll_mv[a], bins = mv_bins)[0]
            soft_hist_pass = np.histogram(soft_roll_mv_pass[a], bins = mv_bins)[0]
            soft_hist_cut = np.histogram(soft_roll_mv_cut[a], bins = mv_bins)[0]            

            mv_2d_run[:,rr,a,0,0] = rf_hist
            mv_2d_run[:,rr,a,0,1] = rf_hist_pass
            mv_2d_run[:,rr,a,0,2] = rf_hist_cut
            mv_2d_run[:,rr,a,1,0] = cal_hist
            mv_2d_run[:,rr,a,1,1] = cal_hist_pass
            mv_2d_run[:,rr,a,1,2] = cal_hist_cut
            mv_2d_run[:,rr,a,2,0] = soft_hist
            mv_2d_run[:,rr,a,2,1] = soft_hist_pass
            mv_2d_run[:,rr,a,2,2] = soft_hist_cut

            mv_2d_vol[:,a,0,0,config_arr1-1] += rf_hist
            mv_2d_vol[:,a,0,1,config_arr1-1] += rf_hist_pass
            mv_2d_vol[:,a,0,2,config_arr1-1] += rf_hist_cut
            mv_2d_vol[:,a,1,0,config_arr1-1] += cal_hist
            mv_2d_vol[:,a,1,1,config_arr1-1] += cal_hist_pass
            mv_2d_vol[:,a,1,2,config_arr1-1] += cal_hist_cut
            mv_2d_vol[:,a,2,0,config_arr1-1] += soft_hist
            mv_2d_vol[:,a,2,1,config_arr1-1] += soft_hist_pass
            mv_2d_vol[:,a,2,2,config_arr1-1] += soft_hist_cut

            del rf_hist, rf_hist_pass, rf_hist_cut, cal_hist, cal_hist_pass, cal_hist_cut, soft_hist, soft_hist_pass, soft_hist_cut
        del rf_roll_mv, cal_roll_mv, soft_roll_mv, rf_roll_mv_pass, cal_roll_mv_pass, soft_roll_mv_pass, rf_roll_mv_cut, cal_roll_mv_cut, soft_roll_mv_cut        


path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Offset_Hist_A{Station}.h5'
hf = h5py.File(file_name, 'w')

hf.create_dataset('run_tot', data=run_tot, compression="gzip", compression_opts=9)
hf.create_dataset('run_range', data=run_range, compression="gzip", compression_opts=9)
hf.create_dataset('bad_runs', data=bad_runs, compression="gzip", compression_opts=9)

hf.create_dataset('config_arr', data=config_arr, compression="gzip", compression_opts=9)
hf.create_dataset('unix_arr', data=unix_arr, compression="gzip", compression_opts=9)
hf.create_dataset('date_arr', data=date_arr, compression="gzip", compression_opts=9)

hf.create_dataset('mv_range', data=mv_range, compression="gzip", compression_opts=9)
hf.create_dataset('mv_bins', data=mv_bins, compression="gzip", compression_opts=9)
hf.create_dataset('mv_bin_center', data=mv_bin_center, compression="gzip", compression_opts=9)

hf.create_dataset('mv_2d_run', data=mv_2d_run, compression="gzip", compression_opts=9)
hf.create_dataset('mv_2d_vol', data=mv_2d_vol, compression="gzip", compression_opts=9)

hf.close()

print('file is in:',path+file_name)
file_size = np.round(os.path.getsize(path+file_name)/1204/1204,2)
print('file size is', file_size, 'MB') 
print('Done!!')





















