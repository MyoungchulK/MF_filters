import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf/'
del d_run_range

ang_num = np.arange(2, dtype = int)
ang_len = len(ang_num)
pol_num = np.arange(2, dtype = int)
pol_len = len(pol_num)
sol_num = np.arange(3, dtype = int)
sol_len = len(sol_num)
rad = np.array([41, 170, 300, 450, 600], dtype = float)
rad_len = len(rad)
rad_num = np.arange(rad_len, dtype = int)
theta = 90 - np.linspace(0.5, 179.5, 179 + 1)
the_len = len(theta)
flat_len = the_len * rad_len * sol_len
theta_ex = np.full((the_len, rad_len, sol_len), np.nan, dtype = float)
theta_ex[:] = theta[:, np.newaxis, np.newaxis]
theta_flat = np.reshape(theta_ex, (flat_len))
rad_ex = np.full((the_len, rad_len, sol_len), np.nan, dtype = float)
rad_ex[:] = rad[np.newaxis, :, np.newaxis]
rad_flat = np.reshape(rad_ex, (flat_len))

sur_ang = np.array([180, 180, 37, 24, 17], dtype = float)
theta_map = np.full((pol_len, the_len, rad_len, sol_len), np.nan, dtype = float)
theta_map[:] = theta[np.newaxis, :, np.newaxis, np.newaxis]
sur_bool = theta_map <= sur_ang[np.newaxis, np.newaxis, :, np.newaxis]
sur_bool_flat = np.reshape(sur_bool, (pol_len, flat_len))

runs = np.copy(d_run_tot)
configs = np.full((d_len), 0, dtype = int)
b_runs = np.in1d(runs, bad_runs).astype(int)
run_ep = np.full((0), 0, dtype = int)
evt_ep = np.copy(run_ep)
trig_ep = np.copy(run_ep)
con_ep = np.copy(run_ep)
coef_r_max = np.full((pol_len, rad_len, sol_len,0), 0, dtype = float) # pol, rad, sol, evt
coord_r_max = np.full((ang_len, pol_len, rad_len, sol_len, 0), 0, dtype = float) # thepi, pol, rad, sol, evt
coef_max = np.full((pol_len, 0), 0, dtype = float) # pol, evt
coord_max = np.full((ang_len + 1, pol_len, 0), 0, dtype = float) # thepir, pol, evt
coef_s_max = np.copy(coef_max)
coord_s_max = np.copy(coord_max)
mf_max = np.copy(coef_max) # pols, evts
mf_temp = np.full((pol_len, ang_len, 0), 0, dtype = float) # pols, thephi, evts 

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    configs[r] = hf['config'][2]
    evt = hf['evt_num'][:]
    trig_type = hf['trig_type'][:]
    num_evts = len(evt)
    evt_num = np.arange(num_evts, dtype = int)
    run_r = np.full((num_evts), d_run_tot[r], dtype = int)
    con_r = np.full((num_evts), configs[r], dtype = int)
    run_ep = np.concatenate((run_ep, run_r))
    evt_ep = np.concatenate((evt_ep, evt))
    trig_ep = np.concatenate((trig_ep, trig_type))
    con_ep = np.concatenate((con_ep, con_r))
    del trig_type, run_r, con_r, evt 

    coef_tot = hf['coef'][:] # pol, theta, rad, sol, evt
    coord_tot = hf['coord'][:] # pol, theta, rad, sol, evt

    coef_r_max_idx = np.nanargmax(coef_tot, axis = 1) # pol, rad, ray, evt
    coef_r_max1 = coef_tot[pol_num[:, np.newaxis, np.newaxis, np.newaxis], coef_r_max_idx, rad_num[np.newaxis, :, np.newaxis, np.newaxis], sol_num[np.newaxis, np.newaxis, :, np.newaxis], evt_num[np.newaxis, np.newaxis, np.newaxis, :]] # pol, rad, ray, evt
    coord_r_max1 = np.full((ang_len, pol_len, rad_len, sol_len, num_evts), np.nan, dtype = float) # thepi, pol, rad, ray, evt
    coord_r_max1[0] = theta[coef_r_max_idx]
    coord_r_max1[1] = coord_tot[pol_num[:, np.newaxis, np.newaxis, np.newaxis], coef_r_max_idx, rad_num[np.newaxis, :, np.newaxis, np.newaxis], sol_num[np.newaxis, np.newaxis, :, np.newaxis], evt_num[np.newaxis, np.newaxis, np.newaxis, :]]
    coef_r_max = np.concatenate((coef_r_max, coef_r_max1), axis = 3)
    coord_r_max = np.concatenate((coord_r_max, coord_r_max1), axis = 4)
    del coef_r_max_idx, coef_r_max1, coord_r_max1

    coef_re = np.reshape(coef_tot, (pol_len, flat_len, -1))
    coord_re = np.reshape(coord_tot, (pol_len, flat_len, -1))
    coef_max_idx = np.nanargmax(coef_re, axis = 1)
    coef_max1 = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]] # pol, evt
    coord_max1 = np.full((ang_len + 1, pol_len, num_evts), np.nan, dtype = float) # thepir, pol, evt
    coord_max1[0] = theta_flat[coef_max_idx]
    coord_max1[1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
    coord_max1[2] = rad_flat[coef_max_idx]
    coef_max = np.concatenate((coef_max, coef_max1), axis = 1)
    coord_max = np.concatenate((coord_max, coord_max1), axis = 2)
    del coef_max1, coord_max1, 

    sur_bool_flat_ex = np.repeat(sur_bool_flat[:, :, np.newaxis], num_evts, axis = 2)
    coef_re[sur_bool_flat_ex] = np.nan
    coord_re[sur_bool_flat_ex] = np.nan
    coef_max_idx = np.nanargmax(coef_re, axis = 1)
    coef_max2 = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]] # pol, evt
    coord_max2 = np.full((ang_len + 1, pol_len, num_evts), np.nan, dtype = float) # thepir, pol, evt
    coord_max2[0] = theta_flat[coef_max_idx]
    coord_max2[1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
    coord_max2[2] = rad_flat[coef_max_idx]
    coef_s_max = np.concatenate((coef_s_max, coef_max2), axis = 1)
    coord_s_max = np.concatenate((coord_s_max, coord_max2), axis = 2)
    del coef_re, coord_re, coef_max_idx, coef_max2, coord_max2, sur_bool_flat_ex
    del coef_tot, coord_tot, evt_num

    m_name = f'{m_path}mf_A{Station}_R{d_run_tot[r]}.h5'
    hf = h5py.File(m_name, 'r')
    mf_m = hf['mf_max'][:pol_len]
    mf_t_p = hf['mf_temp'][:, 1:3]
    #mf_t_c = hf['mf_temp_com'][1:3] # of pols, theta n phi, # of evts
    mf_max = np.concatenate((mf_max, mf_m), axis = 1)
    mf_temp = np.concatenate((mf_temp, mf_t_p), axis = 2) 
    del m_name, hf, mf_m, mf_t, mf_t_p, mf_t_c
    del num_evts
    
print(coef_max.shape)
print(coord_max.shape)
print(mf_max.shape)
print(mf_temp.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Data_Summary_v14_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('runs', data=runs, compression="gzip", compression_opts=9)
hf.create_dataset('b_runs', data=b_runs, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('run_ep', data=run_ep, compression="gzip", compression_opts=9)
hf.create_dataset('evt_ep', data=evt_ep, compression="gzip", compression_opts=9)
hf.create_dataset('trig_ep', data=trig_ep, compression="gzip", compression_opts=9)
hf.create_dataset('con_ep', data=con_ep, compression="gzip", compression_opts=9)
hf.create_dataset('coef_r_max', data=coef_r_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_r_max', data=coord_r_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_s_max', data=coef_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_s_max', data=coord_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






