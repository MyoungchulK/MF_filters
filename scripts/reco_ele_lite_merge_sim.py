import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_example_run
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Type = str(sys.argv[2])

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/*{Type}*'
d_len = file_sorter(d_path)[-1]
del d_path 

r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_sim/'
rl_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_lite_sim/'

if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000
num_sols = 2
num_ants = 16
evt_num = np.arange(num_evts, dtype = int)
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
z = np.sin(np.radians(theta)) * rad[0]

flat_len = the_len * rad_len * sol_len
theta_ex = np.full((the_len, rad_len, sol_len), np.nan, dtype = float)
theta_ex[:] = theta[:, np.newaxis, np.newaxis]
theta_flat = np.reshape(theta_ex, (flat_len))
rad_ex = np.full((the_len, rad_len, sol_len), np.nan, dtype = float)
rad_ex[:] = rad[np.newaxis, :, np.newaxis]
rad_flat = np.reshape(rad_ex, (flat_len))
z_flat = np.sin(np.radians(theta_flat)) * rad_flat
del theta_ex, rad_ex

sur_ang = np.array([180, 180, 37, 24, 17], dtype = float)
theta_map = np.full((pol_len, the_len, rad_len, sol_len), np.nan, dtype = float)
theta_map[:] = theta[np.newaxis, :, np.newaxis, np.newaxis]
sur_bool = theta_map <= sur_ang[np.newaxis, np.newaxis, :, np.newaxis]
sur_bool_flat = np.reshape(sur_bool, (pol_len, flat_len))
del sur_ang, theta_map, sur_bool

# temp
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

fla_r = np.arange(3, dtype = int) + 1
en_r = np.arange(16, 21, dtype = int)
con_r = np.arange(num_configs, dtype = int) + 1

if Type == 'signal':
    sim_r = np.arange(80, dtype = int)
    run_map = np.full((4, d_len), 0, dtype = int)
    counts = 0
    for e in range(len(en_r)):
        for f in range(len(fla_r)):
            for c in range(num_configs):
                for r in range(len(sim_r)):
                    run_map[:, counts] = np.array([en_r[e], fla_r[f], con_r[c], sim_r[r]], dtype = int)
                    counts += 1
if Type == 'noise':
    sim_r = np.arange(1000, dtype = int)
    run_map = np.full((2, d_len), 0, dtype = int)
    counts = 0
    for c in range(num_configs):
        for r in range(len(sim_r)):
            run_map[:, counts] = np.array([con_r[c], sim_r[r]], dtype = int)
            counts += 1
print(run_map.shape)
print(run_map)

for r in tqdm(range(d_len)):
    
  #if r > 5057:

    if Type == 'signal':
        hf_name = f'_AraOut.{Type}_E{run_map[0, r]}_F{run_map[1, r]}_A{Station}_R{run_map[2, r]}.txt.run{run_map[3, r]}.h5'
    if Type == 'noise':
        hf_name = f'_AraOut.{Type}_A{Station}_R{run_map[0, r]}.txt.run{run_map[1, r]}.h5'

    try:
        hf = h5py.File(f'{r_path}reco_ele{hf_name}', 'r')
        entry_num = hf['entry_num'][:]
        coef_tot = hf['coef'][:] # pol, theta, rad, sol, evt
        coord_tot = hf['coord'][:] # pol, theta, rad, sol, evt
        coef_tot[np.isnan(coef_tot)] = -1
        del hf
        coef_re = np.reshape(coef_tot, (pol_len, flat_len, -1))
        coord_re = np.reshape(coord_tot, (pol_len, flat_len, -1))

        coef_tot1 = np.copy(coef_tot[:, :, 0, 0, :]) # pol, theta, (rad), (sol), evt
        coord_tot1 = np.copy(coord_tot[:, :, 0, 0, :]) # pol, theta, (rad), (sol), evt
        del coef_tot, coord_tot

        coef_idx = np.nanargmax(coef_tot1, axis = 1)
        coef_cal = coef_tot1[pol_num[:, np.newaxis], coef_idx, evt_num[np.newaxis, :]] # pol, evt
        neg_idx = coef_cal < 0
        coef_cal[neg_idx] = np.nan
        del coef_tot1

        coord_cal = np.full((ang_len + 1, pol_len, num_evts), np.nan, dtype = float) # thepiz, pol, evt
        coord_cal[0] = theta[coef_idx]
        coord_cal[1] = coord_tot1[pol_num[:, np.newaxis], coef_idx, evt_num[np.newaxis, :]]
        coord_cal[2] = z[coef_idx]
        coord_cal[:, neg_idx] = np.nan
        del coord_tot1, coef_idx, neg_idx

        if not os.path.exists(rl_path):
            os.makedirs(rl_path)
        hf_l = h5py.File(f'{rl_path}reco_ele_lite{hf_name}', 'w')
        hf_l.create_dataset('entry_num', data=entry_num, compression="gzip", compression_opts=9)
        hf_l.create_dataset('coef_cal', data=coef_cal, compression="gzip", compression_opts=9)
        hf_l.create_dataset('coord_cal', data=coord_cal, compression="gzip", compression_opts=9)
        del entry_num
 
        for t in range(2):
            if t == 1:   
                coef_re[sur_bool_flat] = -1
                coord_re[sur_bool_flat] = np.nan
            coef_max_idx = np.nanargmax(coef_re, axis = 1)
            coef_max1 = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]] # pol, ev
            neg_idx = coef_max1 < 0
            coef_max1[neg_idx] = np.nan
            coord_max1 = np.full((ang_len + 2, pol_len, num_evts), np.nan, dtype = float) # thepir, pol, evt
            coord_max1[0] = theta_flat[coef_max_idx]
            coord_max1[1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
            coord_max1[2] = rad_flat[coef_max_idx]
            coord_max1[3] = z_flat[coef_max_idx]
            coord_max1[:, neg_idx] = np.nan
            del coef_max_idx, neg_idx
            if t == 0:
                hf_l.create_dataset('coef_max', data=coef_max1, compression="gzip", compression_opts=9)
                hf_l.create_dataset('coord_max', data=coord_max1, compression="gzip", compression_opts=9)
            else:
                hf_l.create_dataset('coef_s_max', data=coef_max1, compression="gzip", compression_opts=9)
                hf_l.create_dataset('coord_s_max', data=coord_max1, compression="gzip", compression_opts=9)
            del coef_max1, coord_max1
        del coef_re, coord_re
        hf_l.close()
    except FileNotFoundError:
        print(f'{r_path}reco_ele{hf_name}')

print('done!')



