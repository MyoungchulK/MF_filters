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
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

i_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/'
s_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_sim/'
sb_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_banila_sim/'
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_sim/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_sim/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_sim/'

if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000
num_ants = 16
evt_num = np.arange(num_evts, dtype = int)
pol_num = np.arange(3, dtype = int)
ang_num = np.arange(2, dtype = int)
rad_num = np.arange(3, dtype = int)
sur_cut = 35
rad_o = np.array([41, 170, 300], dtype = float)
nfour = float(2048 / 2 / 2 * 0.5)

sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
radius = np.full((d_len), np.nan, dtype = float)
inu_thrown = np.copy(radius)
coef = np.full((d_len, len(pol_num), len(rad_num), 2, num_evts), np.nan, dtype = float) # run, pol, rad, sol, evt
coord = np.full((d_len, len(pol_num), len(ang_num), len(rad_num), 2, num_evts), np.nan, dtype = float) # run, pol, thephi, rad, sol, evt
coef_max = np.full((d_len, len(pol_num), num_evts), np.nan, dtype = float) # run, pol, evt
coord_max = np.full((d_len, len(ang_num) + 1, len(pol_num), num_evts), np.nan, dtype = float) # run, pol, thephir, evt
mf_max = np.full((d_len, len(pol_num), num_evts), np.nan, dtype = float) # run, pol, evt
mf_max_each = np.full((d_len, len(pol_num), 2, 7, 6, num_evts), np.nan, dtype = float) # run, pol, sho, the, off, evt
mf_temp = np.full((d_len, len(pol_num), 2, num_evts), np.nan, dtype = float) # run, pol, thephi, evt
snr = np.full((d_len, num_ants, num_evts), np.nan, dtype = float)
snr_all = np.copy(snr)
snr_max = np.full((d_len, 3, num_evts), np.nan, dtype = float)
snr_b = np.copy(snr)
snr_b_all = np.copy(snr)
snr_b_max = np.copy(snr_max)
qual_indi = np.full((d_len, num_evts, 9), 0, dtype = int)
qual_tot = np.full((d_len, num_evts), 0, dtype = int)
evt_rate = np.full((d_len, num_evts), np.nan, dtype = float)
one_weight = np.full((d_len, num_evts), np.nan, dtype = float)
nan_flag = np.full((d_len, 2, 2, num_evts), 0, dtype = int)

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
    
  #if r < 10:

    if Type == 'signal':
        hf_name = f'_AraOut.{Type}_E{run_map[0, r]}_F{run_map[1, r]}_A{Station}_R{run_map[2, r]}.txt.run{run_map[3, r]}.h5'
    if Type == 'noise':
        hf_name = f'_AraOut.{Type}_A{Station}_R{run_map[0, r]}.txt.run{run_map[1, r]}.h5'
    try:
        hf = h5py.File(f'{i_path}sub_info{hf_name}', 'r')
    except FileNotFoundError:
        print(f'{i_path}sub_info{hf_name}')
        continue
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    radius[r] = hf['radius'][:]
    inu_thrown[r] = hf['inu_thrown'][-1]
    del hf

    try:
        hf = h5py.File(f'{q_path}qual_cut{hf_name}', 'r')
        qual_tot[r] = (hf['tot_qual_cut_sum'][:] != 0).astype(int)
        qual_indi[r] = (hf['tot_qual_cut'][:] != 0).astype(int)
        evt_rate[r] = hf['evt_rate'][:]
        one_weight[r] = hf['one_weight'][:]
        del hf
    except FileNotFoundError:
        print(f'{q_path}qual_cut{hf_name}')

    ex_run = get_example_run(Station, config[r])
    bad_ant = known_issue.get_bad_antenna(ex_run)
    try:
        hf = h5py.File(f'{s_path}snr{hf_name}', 'r')
        snr_tot = hf['snr'][:]
        snr_all[r] = snr_tot
        snr_tot[bad_ant] = np.nan
        snr[r] = snr_tot
        snr_max[r, 0] = -np.sort(-snr_tot[:8], axis = 0)[2]
        snr_max[r, 1] = -np.sort(-snr_tot[8:], axis = 0)[2]
        snr_max[r, 2] = -np.sort(-snr_tot, axis = 0)[2]
        del hf, snr_tot
    except FileNotFoundError:
        print(f'{s_path}snr{hf_name}')

    try:
        hf = h5py.File(f'{sb_path}snr_banila{hf_name}', 'r')
        snr_tot = hf['snr'][:]
        snr_b_all[r] = snr_tot
        snr_tot[bad_ant] = np.nan   
        snr_b[r] = snr_tot
        snr_b_max[r, 0] = -np.sort(-snr_tot[:8], axis = 0)[2]
        snr_b_max[r, 1] = -np.sort(-snr_tot[8:], axis = 0)[2]
        snr_b_max[r, 2] = -np.sort(-snr_tot, axis = 0)[2]
        del hf, snr_tot
    except FileNotFoundError:
        print(f'{sb_path}snr_banila{hf_name}')

    try:
        hf = h5py.File(f'{r_path}reco{hf_name}', 'r')
        coef_tot = hf['coef'][:] # pol, rad, sol, evt
        coord_tot = hf['coord'][:] # pol, tp, rad, sol, evt
        coef[r] = coef_tot
        coord[r] = coord_tot
        coef_re = np.reshape(coef_tot, (3, 6, -1))
        coord_re = np.reshape(coord_tot, (3, 2, 6, -1))
        coord_re = np.transpose(coord_re, (1, 0, 2, 3))
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_max[r] = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_max[r, :2] = coord_re[ang_num[:, np.newaxis, np.newaxis], pol_num[np.newaxis, :, np.newaxis], coef_max_idx, evt_num[np.newaxis, np.newaxis, :]]
        coord_max[r, 2] = rad_o[coef_max_idx // 3]
        del hf, coef_tot, coord_tot, coef_re, coord_re, coef_max_idx
    except FileNotFoundError:
        print(f'{r_path}reco{hf_name}')

    try:
        hf = h5py.File(f'{m_path}mf{hf_name}', 'r')
        mf_max[r] = hf['mf_max'][:] # pol, evt
        mf_max_each[r] = hf['mf_max_each'][:] # pol, sho, the, off, evt
        mf_temp[r, :2] = hf['mf_temp'][:, 1:3] # of pols, theta n phi, # of evts
        mf_temp[r, 2] = hf['mf_temp_com'][1:3] # of pols, theta n phi, # of evts
        del hf
    except FileNotFoundError:
        print(f'{m_path}mf{hf_name}')
    del hf_name


qual = (np.nansum(qual_indi[:, :, 0:2], axis = 2) != 0).astype(int)
qual_cw = np.copy(qual_indi[:, :, 2])
qual_op = np.copy(qual_indi[:, :, 3])
qual_cp = np.copy(qual_indi[:, :, 4])
qual_corr = np.copy(qual_indi[:, :, 5])
qual_ver = (np.nansum(qual_indi[:, :, 6:8], axis = 2) != 0).astype(int)
qual_mf = np.copy(qual_indi[:, :, 8])

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_{Type}_v11_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('run_map', data=run_map, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown', data=inu_thrown, compression="gzip", compression_opts=9)
hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight', data=one_weight, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max_each', data=mf_max_each, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.create_dataset('snr', data=snr, compression="gzip", compression_opts=9)
hf.create_dataset('snr_all', data=snr_all, compression="gzip", compression_opts=9)
hf.create_dataset('snr_b', data=snr_b, compression="gzip", compression_opts=9)
hf.create_dataset('snr_b_all', data=snr_b_all, compression="gzip", compression_opts=9)
hf.create_dataset('snr_max', data=snr_max, compression="gzip", compression_opts=9)
hf.create_dataset('snr_b_max', data=snr_b_max, compression="gzip", compression_opts=9)
hf.create_dataset('qual_indi', data=qual_indi, compression="gzip", compression_opts=9)
hf.create_dataset('qual', data=qual, compression="gzip", compression_opts=9)
hf.create_dataset('qual_cw', data=qual_cw, compression="gzip", compression_opts=9)
hf.create_dataset('qual_op', data=qual_op, compression="gzip", compression_opts=9)
hf.create_dataset('qual_cp', data=qual_cp, compression="gzip", compression_opts=9)
hf.create_dataset('qual_corr', data=qual_corr, compression="gzip", compression_opts=9)
hf.create_dataset('qual_ver', data=qual_ver, compression="gzip", compression_opts=9)
hf.create_dataset('qual_mf', data=qual_mf, compression="gzip", compression_opts=9)
hf.create_dataset('qual_tot', data=qual_tot, compression="gzip", compression_opts=9)
hf.create_dataset('nan_flag', data=nan_flag, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




