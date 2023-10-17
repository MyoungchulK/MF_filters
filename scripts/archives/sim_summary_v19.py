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
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_sim/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_sim/'
sb_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_banila_sim/'

if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000
num_sols = 2
num_ants = 16
evt_num = np.arange(num_evts, dtype = int)
pol_num = np.arange(2, dtype = int)
pol_len = len(pol_num)
ang_num = np.arange(2, dtype = int)
rad = np.array([41, 170, 300, 450, 600], dtype = float)
rad_len = len(rad)
rad_num = np.arange(rad_len, dtype = int)
sol_num = np.arange(3, dtype = int)
sol_len = len(sol_num)
nfour = float(2048 / 2 / 2 * 0.5)
theta = 90 - np.linspace(0.5, 179.5, 179 + 1)
the_len = len(theta)
the_h_len = the_len // 2

rad_3_len = int(rad_len - 2)
flat_3_len = the_len * rad_3_len
theta_3_ex = np.full((the_len, rad_3_len), np.nan, dtype = float)
theta_3_ex[:] = theta[:, np.newaxis]
theta_3_flat = np.reshape(theta_3_ex, (flat_3_len))
rad_3_ex = np.full((the_len, rad_3_len), np.nan, dtype = float)
rad_3_ex[:] = rad[2:][np.newaxis, :]
rad_3_flat = np.reshape(rad_3_ex, (flat_3_len))

flat_len = the_len * rad_len
theta_ex = np.full((the_len, rad_len), np.nan, dtype = float)
theta_ex[:] = theta[:, np.newaxis]
theta_flat = np.reshape(theta_ex, (flat_len))
rad_ex = np.full((the_len, rad_len), np.nan, dtype = float)
rad_ex[:] = rad[np.newaxis, :]
rad_flat = np.reshape(rad_ex, (flat_len))

sur_ang = np.array([180, 180, 37, 24, 17], dtype = float)
theta_map = np.full((pol_len, the_len, rad_len, num_evts), np.nan, dtype = float)
theta_map[:] = theta[np.newaxis, :, np.newaxis, np.newaxis]
sur_bool = theta_map <= sur_ang[np.newaxis, np.newaxis, :, np.newaxis]
sur_bool_flat = np.reshape(sur_bool, (pol_len, flat_len, -1))

x = rad * np.cos(np.radians(sur_ang))
y = rad * np.sin(np.radians(sur_ang)) + 10
sur_10_ang = np.ceil(np.degrees(np.arctan(y / x)))
sur_10_ang[:2] = -180
print(sur_10_ang)
sur_10_bool = theta_map >= sur_10_ang[np.newaxis, np.newaxis, :, np.newaxis]
sur_10_bool_flat = np.reshape(sur_10_bool, (pol_len, flat_len, -1))

sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
radius = np.full((d_len), np.nan, dtype = float)
inu_thrown = np.copy(radius)
coef_r_max = np.full((d_len, pol_len, rad_len, sol_len, num_evts), np.nan, dtype = float) # run, pol, rad, sol, evt
coord_r_max = np.full((d_len, len(ang_num), pol_len, rad_len, sol_len, num_evts), np.nan, dtype = float) # run, thepi, pol, rad, sol, evt
coef_max = np.full((d_len, pol_len, num_evts), np.nan, dtype = float) # run, pol, evt
coord_max = np.full((d_len, len(ang_num) + 1, pol_len, num_evts), np.nan, dtype = float) # run, thepir, pol, evt
coef_3_max = np.copy(coef_max)
coord_3_max = np.copy(coord_max)
coef_s_max = np.copy(coef_max)
coord_s_max = np.copy(coord_max)
coef_s10_max = np.copy(coef_max)
coord_s10_max = np.copy(coord_max)
mf_max = np.full((d_len, pol_len, num_evts), np.nan, dtype = float) # run, pol, evt
mf_max_each = np.full((d_len, pol_len, 2, 7, 6, num_evts), np.nan, dtype = float) # run, pol, sho, the, off, evt
mf_temp = np.full((d_len, pol_len, 2, num_evts), np.nan, dtype = float) # run, pol, thephi, evt
if Type == 'signal':
    pnu = np.full((d_len, num_evts), np.nan, dtype = float)
    flavor = np.copy(sim_run)
    exponent = np.full((d_len, 2), 0, dtype = int)
    weight = np.copy(pnu)
    probability = np.copy(pnu)
    nuflavorint = np.copy(pnu)
    nu_nubar = np.copy(pnu)
    currentint = np.copy(pnu)
    elast_y = np.copy(pnu)
    posnu = np.full((d_len, 6, num_evts), np.nan, dtype = float)
    posnu_antcen = np.copy(posnu)
    nnu = np.copy(posnu)
    rec_ang = np.full((d_len, 2, num_ants, num_evts), np.nan, dtype = float)
    view_ang = np.copy(rec_ang)
    arrival_time = np.copy(rec_ang)
    sig_in = np.full((d_len, num_evts), 0, dtype = int)
    sig_in_wide = np.full((d_len, num_evts), 0, dtype = int)
    signal_bin = np.copy(rec_ang)
    ray_step_edge = np.full((d_len, 2, 2, 2, num_ants, num_evts), np.nan, dtype = float) # rays, xz, edge
    ray_in_air = np.full((d_len, num_evts), 0, dtype = int)
    snr_b_max = np.full((d_len, pol_len + 1, num_evts), np.nan, dtype = float)

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
        hf = h5py.File(f'{i_path}sub_info{hf_name}', 'r')
    except FileNotFoundError:
        print(f'{i_path}sub_info{hf_name}')
        continue
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    radius[r] = hf['radius'][:]
    inu_thrown[r] = hf['inu_thrown'][-1]
    if Type == 'signal':
        pnu[r] = hf['pnu'][:]
        flavor[r] = cons[4]
        exponent[r] = hf['exponent_range'][:]
        weight[r] = hf['weight'][:]
        probability[r] = hf['probability'][:]
        nuflavorint[r] = hf['nuflavorint'][:]
        nu_nubar[r] = hf['nu_nubar'][:]
        currentint[r] = hf['currentint'][:]
        elast_y[r] = hf['elast_y'][:]
        posnu[r] = hf['posnu'][:]
        posnu_antcen[r] = hf['posnu_antcen'][:]
        nnu[r] = hf['nnu'][:]
        rec_ang[r] = hf['rec_ang'][:]
        view_ang[r] = hf['view_ang'][:]
        arrival_time[r] = hf['arrival_time'][:]
        wf_time = hf['wf_time'][:]
        wf_dege = np.array([wf_time[0] - 0.5, wf_time[-1] + 0.5])
        wf_dege_wide = np.array([wf_time[0] - nfour - 0.5, wf_time[-1] + nfour + 0.5])
        sig_bin = hf['signal_bin'][:]
        sig_in[r] = np.nansum(np.digitize(sig_bin, wf_dege) == 1, axis = (0, 1))
        sig_in_wide[r] = np.nansum(np.digitize(sig_bin, wf_dege_wide) == 1, axis = (0, 1))
        signal_bin[r] = sig_bin
        ray_step_edge[r] = hf['ray_step_edge'][:]
        ray_step_edge_re = np.reshape(ray_step_edge[r][:, 1, 0], (num_sols * num_ants, -1))
        ray_in_air[r] = (~np.any(ray_step_edge_re >= 0, axis = 0)).astype(int)
        del wf_time, sig_bin, wf_dege, wf_dege_wide, ray_step_edge_re
    del hf

    if Type == 'signal':
        ex_run = get_example_run(Station, config[r])
        bad_ant = known_issue.get_bad_antenna(ex_run)
        try:
            hf = h5py.File(f'{sb_path}snr_banila{hf_name}', 'r')
            snr_tot = hf['snr'][:]
            snr_tot[bad_ant] = np.nan
            snr_b_max[r, 0] = -np.sort(-snr_tot[:8], axis = 0)[2]
            snr_b_max[r, 1] = -np.sort(-snr_tot[8:], axis = 0)[2]
            snr_b_max[r, 2] = -np.sort(-snr_tot, axis = 0)[2]
            del hf, snr_tot
        except FileNotFoundError:
            print(f'{sb_path}snr_banila{hf_name}')
        del ex_run, bad_ant

    try:
        #print(f'{r_path}reco_ele{hf_name}')
        hf = h5py.File(f'{r_path}reco_ele{hf_name}', 'r')
        
        coef_tot = hf['coef'][:] # pol, theta, rad, sol, evt
        coord_tot = hf['coord'][:] # pol, theta, rad, sol, evt

        # max in each r
        coef_r_max_idx = np.nanargmax(coef_tot, axis = 1)
        coef_r_max[r] = coef_tot[pol_num[:, np.newaxis, np.newaxis, np.newaxis], coef_r_max_idx, rad_num[np.newaxis, :, np.newaxis, np.newaxis], sol_num[np.newaxis, np.newaxis, :, np.newaxis], evt_num[np.newaxis, np.newaxis, np.newaxis, :]]
        coord_r_max[r, 0] = theta[coef_r_max_idx]
        coord_r_max[r, 1] = coord_tot[pol_num[:, np.newaxis, np.newaxis, np.newaxis], coef_r_max_idx, rad_num[np.newaxis, :, np.newaxis, np.newaxis], sol_num[np.newaxis, np.newaxis, :, np.newaxis], evt_num[np.newaxis, np.newaxis, np.newaxis, :]]

        # max for all
        coef_re = np.reshape(coef_tot[:, :, 2:, 2], (pol_len, flat_3_len, -1)) # pol, theta x rad, (sol), evt
        coord_re = np.reshape(coord_tot[:, :, 2:, 2], (pol_len, flat_3_len, -1)) # pol, theta x rad, (sol), evt
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_3_max[r] = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_3_max[r, 0] = theta_3_flat[coef_max_idx]
        coord_3_max[r, 1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_3_max[r, 2] = rad_3_flat[coef_max_idx]

        coef_re = np.reshape(coef_tot[:, :, :, 2], (pol_len, flat_len, -1)) # pol, theta x rad, (sol), evt
        coord_re = np.reshape(coord_tot[:, :, :, 2], (pol_len, flat_len, -1)) # pol, theta x rad, (sol), evt
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_max[r] = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_max[r, 0] = theta_flat[coef_max_idx]
        coord_max[r, 1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_max[r, 2] = rad_flat[coef_max_idx]
 
        coef_re[sur_bool_flat] = np.nan
        coord_re[sur_bool_flat] = np.nan
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_s_max[r] = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_s_max[r, 0] = theta_flat[coef_max_idx]
        coord_s_max[r, 1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_s_max[r, 2] = rad_flat[coef_max_idx]

        coef_re[sur_10_bool_flat] = np.nan
        coord_re[sur_10_bool_flat] = np.nan
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_s10_max[r] = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_s10_max[r, 0] = theta_flat[coef_max_idx]
        coord_s10_max[r, 1] = coord_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_s10_max[r, 2] = rad_flat[coef_max_idx]
        del hf, coef_tot, coord_tot, coef_re, coord_re, coef_max_idx, coef_r_max_idx
    except FileNotFoundError:
        print(f'{r_path}reco_ele{hf_name}')

    try:
        hf = h5py.File(f'{m_path}mf{hf_name}', 'r')
        mf_max[r] = hf['mf_max'][:pol_len] # pol, evt
        mf_max_each[r] = hf['mf_max_each'][:pol_len] # pol, sho, the, off, evt
        mf_temp[r] = hf['mf_temp'][:, 1:3] # of pols, theta n phi, # of evts
        #mf_temp[r, 2] = hf['mf_temp_com'][1:3] # of pols, theta n phi, # of evts
        del hf
    except FileNotFoundError:
        print(f'{m_path}mf{hf_name}')
    del hf_name

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_{Type}_v16_A{Station}.h5'
hf = h5py.File(file_name, 'w')
if Type == 'signal':
    hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
    hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
    hf.create_dataset('exponent', data=exponent, compression="gzip", compression_opts=9)
    hf.create_dataset('weight', data=weight, compression="gzip", compression_opts=9)
    hf.create_dataset('probability', data=probability, compression="gzip", compression_opts=9)
    hf.create_dataset('nuflavorint', data=nuflavorint, compression="gzip", compression_opts=9)
    hf.create_dataset('nu_nubar', data=nu_nubar, compression="gzip", compression_opts=9)
    hf.create_dataset('currentint', data=currentint, compression="gzip", compression_opts=9)
    hf.create_dataset('elast_y', data=elast_y, compression="gzip", compression_opts=9)
    hf.create_dataset('posnu', data=posnu, compression="gzip", compression_opts=9)
    hf.create_dataset('posnu_antcen', data=posnu_antcen, compression="gzip", compression_opts=9)
    hf.create_dataset('nnu', data=nnu, compression="gzip", compression_opts=9)
    hf.create_dataset('rec_ang', data=rec_ang, compression="gzip", compression_opts=9)
    hf.create_dataset('view_ang', data=view_ang, compression="gzip", compression_opts=9)
    hf.create_dataset('arrival_time', data=arrival_time, compression="gzip", compression_opts=9)
    hf.create_dataset('sig_in', data=sig_in, compression="gzip", compression_opts=9)
    hf.create_dataset('sig_in_wide', data=sig_in_wide, compression="gzip", compression_opts=9)
    hf.create_dataset('signal_bin', data=signal_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('ray_step_edge', data=ray_step_edge, compression="gzip", compression_opts=9)
    hf.create_dataset('ray_in_air', data=ray_in_air, compression="gzip", compression_opts=9)
    hf.create_dataset('snr_b_max', data=snr_b_max, compression="gzip", compression_opts=9)
hf.create_dataset('run_map', data=run_map, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown', data=inu_thrown, compression="gzip", compression_opts=9)
hf.create_dataset('coef_r_max', data=coef_r_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_r_max', data=coord_r_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_3_max', data=coef_3_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_3_max', data=coord_3_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_s_max', data=coef_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_s_max', data=coord_s_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_s10_max', data=coef_s10_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_s10_max', data=coord_s10_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max_each', data=mf_max_each, compression="gzip", compression_opts=9)
hf.create_dataset('mf_temp', data=mf_temp, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




