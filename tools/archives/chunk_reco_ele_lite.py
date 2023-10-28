import numpy as np
from tqdm import tqdm
import h5py
import os

def reco_ele_lite_collector(st, run, analyze_blind_dat = False, no_tqdm = False):

    print('Collecting reco ele lite starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_utility import size_checker

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
    z_flat = np.sin(np.radians(theta_flat)) * rad_flat
    del theta_ex, rad_ex

    sur_ang = np.array([180, 180, 37, 24, 17], dtype = float)
    theta_map = np.full((pol_len, the_len, rad_len, sol_len), np.nan, dtype = float)
    theta_map[:] = theta[np.newaxis, :, np.newaxis, np.newaxis]
    sur_bool = theta_map <= sur_ang[np.newaxis, np.newaxis, :, np.newaxis]
    sur_bool_flat = np.reshape(sur_bool, (pol_len, flat_len))
    del sur_ang, theta_map, sur_bool

    # load big file
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    reco_dat = run_info.get_result_path(file_type = 'reco_ele', verbose = True)
    hf = h5py.File(reco_dat, 'r')
    evts = hf['evt_num'][:]
    num_evts = len(evts)
    evt_num = np.arange(num_evts, dtype = int)
    coef_tot = hf['coef'][:] # pol, theta, rad, sol, evt
    coord_tot = hf['coord'][:] # pol, theta, rad, sol, evt
    coef_tot[np.isnan(coef_tot)] = -1
    del run_info, reco_dat, hf

    coef_re = np.reshape(coef_tot, (pol_len, flat_len, -1))
    coord_re = np.reshape(coord_tot, (pol_len, flat_len, -1))
    del coef_tot, coord_tot

    coef_max = np.full((pol_len, num_evts), np.nan, dtype = float) # pol, evt
    coord_max = np.full((ang_len + 2, pol_len, num_evts), np.nan, dtype = float) # thepirz, pol, evt
    coef_s_max = np.copy(coef_max)
    coord_s_max = np.copy(coord_max)

    for t in range(2):
        if t == 1:
            coef_re[sur_bool_flat] = -1
            coord_re[sur_bool_flat] = np.nan
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_max1 = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]] # pol, evt
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
            coef_max[:] = coef_max1
            coord_max[:] = coord_max1
        else:
            coef_s_max[:] = coef_max1
            coord_s_max[:] = coord_max1
        del coef_max1, coord_max1
 
   
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{st}/reco_ele_lite{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'reco_ele_lite{blind_type}_A{st}_R{run}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    hf.create_dataset('evt_num', data=evts, compression="gzip", compression_opts=9)
    hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
    hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
    hf.create_dataset('coef_s_max', data=coef_s_max, compression="gzip", compression_opts=9)
    hf.create_dataset('coord_s_max', data=coord_s_max, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {output_path}{h5_file_name}.', size_checker(f'{output_path}{h5_file_name}'))

    print('Reco ele lite collecting is done!')

    return








