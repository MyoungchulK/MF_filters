import numpy as np
from tqdm import tqdm
import h5py

def cw_time_collector(Station, Run, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import hist_loader

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')
    del run_info

    print('data loading')
    sub_weight = cw_hf['sub_weight'][:]
    sub_ratio = cw_hf['sub_ratio'][:]
    sub_tot_ratio = cw_hf['sub_tot_ratio'][:]
    sub_power = cw_hf['sub_power'][:]
    sub_amp_err = cw_hf['sub_amp_err'][:]
    sub_phase_err = cw_hf['sub_phase_err'][:]
    sub_amp_bound = cw_hf['sub_amp_bound'][:]
    sub_phase_bound = cw_hf['sub_phase_bound'][:]
    sub_freq = cw_hf['sub_freq'][:]
    power_bins = cw_hf['power_bins'][:]
    amp_err_bins = cw_hf['amp_err_bins'][:]
    phase_err_bins = cw_hf['phase_err_bins'][:]
    bound_bins = cw_hf['bound_bins'][:]
    freq_bins = cw_hf['freq_bins'][:]
    unix_time = cw_hf['unix_time'][:]
    clean_unix = cw_hf['clean_unix'][:]

    print('hist')
    sec_to_min = 60
    unix_min_i = int(np.floor(np.nanmin(unix_time) / sec_to_min) * sec_to_min)
    unix_min_f = int(np.ceil(np.nanmax(unix_time) / sec_to_min) * sec_to_min)
    unix_min_bins = np.linspace(unix_min_i, unix_min_f, (unix_min_f - unix_min_i)//60 + 1, dtype = int)
    unix_min_range = np.arange(unix_min_bins[0], np.nanmax(unix_time), 60)
    ara_hist = hist_loader(unix_min_bins, ratio_bins)
    unix_min_bin_center = ara_hist.bin_x_center

    sol_pad = 200
    num_ants = 16
    if len(clean_unix) == 0:
        clean_unix_ant = np.full((num_ants, clean_unix), np.nan, dtype = float)
        clean_unix_all = np.full((sol_pad, num_ants, clean_unix), np.nan, dtype = float)
    else:
        clean_unix_ant = np.repeat(clean_unix[np.newaxis, :], num_ants, axis = 0)
        clean_unix_all = np.repeat(clean_unix_ant[np.newaxis, :, :], sol_pad, axis = 0)

    unix_ratio_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_ratio, weight = sub_weight, use_flat = True)
    unix_tot_ratio_rf_cut_map = ara_hist.get_2d_hist(clean_unix_ant, sub_tot_ratio)
    unix_ratio_rf_cut_map_max = ara_hist.get_2d_hist_max(unix_ratio_rf_cut_map)
    unix_tot_ratio_rf_cut_map_max = ara_hist.get_2d_hist_max(unix_tot_ratio_rf_cut_map)
    del ara_hist, sol_pad

    ara_hist = hist_loader(unix_min_bins, power_bins)
    unix_power_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_power, weight = sub_weight, use_flat = True)
    unix_power_rf_cut_map_max = ara_hist.get_2d_hist_max(unix_power_rf_cut_map)
    del ara_hist

    ara_hist = hist_loader(unix_min_bins, amp_err_bins)
    unix_amp_err_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_amp_err, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(unix_min_bins, phase_err_bins)
    unix_phase_err_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_phase_err, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(unix_min_bins, bound_bins)
    unix_amp_bound_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_amp_bound, weight = sub_weight, use_flat = True)
    unix_phase_bound_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_phase_bound, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(unix_min_bins, freq_bins)
    unix_freq_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_freq, weight = sub_weight, use_flat = True)
    temp_max_idx = np.nanargmax(unix_freq_rf_cut_map, axis = 1)
    unix_freq_rf_cut_map_max = freq_bin_center[temp_max_idx]
    del ara_hist, clean_unix_ant, clean_unix_all, num_ants, temp_max_idx
    cw_hf.close()

    print('saving')
    hf = h5py.File(cw_dat, 'r+')
    del hf['unix_min_range']
    del hf['unix_min_bins']
    del hf['unix_min_bin_center']
    del hf['unix_ratio_rf_cut_map']
    del hf['unix_tot_ratio_rf_cut_map']
    del hf['unix_ratio_rf_cut_map_max']
    del hf['unix_tot_ratio_rf_cut_map_max']
    del hf['unix_power_rf_cut_map']
    del hf['unix_power_rf_cut_map_max']
    del hf['unix_amp_err_rf_cut_map']
    del hf['unix_phase_err_rf_cut_map']
    del hf['unix_amp_bound_rf_cut_map']
    del hf['unix_phase_bound_rf_cut_map']
    del hf['unix_freq_rf_cut_map']
    del hf['unix_freq_rf_cut_map_max']

    hf.create_dataset('unix_min_range', data=unix_min_range, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_min_bins', data=unix_min_bins, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_min_bin_center', data=unix_min_bin_center, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_ratio_rf_cut_map', data=unix_ratio_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_tot_ratio_rf_cut_map', data=unix_tot_ratio_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_ratio_rf_cut_map_max', data=unix_ratio_rf_cut_map_max, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_tot_ratio_rf_cut_map_max', data=unix_tot_ratio_rf_cut_map_max, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_power_rf_cut_map', data=unix_power_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_power_rf_cut_map_max', data=unix_power_rf_cut_map_max, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_amp_err_rf_cut_map', data=unix_amp_err_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_phase_err_rf_cut_map', data=unix_phase_err_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_amp_bound_rf_cut_map', data=unix_amp_bound_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_phase_bound_rf_cut_map', data=unix_phase_bound_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_freq_rf_cut_map', data=unix_freq_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_freq_rf_cut_map_max', data=unix_freq_rf_cut_map_max, compression="gzip", compression_opts=9)
    hf.close()

    hf = h5py.File(cw_dat, 'r')
    for f in list(hf):
        print(f)
    hf.close()
    del hf, cw_dat

    print('cw collecting is done!')

    return



