import numpy as np
from tqdm import tqdm
import h5py

def cw_add_collector(Station, Run, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import hist_loader
    from tools.ara_known_issue import known_issue_loader

    knwon_issue = known_issue_loader(Station)
    bad_ant = knwon_issue.get_bad_antenna(Run, good_ant_true = True)
    bad_ant_idx = bad_ant != 0
    del knwon_issue, bad_ant

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')

    rf_evt = cw_hf['rf_evt'][:]
    clean_evt = cw_hf['clean_evt'][:]
    clean_idx = np.in1d(rf_evt, clean_evt)
    del rf_evt, clean_evt

    print('data loading')
    ratio_bins = cw_hf['ratio_bins'][:]
    amp_err_bins = cw_hf['amp_err_bins'][:]
    phase_err_bins = cw_hf['phase_err_bins'][:]
    sub_freq = cw_hf['sub_freq'][:]
    sub_freq = np.roll(sub_freq, 1, axis = 0)
    sub_freq[0] = np.nan
    sub_amp = cw_hf['sub_amp'][:]
    sub_amp = np.roll(sub_amp, 1, axis = 0)
    sub_amp[0] = np.nan
    sub_ratio = cw_hf['sub_ratio'][:]
    sub_amp_err = cw_hf['sub_amp_err'][:]
    sub_phase_err = cw_hf['sub_phase_err'][:]
    sub_ratio = np.roll(sub_ratio, 1, axis = 0)
    sub_amp_err = np.roll(sub_amp_err, 1, axis = 0)
    sub_phase_err = np.roll(sub_phase_err, 1, axis = 0)
    sub_amp_err[0, bad_ant_idx] = 0
    sub_amp_err[0, ~bad_ant_idx] = np.nan
    sub_phase_err[0, bad_ant_idx] = 0
    sub_phase_err[0, ~bad_ant_idx] = np.nan
    sub_ratio[0, bad_ant_idx] = 0
    sub_ratio[0, ~bad_ant_idx] = np.nan
    del bad_ant_idx

    print('hist making')
    ara_hist = hist_loader(ratio_bins)
    ratio_rf_hist = ara_hist.get_1d_hist(sub_ratio, use_flat = True)
    ratio_rf_cut_hist = ara_hist.get_1d_hist(sub_ratio, cut = ~clean_idx, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_err_bins)
    amp_err_rf_hist = ara_hist.get_1d_hist(sub_amp_err, use_flat = True)
    amp_err_rf_cut_hist = ara_hist.get_1d_hist(sub_amp_err, cut = ~clean_idx, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(phase_err_bins)
    phase_err_rf_hist = ara_hist.get_1d_hist(sub_phase_err, use_flat = True)
    phase_err_rf_cut_hist = ara_hist.get_1d_hist(sub_phase_err, cut = ~clean_idx, use_flat = True)
    del ara_hist

    print('2d')
    ara_hist = hist_loader(amp_err_bins, ratio_bins)
    amp_err_ratio_rf_map = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, use_flat = True)
    amp_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, cut = ~clean_idx, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(phase_err_bins, ratio_bins)
    phase_err_ratio_rf_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, use_flat = True)
    phase_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, cut = ~clean_idx, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_err_bins, phase_err_bins)
    amp_err_phase_err_rf_map = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, use_flat = True)
    amp_err_phase_err_rf_cut_map = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, cut = ~clean_idx, use_flat = True)
    del ara_hist, cw_hf, clean_idx, ratio_bins, amp_err_bins, phase_err_bins

    print('saving')
    hf = h5py.File(cw_dat, 'r+')  
    sub_freq_r = hf['sub_freq'][:]
    sub_freq_r[...] = sub_freq
    sub_amp_r = hf['sub_amp'][:]
    sub_amp_r[...] = sub_amp
    sub_ratio_r = hf['sub_ratio'][:]
    sub_ratio_r[...] = sub_ratio
    sub_amp_err_r = hf['sub_amp_err'][:]
    sub_amp_err_r[...] = sub_amp_err
    sub_phase_err_r = hf['sub_phase_err'][:]
    sub_phase_err_r[...] = sub_phase_err
    ratio_rf_hist_r = hf['ratio_rf_hist'][:]
    ratio_rf_hist_r[...] = ratio_rf_hist
    ratio_rf_cut_hist_r = hf['ratio_rf_cut_hist'][:]
    ratio_rf_cut_hist_r[...] = ratio_rf_cut_hist
    amp_err_rf_hist_r = hf['amp_err_rf_hist'][:]
    amp_err_rf_hist_r[...] = amp_err_rf_hist
    amp_err_rf_cut_hist_r = hf['amp_err_rf_cut_hist'][:]
    amp_err_rf_cut_hist_r[...] = amp_err_rf_cut_hist
    phase_err_rf_hist_r = hf['phase_err_rf_hist'][:]
    phase_err_rf_hist_r[...] = phase_err_rf_hist
    phase_err_rf_cut_hist_r = hf['phase_err_rf_cut_hist'][:]
    phase_err_rf_cut_hist_r[...] = phase_err_rf_cut_hist
    amp_err_ratio_rf_map_r = hf['amp_err_ratio_rf_map'][:]
    amp_err_ratio_rf_map_r[...] = amp_err_ratio_rf_map
    amp_err_ratio_rf_cut_map_r = hf['amp_err_ratio_rf_cut_map'][:]
    amp_err_ratio_rf_cut_map_r[...] = amp_err_ratio_rf_cut_map
    phase_err_ratio_rf_map_r = hf['phase_err_ratio_rf_map'][:]
    phase_err_ratio_rf_map_r[...] = phase_err_ratio_rf_map
    phase_err_ratio_rf_cut_map_r = hf['phase_err_ratio_rf_cut_map'][:]
    phase_err_ratio_rf_cut_map_r[...] = phase_err_ratio_rf_cut_map
    amp_err_phase_err_rf_map_r = hf['amp_err_phase_err_rf_map'][:]
    amp_err_phase_err_rf_map_r[...] = amp_err_phase_err_rf_map
    amp_err_phase_err_rf_cut_map_r = hf['amp_err_phase_err_rf_cut_map'][:]
    amp_err_phase_err_rf_cut_map_r[...] = amp_err_phase_err_rf_cut_map
    hf.close()
    del sub_freq, sub_amp, sub_ratio, sub_amp_err, sub_phase_err 
    del hf, ratio_rf_hist, ratio_rf_cut_hist, amp_err_rf_hist, amp_err_rf_cut_hist, phase_err_rf_hist, phase_err_rf_cut_hist
    del amp_err_ratio_rf_map, amp_err_ratio_rf_cut_map, phase_err_ratio_rf_map, phase_err_ratio_rf_cut_map, amp_err_phase_err_rf_map, amp_err_phase_err_rf_cut_map

    hf = h5py.File(cw_dat, 'r')
    for f in list(hf):
        print(f)
    hf.close()
    del hf, run_info, cw_dat

    print('cw collecting is done!')

    return 

