import numpy as np
from tqdm import tqdm
import h5py

def cw_hist_collector(Station, Run, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer_temp import hist_loader
    from tools.ara_known_issue import known_issue_loader

    knwon_issue = known_issue_loader(Station)
    bad_ant = knwon_issue.get_bad_antenna(Run, good_ant_true = True)
    bad_ant_idx = bad_ant != 0
    del knwon_issue, bad_ant

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')
    del run_info

    rf_evt = cw_hf['rf_evt'][:]
    clean_evt = cw_hf['clean_evt'][:]
    clean_idx = np.in1d(rf_evt, clean_evt)
    del rf_evt, clean_evt

    print('data loading')
    freq_bins = cw_hf['freq_bins'][:]
    amp_bins = cw_hf['amp_bins'][:]
    power_bins = cw_hf['power_bins'][:]
    ratio_bins = cw_hf['ratio_bins'][:]
    amp_err_bins = cw_hf['amp_err_bins'][:]
    phase_err_bins = cw_hf['phase_err_bins'][:]
    sub_freq = cw_hf['sub_freq'][:]
    sub_amp = cw_hf['sub_amp'][:]
    sub_amp_err = cw_hf['sub_amp_err'][:]
    sub_phase_err = cw_hf['sub_phase_err'][:]
    sub_power = cw_hf['sub_power'][:]
    sub_ratio = cw_hf['sub_ratio'][:]

    sub_freq = np.roll(sub_freq, 1, axis = 0)
    sub_freq[0] = np.nan
    sub_amp = np.roll(sub_amp, 1, axis = 0)
    sub_amp[0] = np.nan
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
    
    sub_sum = np.nansum(sub_power, axis = 0)
    sub_weight = sub_power / sub_sum[np.newaxis, :, :]
    print(sub_weight.shape)
    del sub_sum

    print('fft map')
    ara_hist = hist_loader(freq_bins, amp_bins)
    sub_rf_map = ara_hist.get_2d_hist(sub_freq, sub_amp, use_flat = True)
    sub_rf_cut_map = ara_hist.get_2d_hist(sub_freq, sub_amp, cut = ~clean_idx, use_flat = True
    sub_rf_map_w = ara_hist.get_2d_hist(sub_freq, sub_amp, weight = sub_weight, use_flat = True)
    sub_rf_cut_map_w = ara_hist.get_2d_hist(sub_freq, sub_amp, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    print('1d')
    ara_hist = hist_loader(power_bins)
    power_rf_hist = ara_hist.get_1d_hist(sub_power, use_flat = True)
    power_rf_cut_hist = ara_hist.get_1d_hist(sub_power, cut = ~clean_idx, use_flat = True)
    power_rf_hist_w = ara_hist.get_1d_hist(sub_power, weight = sub_weight, use_flat = True)
    power_rf_cut_hist_w = ara_hist.get_1d_hist(sub_power, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(ratio_bins)
    ratio_rf_hist = ara_hist.get_1d_hist(sub_ratio, use_flat = True)
    ratio_rf_cut_hist = ara_hist.get_1d_hist(sub_ratio, cut = ~clean__idx, use_flat = True)
    ratio_rf_hist_w = ara_hist.get_1d_hist(sub_ratio, weight = sub_weight, use_flat = True)
    ratio_rf_cut_hist_w = ara_hist.get_1d_hist(sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_err_bins)
    amp_err_rf_hist = ara_hist.get_1d_hist(sub_amp_err, use_flat = True)
    amp_err_rf_cut_hist = ara_hist.get_1d_hist(sub_amp_err, cut = ~clean_idx, use_flat = True)
    amp_err_rf_hist_w = ara_hist.get_1d_hist(sub_amp_err, weight = sub_weight, use_flat = True)
    amp_err_rf_cut_hist_w = ara_hist.get_1d_hist(sub_amp_err, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(phase_err_bins)
    phase_err_ratio_rf_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, use_flat = True)
    phase_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, cut = ~clean_idx, use_flat = True)
    phase_err_rf_hist_w = ara_hist.get_1d_hist(sub_phase_err, weight = sub_weight, use_flat = True)
    phase_err_rf_cut_hist_w = ara_hist.get_1d_hist(sub_phase_err, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    print('2d')
    ara_hist = hist_loader(amp_err_bins, ratio_bins)
    amp_err_ratio_rf_map = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, use_flat = True)
    amp_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, cut = ~clean__idx, use_flat = True)
    amp_err_ratio_rf_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, weight = sub_weight, use_flat = True)
    amp_err_ratio_rf_cut_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(phase_err_bins, ratio_bins)
    phase_err_ratio_rf_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, use_flat = True)
    phase_err_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, cut = ~clean__idx, use_flat = True)
    phase_err_ratio_rf_map_w = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, weight = sub_weight, use_flat = True)
    phase_err_ratio_rf_cut_map_w = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_bins, ratio_bins)
    amp_ratio_rf_map = ara_hist.get_2d_hist(sub_amp, sub_ratio, use_flat = True)
    amp_ratio_rf_cut_map = ara_hist.get_2d_hist(sub_amp, sub_ratio, cut = ~clean_idx, use_flat = True)
    amp_ratio_rf_map_w = ara_hist.get_2d_hist(sub_amp, sub_ratio, weight = sub_weight, use_flat = True)
    amp_ratio_rf_cut_map_w = ara_hist.get_2d_hist(sub_amp, sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_err_bins, phase_err_bins)
    amp_err_phase_err_rf_map = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, use_flat = True)
    amp_err_phase_err_rf_cut_map = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, cut = ~clean_idx, use_flat = True)
    amp_err_phase_err_rf_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, weight = sub_weight, use_flat = True)
    amp_err_phase_err_rf_cut_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist, cw_hf, clean_idx, freq_bins, amp_bins, power_bins, ratio_bins, amp_err_bins, phase_err_bins
    del sub_freq, sub_amp, sub_power, sub_ratio, sub_amp_err, sub_phase_err
    
    print('saving')
    hf = h5py.File(cw_dat, 'r+')
    del hf['sub_freq']
    del hf['sub_amp']
    del hf['sub_amp_err']
    del hf['sub_phase_err']
    del hf['sub_ratio']
    #del hf['sub_weight']
    del hf['sub_rf_map']
    del hf['sub_rf_cut_map']
    del hf['power_rf_hist']
    del hf['power_rf_cut_hist']
    del hf['ratio_rf_hist']
    del hf['ratio_rf_cut_hist']
    del hf['amp_err_rf_hist']
    del hf['amp_err_rf_cut_hist']
    del hf['phase_err_rf_hist']
    del hf['phase_err_rf_cut_hist']
    del hf['amp_err_ratio_rf_map']
    del hf['amp_err_ratio_rf_cut_map']
    del hf['phase_err_ratio_rf_map']
    del hf['phase_err_ratio_rf_cut_map']
    del hf['amp_ratio_rf_map']
    del hf['amp_ratio_rf_cut_map']
    del hf['amp_err_phase_err_rf_map']
    del hf['amp_err_phase_err_rf_cut_map']
    """del hf['sub_rf_map_w']
    del hf['sub_rf_cut_map_w']
    del hf['power_rf_hist_w']
    del hf['power_rf_cut_hist_w']
    del hf['ratio_rf_hist_w']
    del hf['ratio_rf_cut_hist_w']
    del hf['amp_err_rf_hist_w']
    del hf['amp_err_rf_cut_hist_w']
    del hf['phase_err_rf_hist_w']
    del hf['phase_err_rf_cut_hist_w']
    del hf['amp_err_ratio_rf_map_w']
    del hf['amp_err_ratio_rf_cut_map_w']
    del hf['phase_err_ratio_rf_map_w']
    del hf['phase_err_ratio_rf_cut_map_w']
    del hf['amp_ratio_rf_map_w']
    del hf['amp_ratio_rf_cut_map_w']
    del hf['amp_err_phase_err_rf_map_w']
    del hf['amp_err_phase_err_rf_cut_map_w']"""    

    hf.create_dataset('sub_freq', data=sub_freq, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_amp', data=sub_amp, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_amp_err', data=sub_amp_err, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_phase_err', data=sub_phase_err, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_ratio', data=sub_ratio, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_weight', data=sub_weight, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_rf_map', data=sub_rf_map, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_rf_cut_map', data=sub_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('power_rf_hist', data=power_rf_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('power_rf_cut_hist', data=power_rf_cut_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_rf_hist', data=ratio_rf_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_rf_cut_hist', data=ratio_rf_cut_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_rf_hist', data=amp_err_rf_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_rf_cut_hist', data=amp_err_rf_cut_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_rf_hist', data=phase_err_rf_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_rf_cut_hist', data=phase_err_rf_cut_hist, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_ratio_rf_map', data=amp_err_ratio_rf_map, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_ratio_rf_cut_map', data=amp_err_ratio_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_ratio_rf_map', data=phase_err_ratio_rf_map, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_ratio_rf_cut_map', data=phase_err_ratio_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_ratio_rf_map', data=amp_ratio_rf_map, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_ratio_rf_cut_map', data=amp_ratio_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_phase_err_rf_map', data=amp_err_phase_err_rf_map, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_phase_err_rf_cut_map', data=amp_err_phase_err_rf_cut_map, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_rf_map_w', data=sub_rf_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('sub_rf_cut_map_w', data=sub_rf_cut_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('power_rf_hist_w', data=power_rf_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('power_rf_cut_hist_w', data=power_rf_cut_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_rf_hist_w', data=ratio_rf_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio_rf_cut_hist_w', data=ratio_rf_cut_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_rf_hist_w', data=amp_err_rf_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_rf_cut_hist_w', data=amp_err_rf_cut_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_rf_hist_w', data=phase_err_rf_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_rf_cut_hist_w', data=phase_err_rf_cut_hist_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_ratio_rf_map_w', data=amp_err_ratio_rf_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_ratio_rf_cut_map_w', data=amp_err_ratio_rf_cut_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_ratio_rf_map_w', data=phase_err_ratio_rf_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('phase_err_ratio_rf_cut_map_w', data=phase_err_ratio_rf_cut_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_ratio_rf_map_w', data=amp_ratio_rf_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_ratio_rf_cut_map_w', data=amp_ratio_rf_cut_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_phase_err_rf_map_w', data=amp_err_phase_err_rf_map_w, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_phase_err_rf_cut_map_w', data=amp_err_phase_err_rf_cut_map_w, compression="gzip", compression_opts=9)
    hf.close()
    del hf, sub_rf_map_w, sub_rf_cut_map_w, power_rf_hist_w, power_rf_cut_hist_w, ratio_rf_hist_w, ratio_rf_cut_hist_w, amp_err_rf_hist_w, amp_err_rf_cut_hist_w, phase_err_rf_hist_w, phase_err_rf_cut_hist_w
    del amp_err_ratio_rf_map_w, amp_err_ratio_rf_cut_map_w, phase_err_ratio_rf_map_w, phase_err_ratio_rf_cut_map_w, amp_err_phase_err_rf_map_w, amp_err_phase_err_rf_cut_map_w
    del amp_ratio_rf_map_w, amp_ratio_rf_cut_map_w
    
    hf = h5py.File(cw_dat, 'r')
    for f in list(hf):
        print(f)
    hf.close()
    del hf, cw_dat

    print('cw collecting is done!')

    return 

