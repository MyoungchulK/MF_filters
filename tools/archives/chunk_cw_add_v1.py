import numpy as np
from tqdm import tqdm
import h5py

def cw_add_collector(Station, Run, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import hist_loader

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

    sub_weight = np.count_nonzero(~np.isnan(sub_power), axis = 0)
    sub_weight_idx = sub_weight == 0
    sub_weight = sub_weight.astype(float)
    sub_weight[sub_weight_idx] = np.nan
    sub_weight = 1 / sub_weight
    print(sub_weight.shape)
    sub_weight = np.repeat(sub_weight[np.newaxis, :, :], sub_power.shape[0], axis = 0)
    print(sub_weight.shape)

    print('fft map')
    ara_hist = hist_loader(freq_bins, amp_bins)
    sub_rf_map_w = ara_hist.get_2d_hist(sub_freq, sub_amp, weight = sub_weight, use_flat = True)
    sub_rf_cut_map_w = ara_hist.get_2d_hist(sub_freq, sub_amp, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    print('1d')
    ara_hist = hist_loader(power_bins)
    power_rf_hist_w = ara_hist.get_1d_hist(sub_power, weight = sub_weight, use_flat = True)
    power_rf_cut_hist_w = ara_hist.get_1d_hist(sub_power, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(ratio_bins)
    ratio_rf_hist_w = ara_hist.get_1d_hist(sub_ratio, weight = sub_weight, use_flat = True)
    ratio_rf_cut_hist_w = ara_hist.get_1d_hist(sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_err_bins)
    amp_err_rf_hist_w = ara_hist.get_1d_hist(sub_amp_err, weight = sub_weight, use_flat = True)
    amp_err_rf_cut_hist_w = ara_hist.get_1d_hist(sub_amp_err, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(phase_err_bins)
    phase_err_rf_hist_w = ara_hist.get_1d_hist(sub_phase_err, weight = sub_weight, use_flat = True)
    phase_err_rf_cut_hist_w = ara_hist.get_1d_hist(sub_phase_err, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    print('2d')
    ara_hist = hist_loader(amp_err_bins, ratio_bins)
    amp_err_ratio_rf_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, weight = sub_weight, use_flat = True)
    amp_err_ratio_rf_cut_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(phase_err_bins, ratio_bins)
    phase_err_ratio_rf_map_w = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, weight = sub_weight, use_flat = True)
    phase_err_ratio_rf_cut_map_w = ara_hist.get_2d_hist(sub_phase_err, sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_bins, ratio_bins)
    amp_ratio_rf_map_w = ara_hist.get_2d_hist(sub_amp, sub_ratio, weight = sub_weight, use_flat = True)
    amp_ratio_rf_cut_map_w = ara_hist.get_2d_hist(sub_amp, sub_ratio, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist

    ara_hist = hist_loader(amp_err_bins, phase_err_bins)
    amp_err_phase_err_rf_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, weight = sub_weight, use_flat = True)
    amp_err_phase_err_rf_cut_map_w = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, cut = ~clean_idx, weight = sub_weight, use_flat = True)
    del ara_hist, cw_hf, clean_idx, freq_bins, amp_bins, power_bins, ratio_bins, amp_err_bins, phase_err_bins
    del sub_freq, sub_amp, sub_power, sub_ratio, sub_amp_err, sub_phase_err
    
    print('saving')
    hf = h5py.File(cw_dat, 'a')  
    hf.create_dataset('sub_weight', data=sub_weight, compression="gzip", compression_opts=9) 
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

