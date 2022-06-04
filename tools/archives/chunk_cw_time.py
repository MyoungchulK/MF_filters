import numpy as np
from tqdm import tqdm
import h5py
from datetime import datetime

def get_max(dat, bin_cen):
    temp_map = np.full(dat.shape, np.nan, dtype = float)
    temp_map[dat > 0.5] = 1
    temp_map *= bin_cen[np.newaxis, :, np.newaxis]
    temp_map = np.nanmax(temp_map, axis = 1)

    return temp_map

def cw_time_collector(Station, Run, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import hist_loader

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')
    del run_info, cw_dat

    evt_num = cw_hf['evt_num'][:]
    rf_evt = cw_hf['rf_evt'][:]
    clean_evt = cw_hf['clean_evt'][:]
    unix_time = cw_hf['unix_time'][:]
    rf_idx = np.in1d(evt_num, rf_evt)
    rf_unix_time = unix_time[rf_idx]
    clean_idx = np.in1d(rf_evt, clean_evt)
    clean_unix_time = rf_unix_time[clean_idx]
    del rf_idx

    date_min = datetime.fromtimestamp(np.nanmin(rf_unix_time))
    date_min = date_min.strftime('%Y%m%d%H%M%S')
    date_min = int(date_min)
    date_min = np.floor(date_min/100) * 100
    date_min = date_min.astype(int)   
    date_min_str = str(date_min)
    print(date_min_str)
    try:
        date_min = datetime(int(date_min_str[:4]), int(date_min_str[4:6]), int(date_min_str[6:8]), int(date_min_str[8:10]), int(date_min_str[10:12])) 
    except ValueError:
        mm = 0
        hh = int(date_min_str[8:10]) + 1
        try:
            date_min = datetime(int(date_min_str[:4]), int(date_min_str[4:6]), int(date_min_str[6:8]), hh, mm)
        except ValueError:
            hh = 0
            dd = int(date_min_str[6:8]) + 1
            date_max = datetime(int(date_min_str[:4]), int(date_min_str[4:6]), dd, hh)
    unix_min = int(datetime.timestamp(date_min))
    print(date_min)
    print(unix_min)

    date_max = datetime.fromtimestamp(np.nanmax(rf_unix_time))
    date_max = date_max.strftime('%Y%m%d%H%M%S') 
    date_max = int(date_max)
    date_max = np.ceil(date_max/100) * 100
    date_max = date_max.astype(int)
    date_max_str = str(date_max)
    print(date_max_str)
    try:
        date_max = datetime(int(date_max_str[:4]), int(date_max_str[4:6]), int(date_max_str[6:8]), int(date_max_str[8:10]), int(date_max_str[10:12]))
    except ValueError:
        mm = 0
        hh = int(date_max_str[8:10]) + 1        
        try:
            date_max = datetime(int(date_max_str[:4]), int(date_max_str[4:6]), int(date_max_str[6:8]), hh, mm)
        except ValueError:
            hh = 0
            dd = int(date_max_str[6:8]) + 1
            date_max = datetime(int(date_max_str[:4]), int(date_max_str[4:6]), dd, hh)
    unix_max = int(datetime.timestamp(date_max))
    print(date_max)
    print(unix_max)
    del date_min, date_min_str, date_max, date_max_str

    unix_bins = np.linspace(unix_min, unix_max, (unix_max - unix_min)//60 + 1, dtype = int)
    unix_bin_center = (unix_bins[1:] + unix_bins[:-1]) / 2
    del unix_min, unix_max

    print('data loading')
    freq_bins = cw_hf['freq_bins'][:]
    freq_bin_center = cw_hf['freq_bin_center'][:]
    amp_bins = cw_hf['amp_bins'][:]
    amp_bin_center = cw_hf['amp_bin_center'][:]
    ratio_bins = cw_hf['ratio_bins'][:]
    ratio_bin_center = cw_hf['ratio_bin_center'][:]
    amp_err_bins = cw_hf['amp_err_bins'][:]
    amp_err_bin_center = cw_hf['amp_err_bin_center'][:]
    phase_err_bins = cw_hf['phase_err_bins'][:]
    phase_err_bin_center = cw_hf['phase_err_bin_center'][:]
    sub_freq = cw_hf['sub_freq'][:]
    sub_amp = cw_hf['sub_amp'][:]
    sub_amp_err = cw_hf['sub_amp_err'][:]
    sub_phase_err = cw_hf['sub_phase_err'][:]
    sub_ratio = cw_hf['sub_ratio'][:]
    del cw_hf

    rf_unix_repeat = np.repeat(rf_unix_time[np.newaxis, :], 16, axis = 0)
    rf_unix_repeat = np.repeat(rf_unix_repeat[np.newaxis, :, :], len(sub_freq[:,0,0]), axis = 0)
    rf_unix_repeat = rf_unix_repeat.astype(float)
    print(rf_unix_repeat.shape)

    ara_hist = hist_loader(unix_bins, ratio_bins)
    unix_ratio_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_ratio, use_flat = True)
    unix_ratio_rf_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_ratio, cut = ~clean_idx, use_flat = True)
    unix_ratio_map_max = get_max(unix_ratio_map, ratio_bin_center)
    unix_ratio_rf_map_max = get_max(unix_ratio_rf_map, ratio_bin_center)
    del ara_hist

    ara_hist = hist_loader(unix_bins, amp_bins)
    unix_amp_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_amp, use_flat = True)
    unix_amp_rf_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_amp, cut = ~clean_idx, use_flat = True)
    unix_amp_map_max = get_max(unix_amp_map, amp_bin_center)
    unix_amp_rf_map_max = get_max(unix_amp_rf_map, amp_bin_center)
    del ara_hist

    ara_hist = hist_loader(unix_bins, freq_bins)
    unix_freq_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_freq, use_flat = True)
    unix_freq_rf_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_freq, cut = ~clean_idx, use_flat = True)
    unix_freq_map_max = np.full((len(unix_bin_center),16), np.nan, dtype = float)
    unix_freq_rf_map_max = np.copy(unix_freq_map_max)
    temp_map = np.copy(unix_freq_map)
    temp_map[temp_map < 0.5] = np.nan
    temp_sum = np.nansum(temp_map, axis = 1)
    temp_rf_map = np.copy(unix_freq_rf_map)
    temp_rf_map[temp_rf_map < 0.5] = np.nan
    temp_rf_sum = np.nansum(temp_rf_map, axis = 1)
    for a in tqdm(range(16)):
        for b in range(len(unix_bin_center)):
            if temp_sum[b,a] > 0.5:
                max_idx = np.nanargmax(temp_map[b,:,a])
                unix_freq_map_max[b,a] = freq_bin_center[max_idx]
            if temp_rf_sum[b,a] > 0.5:
                max_rf_idx = np.nanargmax(temp_rf_map[b,:,a])
                unix_freq_rf_map_max[b,a] = freq_bin_center[max_rf_idx]
    del ara_hist, temp_map, temp_sum, temp_rf_map, temp_rf_sum

    ara_hist = hist_loader(unix_bins, amp_err_bins)
    unix_amp_err_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_amp_err, use_flat = True)
    unix_amp_err_rf_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_amp_err, cut = ~clean_idx, use_flat = True)
    unix_amp_err_map_max = get_max(unix_amp_err_map, amp_err_bin_center)
    unix_amp_err_rf_map_max = get_max(unix_amp_err_rf_map, amp_err_bin_center)
    del ara_hist

    ara_hist = hist_loader(unix_bins, phase_err_bins)
    unix_phase_err_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_phase_err, use_flat = True)
    unix_phase_err_rf_map = ara_hist.get_2d_hist(rf_unix_repeat, sub_phase_err, cut = ~clean_idx, use_flat = True)
    unix_phase_err_map_max = get_max(unix_phase_err_map, phase_err_bin_center)
    unix_phase_err_rf_map_max = get_max(unix_phase_err_rf_map, phase_err_bin_center)
    del ara_hist, rf_unix_repeat, clean_idx
    

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'rf_evt':rf_evt,
            'clean_evt':clean_evt,
            'unix_time':unix_time,
            'rf_unix_time':rf_unix_time,
            'clean_unix_time':clean_unix_time,
            'unix_bins':unix_bins,
            'unix_bin_center':unix_bin_center,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'amp_bins':amp_bins,
            'amp_bin_center':amp_bin_center,
            'ratio_bins':ratio_bins,
            'ratio_bin_center':ratio_bin_center,
            'amp_err_bins':amp_err_bins,
            'amp_err_bin_center':amp_err_bin_center,
            'phase_err_bins':phase_err_bins,
            'phase_err_bin_center':phase_err_bin_center,
            'unix_ratio_map':unix_ratio_map,
            'unix_ratio_rf_map':unix_ratio_rf_map,
            'unix_ratio_map_max':unix_ratio_map_max,
            'unix_ratio_rf_map_max':unix_ratio_rf_map_max,
            'unix_amp_map':unix_amp_map,
            'unix_amp_rf_map':unix_amp_rf_map,
            'unix_amp_map_max':unix_amp_map_max,
            'unix_amp_rf_map_max':unix_amp_rf_map_max,
            'unix_freq_map':unix_freq_map,
            'unix_freq_rf_map':unix_freq_rf_map,
            'unix_freq_map_max':unix_freq_map_max,
            'unix_freq_rf_map_max':unix_freq_rf_map_max,
            'unix_amp_err_map':unix_amp_err_map,
            'unix_amp_err_rf_map':unix_amp_err_rf_map,
            'unix_amp_err_map_max':unix_amp_err_map_max,
            'unix_amp_err_rf_map_max':unix_amp_err_rf_map_max,
            'unix_phase_err_map':unix_phase_err_map,
            'unix_phase_err_rf_map':unix_phase_err_rf_map,
            'unix_phase_err_map_max':unix_phase_err_map_max,
            'unix_phase_err_rf_map_max':unix_phase_err_rf_map_max}




