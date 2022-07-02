import numpy as np
from tqdm import tqdm
import h5py

def cw_lite_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_wf_analyzer import hist_loader
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    st = ara_uproot.station_id
    run = ara_uproot.run
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type() 
    unix_min_bins = ara_uproot.get_minute_bins_in_unixtime()
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
    del ara_uproot

    knwon_issue = known_issue_loader(st)
    bad_ant = knwon_issue.get_bad_antenna(run) == 1 
    del knwon_issue

    # qulity cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_cut_dat = run_info.get_result_path(file_type = 'daq_cut', verbose = True)
    daq_cut_hf = h5py.File(daq_cut_dat, 'r')
    daq_cut = daq_cut_hf['total_daq_cut'][:]
    daq_cut[:, 10] = 0 # disable bad unix time
    daq_cut[:, 21] = 0 # disable known bad run
    daq_cut_sum = np.nansum(daq_cut, axis = 1)
    clean_evt_idx = np.logical_and(trig_type == 0, daq_cut_sum == 0)
    clean_entry = entry_num[clean_evt_idx]
    clean_evt = evt_num[clean_evt_idx]
    num_clean_evts = len(clean_evt)
    print(f'Number of clean event is {num_clean_evts}')
    del clean_evt_idx, run_info, daq_cut_dat, daq_cut_hf, daq_cut, st, run, entry_num

    # wf analyzer
    cw_params = np.full((num_ants), 0.02, dtype = float)
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, cw_params = cw_params)
    del cw_params

    # output
    sol_pad = 100
    sub_ratio = np.full((sol_pad, num_ants, num_clean_evts), np.nan, dtype = float)
    sub_ratio[0, ~bad_ant] = 0
    sub_power = np.copy(sub_ratio)

    # loop over the events
    for evt in tqdm(range(num_clean_evts)):
      #if evt == 100:        

        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            if bad_ant[ant]:
                continue                
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True)
            num_sols = wf_int.sin_sub.num_sols + 1
            sub_power[:num_sols, ant, evt] = wf_int.sin_sub.sub_powers
            sub_ratio[:num_sols, ant, evt] = wf_int.sin_sub.sub_ratios
            del raw_t, raw_v, num_sols 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_root, num_clean_evts, clean_entry, wf_int 

    sub_sum = np.nansum(sub_power, axis = 0)
    sub_weight = sub_power / sub_sum[np.newaxis, :, :]
    del sub_sum, sub_power

    print('ratio')
    ratio_range = np.arange(0, 1, 0.02)
    ratio_bins = np.linspace(0, 1, 50 + 1)
    ara_hist = hist_loader(ratio_bins)
    ratio_bin_center = ara_hist.bin_x_center    
    ratio_rf_cut_hist = ara_hist.get_1d_hist(sub_ratio, weight = sub_weight, use_flat = True)
    del ara_hist

    print('time')
    unix_min_range = np.arange(unix_min_bins[0], np.nanmax(unix_time), 60)
    ara_hist = hist_loader(unix_min_bins, ratio_bins)
    unix_min_bin_center = ara_hist.bin_x_center

    clean_unix = unix_time[np.in1d(evt_num, clean_evt)]
    if len(clean_unix) == 0:
        clean_unix_ant = np.full((num_ants, len(clean_unix)), np.nan, dtype = float)
        clean_unix_all = np.full((sol_pad, num_ants, len(clean_unix)), np.nan, dtype = float)
    else:
        clean_unix_ant = np.repeat(clean_unix[np.newaxis, :], num_ants, axis = 0)
        clean_unix_all = np.repeat(clean_unix_ant[np.newaxis, :, :], sol_pad, axis = 0)

    unix_ratio_rf_cut_map = ara_hist.get_2d_hist(clean_unix_all, sub_ratio, weight = sub_weight, use_flat = True)
    unix_ratio_rf_cut_map_max = ara_hist.get_2d_hist_max(unix_ratio_rf_cut_map)
    del ara_hist, sol_pad, clean_unix_ant, clean_unix_all, num_ants

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'clean_unix':clean_unix,
            'pps_number':pps_number,
            'bad_ant':bad_ant.astype(int),
            'daq_cut_sum':daq_cut_sum,
            'sub_ratio':sub_ratio,
            'sub_weight':sub_weight,
            'unix_min_range':unix_min_range,
            'unix_min_bins':unix_min_bins,
            'unix_min_bin_center':unix_min_bin_center,
            'ratio_range':ratio_range,
            'ratio_bins':ratio_bins,
            'ratio_bin_center':ratio_bin_center,
            'ratio_rf_cut_hist':ratio_rf_cut_hist,
            'unix_ratio_rf_cut_map':unix_ratio_rf_cut_map,
            'unix_ratio_rf_cut_map_max':unix_ratio_rf_cut_map_max}



