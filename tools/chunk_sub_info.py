import numpy as np
from tqdm import tqdm

def sub_info_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting sub info starts!')

    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_sensorHk_uproot_loader
    from tools.ara_data_load import ara_eventHk_uproot_loader
    from tools.ara_run_manager import run_info_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    station = ara_uproot.station_id
    run = ara_uproot.run
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    blk_len, rf_blk_len, cal_blk_len, soft_blk_len = ara_uproot.get_block_length()

    #geom info
    ara_geom = ara_geom_loader(station, ara_uproot.year, verbose = True)
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom

    # remove nasty unix error
    evt_sort_idx = np.argsort(evt_num)
    evt_num_sort = evt_num[evt_sort_idx]
    pps_number_sort = pps_number[evt_sort_idx]
    unix_time_sort = unix_time[evt_sort_idx]
    time_stamp_sort = time_stamp[evt_sort_idx]
    trig_type_sort = trig_type[evt_sort_idx]
    blk_len_sort = blk_len[evt_sort_idx]
    rf_blk_len_sort = rf_blk_len[evt_sort_idx]
    cal_blk_len_sort = cal_blk_len[evt_sort_idx]
    soft_blk_len_sort = soft_blk_len[evt_sort_idx]
    del evt_sort_idx

    # remove pps reset
    pps_number_sort_reset = ara_uproot.get_reset_pps_number(use_evt_num_sort = True)

    # event min rate
    unix_min_bins, unix_min_bin_center, unix_min_counts, evt_min_rate_unix, rf_min_rate_unix, cal_min_rate_unix, soft_min_rate_unix = ara_uproot.get_event_rate()
    pps_min_bins, pps_min_bin_center, pps_min_counts, evt_min_rate_pps, rf_min_rate_pps, cal_min_rate_pps, soft_min_rate_pps = ara_uproot.get_event_rate(use_pps =True)
    
    # event sec rate
    unix_sec_bins, unix_sec_bin_center, unix_sec_counts, evt_sec_rate_unix, rf_sec_rate_unix, cal_sec_rate_unix, soft_sec_rate_unix = ara_uproot.get_event_rate(use_sec = True)
    pps_sec_bins, pps_sec_bin_center, pps_sec_counts, evt_sec_rate_pps, rf_sec_rate_pps, cal_sec_rate_pps, soft_sec_rate_pps = ara_uproot.get_event_rate(use_pps =True, use_sec = True)
    del ara_uproot

    # sensorHk info
    run_info = run_info_loader(station, run, analyze_blind_dat = analyze_blind_dat)
    sensor_dat = run_info.get_data_path(file_type = 'sensorHk', return_none = True, verbose = True)
    ara_sensorHk_uproot = ara_sensorHk_uproot_loader(sensor_dat)
    atri_volt, atri_curr, dda_volt, dda_curr, dda_temp, tda_volt, tda_curr, tda_temp = ara_sensorHk_uproot.get_daq_sensor_info()
    sensor_unix_time = ara_sensorHk_uproot.unix_time
    del sensor_dat, ara_sensorHk_uproot

    # event Hk info
    event_dat = run_info.get_data_path(file_type = 'eventHk', return_none = True, verbose = True)
    ara_eventHk_uproot = ara_eventHk_uproot_loader(event_dat)
    sel_dat_type = np.array([0, 4, 10, 11, 12], dtype = int)
    results = ara_eventHk_uproot.get_eventHk_info(use_prescale = True)
    l1_rate = results[sel_dat_type[0]]
    l1_thres = results[sel_dat_type[1]]
    dig_dead = results[sel_dat_type[2]]
    buff_dead = results[sel_dat_type[3]]
    tot_dead = results[sel_dat_type[4]]
    event_unix_time = ara_eventHk_uproot.unix_time
    event_pps_counter = ara_eventHk_uproot.pps_counter
    del run_info, station, run, event_dat, ara_eventHk_uproot, sel_dat_type, results

    print('Sub info collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'blk_len':blk_len,
            'rf_blk_len':rf_blk_len,
            'cal_blk_len':cal_blk_len,
            'soft_blk_len':soft_blk_len,
            'evt_num_sort':evt_num_sort,
            'trig_type_sort':trig_type_sort,
            'unix_time_sort':unix_time_sort,
            'pps_number_sort':pps_number_sort,
            'time_stamp_sort':time_stamp_sort,
            'pps_number_sort_reset':pps_number_sort_reset,
            'blk_len_sort':blk_len_sort,
            'rf_blk_len_sort':rf_blk_len_sort,
            'cal_blk_len_sort':cal_blk_len_sort,
            'soft_blk_len_sort':soft_blk_len_sort,
            'unix_min_bins':unix_min_bins,
            'unix_min_bin_center':unix_min_bin_center,
            'unix_min_counts':unix_min_counts,
            'evt_min_rate_unix':evt_min_rate_unix,
            'rf_min_rate_unix':rf_min_rate_unix,
            'cal_min_rate_unix':cal_min_rate_unix,
            'soft_min_rate_unix':soft_min_rate_unix,
            'pps_min_bins':pps_min_bins,
            'pps_min_bin_center':pps_min_bin_center,
            'pps_min_counts':pps_min_counts,
            'evt_min_rate_pps':evt_min_rate_pps,
            'rf_min_rate_pps':rf_min_rate_pps,
            'cal_min_rate_pps':cal_min_rate_pps,
            'soft_min_rate_pps':soft_min_rate_pps,
            'unix_sec_bins':unix_sec_bins,
            'unix_sec_bin_center':unix_sec_bin_center,
            'unix_sec_counts':unix_sec_counts,
            'evt_sec_rate_unix':evt_sec_rate_unix,
            'rf_sec_rate_unix':rf_sec_rate_unix,
            'cal_sec_rate_unix':cal_sec_rate_unix,
            'soft_sec_rate_unix':soft_sec_rate_unix,
            'pps_sec_bins':pps_sec_bins,
            'pps_sec_bin_center':pps_sec_bin_center,
            'pps_sec_counts':pps_sec_counts,
            'evt_sec_rate_pps':evt_sec_rate_pps,
            'rf_sec_rate_pps':rf_sec_rate_pps,
            'cal_sec_rate_pps':cal_sec_rate_pps,
            'soft_sec_rate_pps':soft_sec_rate_pps,
            'sensor_unix_time':sensor_unix_time,
            'atri_volt':atri_volt,
            'atri_curr':atri_curr,
            'dda_volt':dda_volt,
            'dda_curr':dda_curr,
            'dda_temp':dda_temp,
            'tda_volt':tda_volt,
            'tda_curr':tda_curr,
            'tda_temp':tda_temp,
            'trig_ch':trig_ch,
            'event_unix_time':event_unix_time,
            'event_pps_counter':event_pps_counter,
            'l1_rate':l1_rate,
            'l1_thres':l1_thres,
            'dig_dead':dig_dead,
            'buff_dead':buff_dead,
            'tot_dead':tot_dead}





