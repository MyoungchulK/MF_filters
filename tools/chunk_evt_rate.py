import numpy as np
from tqdm import tqdm

def evt_rate_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting event rate starts!')

    from tools.ara_data_load import ara_uproot_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp

    # remove nasty unix error
    evt_sort_idx = np.argsort(evt_num)
    evt_num_sort = evt_num[evt_sort_idx]
    pps_number_sort = pps_number[evt_sort_idx]
    unix_time_sort = unix_time[evt_sort_idx]
    time_stamp_sort = time_stamp[evt_sort_idx]
    trig_type_sort = trig_type[evt_sort_idx]

    # remove pps reset
    pps_number_sort_reset = ara_uproot.get_reset_pps_number(use_evt_num_sort = True)

    # event rate
    unix_min_bins, unix_min_bin_center, unix_min_counts, evt_rate_unix, rf_rate_unix, cal_rate_unix, soft_rate_unix = ara_uproot.get_event_rate(use_pps = False)
    pps_min_bins, pps_min_bin_center, pps_min_counts, evt_rate_pps, rf_rate_pps, cal_rate_pps, soft_rate_pps = ara_uproot.get_event_rate(use_pps =True)
    del ara_uproot

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'evt_num_sort':evt_num_sort,
            'trig_type_sort':trig_type_sort,
            'unix_time_sort':unix_time_sort,
            'pps_number_sort':pps_number_sort,
            'time_stamp_sort':time_stamp_sort,
            'pps_number_sort_reset':pps_number_sort_reset,
            'unix_min_bins':unix_min_bins,
            'unix_min_bin_center':unix_min_bin_center,
            'unix_min_counts':unix_min_counts,
            'evt_rate_unix':evt_rate_unix,
            'rf_rate_unix':rf_rate_unix,
            'cal_rate_unix':cal_rate_unix,
            'soft_rate_unix':soft_rate_unix,
            'pps_min_bins':pps_min_bins,
            'pps_min_bin_center':pps_min_bin_center,
            'pps_min_counts':pps_min_counts,
            'evt_rate_pps':evt_rate_pps,
            'rf_rate_pps':rf_rate_pps,
            'cal_rate_pps':cal_rate_pps,
            'soft_rate_pps':soft_rate_pps}




