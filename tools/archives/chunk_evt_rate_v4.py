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

    # event rate
    unix_min_bins, unix_min_counts, evt_rate_unix, rf_rate_unix, cal_rate_unix, soft_rate_unix = ara_uproot.get_event_rate(use_pps = False)
    pps_min_bins, pps_min_counts, evt_rate_pps, rf_rate_pps, cal_rate_pps, soft_rate_pps = ara_uproot.get_event_rate(use_pps =True)
    del ara_uproot

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'unix_min_bins':unix_min_bins,
            'unix_min_counts':unix_min_counts,
            'evt_rate_unix':evt_rate_unix,
            'rf_rate_unix':rf_rate_unix,
            'cal_rate_unix':cal_rate_unix,
            'soft_rate_unix':soft_rate_unix,
            'pps_min_bins':pps_min_bins,
            'pps_min_counts':pps_min_counts,
            'evt_rate_pps':evt_rate_pps,
            'rf_rate_pps':rf_rate_pps,
            'cal_rate_pps':cal_rate_pps,
            'soft_rate_pps':soft_rate_pps}




