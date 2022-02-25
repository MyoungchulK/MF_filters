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

    # output array
    sec_to_min = 60
    unix_time_min = unix_time // sec_to_min
    unix_time_min_len = unix_time_min[-1] - unix_time_min[0] + 1

    evt_rate_bins = np.linspace(unix_time_min[0] * sec_to_min, unix_time_min[-1] * sec_to_min, unix_time_min_len, dtype =int)
    last_unix_time = unix_time[-1]
    if evt_rate_bins[-1] != last_unix_time:
        evt_rate_bins = np.append(evt_rate_bins, last_unix_time)
    sec_to_min_arr = np.diff(evt_rate_bins)
    del sec_to_min, unix_time_min_len, last_unix_time

    evt_rate = np.histogram(unix_time, bins = evt_rate_bins)[0] / sec_to_min_arr
    rf_evt_rate = np.histogram(unix_time[trig_type == 0], bins = evt_rate_bins)[0] / sec_to_min_arr
    cal_evt_rate = np.histogram(unix_time[trig_type == 1], bins = evt_rate_bins)[0] / sec_to_min_arr
    soft_evt_rate = np.histogram(unix_time[trig_type == 2], bins = evt_rate_bins)[0] / sec_to_min_arr
    del ara_uproot

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'unix_time_min':unix_time_min,
            'sec_to_min_arr':sec_to_min_arr,
            'evt_rate_bins':evt_rate_bins,
            'evt_rate':evt_rate,
            'rf_evt_rate':rf_evt_rate,
            'cal_evt_rate':cal_evt_rate,
            'soft_evt_rate':soft_evt_rate}





