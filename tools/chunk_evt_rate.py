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
    evt_rate_bins = np.linspace(np.nanmin(unix_time_min), np.nanmax(unix_time_min), np.nanmax(unix_time_min) - np.nanmin(unix_time_min) + 1, dtype = int)
    evt_rate = np.histogram(unix_time_min, bins = evt_rate_bins)[0] / sec_to_min
    rf_evt_rate = np.histogram(unix_time_min[trig_type == 0], bins = evt_rate_bins)[0] / sec_to_min
    cal_evt_rate = np.histogram(unix_time_min[trig_type == 1], bins = evt_rate_bins)[0] / sec_to_min
    soft_evt_rate = np.histogram(unix_time_min[trig_type == 2], bins = evt_rate_bins)[0] / sec_to_min
    del ara_uproot

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'unix_time_min':unix_time_min,
            'evt_rate_bins':evt_rate_bins,
            'evt_rate':evt_rate,
            'rf_evt_rate':rf_evt_rate,
            'cal_evt_rate':cal_evt_rate,
            'soft_evt_rate':soft_evt_rate}





