import numpy as np
from tqdm import tqdm

def evt_rate_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting event rate starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import pre_qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number

    pre_ara_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    bad_event_number = pre_ara_qual.get_bad_event_number()
    del pre_ara_qual

    if np.count_nonzero(bad_event_number == 0) == 0:
        time_bins_unix = np.full((1), np.nan, dtype = float)
        num_secs_unix = np.copy(time_bins_unix)
        evt_rate_unix = np.copy(time_bins_unix)
        rf_evt_rate_unix = np.copy(time_bins_unix)
        cal_evt_rate_unix = np.copy(time_bins_unix)
        soft_evt_rate_unix = np.copy(time_bins_unix)
        time_bins_pps = np.copy(time_bins_unix)
        num_secs_pps = np.copy(time_bins_unix)
        evt_rate_pps = np.copy(time_bins_unix)
        rf_evt_rate_pps = np.copy(time_bins_unix)
        cal_evt_rate_pps = np.copy(time_bins_unix)
        soft_evt_rate_pps = np.copy(time_bins_unix)
    else:
        unix_time_new = unix_time[bad_event_number == 0] 
        pps_number_new = pps_number[bad_event_number == 0]
        trig_type_new = trig_type[bad_event_number == 0]

        time_bins_unix, num_secs_unix, evt_rate_unix, rf_evt_rate_unix, cal_evt_rate_unix, soft_evt_rate_unix = ara_uproot.get_event_rate(unix_time_new, trig_type_new, use_pps = False)
        time_bins_pps, num_secs_pps, evt_rate_pps, rf_evt_rate_pps, cal_evt_rate_pps, soft_evt_rate_pps = ara_uproot.get_event_rate(pps_number_new, trig_type_new, use_pps = True)
    del ara_uproot, unix_time_new, pps_number_new, trig_type_new

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'bad_event_number':bad_event_number,
            'time_bins_unix':time_bins_unix,
            'num_secs_unix':num_secs_unix,
            'evt_rate_unix':evt_rate_unix,
            'rf_evt_rate_unix':rf_evt_rate_unix,
            'cal_evt_rate_unix':cal_evt_rate_unix,
            'soft_evt_rate_unix':soft_evt_rate_unix,
            'time_bins_pps':time_bins_pps,
            'num_secs_pps':num_secs_pps,
            'evt_rate_pps':evt_rate_pps,
            'rf_evt_rate_pps':rf_evt_rate_pps,
            'cal_evt_rate_pps':cal_evt_rate_pps,
            'soft_evt_rate_pps':soft_evt_rate_pps}





