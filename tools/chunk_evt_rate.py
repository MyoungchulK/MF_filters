import numpy as np
from tqdm import tqdm

def evt_rate_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting event rate starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del ara_qual    

    # event rate
    unix_min_bins, unix_min_counts, evt_rate_unix, rf_rate_unix, cal_rate_unix, soft_rate_unix = ara_uproot.get_event_rate(use_pps = False)
    pps_min_bins, pps_min_counts, evt_rate_pps, rf_rate_pps, cal_rate_pps, soft_rate_pps = ara_uproot.get_event_rate(use_pps =True)
    del ara_uproot

    unix_clean_min = np.histogram(unix_time, bins = unix_min_bins, weights = qual_cut_sum)[0].astype(int)
    evt_rate_unix_cut = np.copy(evt_rate_unix)
    evt_rate_unix_cut[unix_clean_min != 0] = np.nan
    rf_rate_unix_cut = np.copy(rf_rate_unix)
    rf_rate_unix_cut[unix_clean_min != 0] = np.nan
    cal_rate_unix_cut = np.copy(cal_rate_unix)
    cal_rate_unix_cut[unix_clean_min != 0] = np.nan
    soft_rate_unix_cut = np.copy(soft_rate_unix)
    soft_rate_unix_cut[unix_clean_min != 0] = np.nan
    
    pps_clean_min = np.histogram(pps_number, bins = pps_min_bins, weights = qual_cut_sum)[0].astype(int)
    evt_rate_pps_cut = np.copy(evt_rate_pps)
    evt_rate_pps_cut[pps_clean_min != 0] = np.nan   
    rf_rate_pps_cut = np.copy(rf_rate_pps)
    rf_rate_pps_cut[pps_clean_min != 0] = np.nan
    cal_rate_pps_cut = np.copy(cal_rate_pps)
    cal_rate_pps_cut[pps_clean_min != 0] = np.nan
    soft_rate_pps_cut = np.copy(soft_rate_pps)
    soft_rate_pps_cut[pps_clean_min != 0] = np.nan
    del qual_cut_sum, unix_clean_min, pps_clean_min

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut,
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
            'soft_rate_pps':soft_rate_pps,
            'evt_rate_unix_cut':evt_rate_unix_cut,
            'rf_rate_unix_cut':rf_rate_unix_cut,
            'cal_rate_unix_cut':cal_rate_unix_cut,
            'soft_rate_unix_cut':soft_rate_unix_cut,
            'evt_rate_pps_cut':evt_rate_pps_cut,
            'rf_rate_pps_cut':rf_rate_pps_cut,
            'cal_rate_pps_cut':cal_rate_pps_cut,
            'soft_rate_pps_cut':soft_rate_pps_cut}





