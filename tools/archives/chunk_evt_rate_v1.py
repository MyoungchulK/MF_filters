import numpy as np
from tqdm import tqdm

def rate_calculator(time_temp, trig_type, use_pps = False):

    sec_to_min = 60
    time = np.copy(time_temp)

    if use_pps:
        time_reset_point = np.where(np.diff(time) < 0)[0]
        if len(time_reset_point) > 0:
            pps_limit = 65536
            time[time_reset_point[0]+1:] += pps_limit
        time_unique = np.sort(np.unique(time))
    else:
        time_unique = np.sort(np.unique(time))
    time_bins = np.arange(np.nanmin(time_unique), np.nanmax(time_unique)+1, sec_to_min, dtype = int)
    time_bins = time_bins.astype(float)
    time_bins -= 0.5
    time_bins = np.append(time_bins, time_unique[-1] + 0.5)

    num_secs = np.diff(time_bins).astype(int)

    evt_rate = np.histogram(time, bins = time_bins)[0] / num_secs
    rf_evt_rate = np.histogram(time[trig_type == 0], bins = time_bins)[0] / num_secs
    cal_evt_rate = np.histogram(time[trig_type == 1], bins = time_bins)[0] / num_secs
    soft_evt_rate = np.histogram(time[trig_type == 2], bins = time_bins)[0] / num_secs

    return time_bins, num_secs, evt_rate, rf_evt_rate, cal_evt_rate, soft_evt_rate

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

    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    bad_event_number = np.copy(total_qual_cut[:,9])
    del ara_qual

    if np.any(bad_event_number != 0):
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
        bad_rate_unix = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_rf_rate_unix = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_cal_rate_unix = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_soft_rate_unix = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_rate_pps = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_rf_rate_pps = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_cal_rate_pps = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_soft_rate_pps = np.full((1, total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_total_rate_unix = np.full((1), np.nan, dtype = float)   
        bad_total_rf_rate_unix = np.full((1), np.nan, dtype = float)   
        bad_total_cal_rate_unix = np.full((1), np.nan, dtype = float)   
        bad_total_soft_rate_unix = np.full((1), np.nan, dtype = float)   
        bad_total_rate_pps = np.full((1), np.nan, dtype = float)
        bad_total_rf_rate_pps = np.full((1), np.nan, dtype = float)
        bad_total_cal_rate_pps = np.full((1), np.nan, dtype = float)
        bad_total_soft_rate_pps = np.full((1), np.nan, dtype = float)

    else:
        time_bins_unix, num_secs_unix, evt_rate_unix, rf_evt_rate_unix, cal_evt_rate_unix, soft_evt_rate_unix = rate_calculator(unix_time, trig_type, use_pps = False)
        time_bins_pps, num_secs_pps, evt_rate_pps, rf_evt_rate_pps, cal_evt_rate_pps, soft_evt_rate_pps = rate_calculator(pps_number, trig_type, use_pps = True)

        bad_rate_unix = np.full((len(evt_rate_unix), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_rf_rate_unix = np.full((len(rf_evt_rate_unix), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_cal_rate_unix = np.full((len(cal_evt_rate_unix), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_soft_rate_unix = np.full((len(soft_evt_rate_unix), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_rate_pps = np.full((len(evt_rate_pps), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_rf_rate_pps = np.full((len(rf_evt_rate_pps), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_cal_rate_pps = np.full((len(cal_evt_rate_pps), total_qual_cut.shape[1]), np.nan, dtype = float)
        bad_soft_rate_pps = np.full((len(soft_evt_rate_pps), total_qual_cut.shape[1]), np.nan, dtype = float)

        tot_qual = np.nansum(total_qual_cut, axis = 1)
        bad_unix_qual = np.histogram(unix_time[tot_qual != 0], bins = time_bins_unix)[0].astype(int)
        bad_pps_qual = np.histogram(pps_number[tot_qual != 0], bins = time_bins_pps)[0].astype(int)
        bad_total_rate_unix = np.copy(evt_rate_unix).astype(float)
        bad_total_rf_rate_unix = np.copy(rf_evt_rate_unix).astype(float)
        bad_total_cal_rate_unix = np.copy(cal_evt_rate_unix).astype(float)
        bad_total_soft_rate_unix = np.copy(soft_evt_rate_unix).astype(float)
        bad_total_rate_unix[bad_unix_qual == 0] = np.nan
        bad_total_rf_rate_unix[bad_unix_qual == 0] = np.nan
        bad_total_cal_rate_unix[bad_unix_qual == 0] = np.nan
        bad_total_soft_rate_unix[bad_unix_qual == 0] = np.nan
        bad_total_rate_pps = np.copy(evt_rate_pps).astype(float)
        bad_total_rf_rate_pps = np.copy(rf_evt_rate_pps).astype(float)
        bad_total_cal_rate_pps = np.copy(cal_evt_rate_pps).astype(float)
        bad_total_soft_rate_pps = np.copy(soft_evt_rate_pps).astype(float)
        bad_total_rate_pps[bad_pps_qual == 0] = np.nan
        bad_total_rf_rate_pps[bad_pps_qual == 0] = np.nan
        bad_total_cal_rate_pps[bad_pps_qual == 0] = np.nan
        bad_total_soft_rate_pps[bad_pps_qual == 0] = np.nan
        del bad_unix_qual, bad_pps_qual

        for q in tqdm(range(total_qual_cut.shape[1])):
            bad_unix_qual = np.histogram(unix_time[total_qual_cut[:, q] != 0], bins = time_bins_unix)[0].astype(int)
            bad_pps_qual = np.histogram(pps_number[total_qual_cut[:, q] != 0], bins = time_bins_pps)[0].astype(int)

            bad_rate_unix[:, q] = np.copy(evt_rate_unix).astype(float)
            bad_rate_unix[bad_unix_qual == 0, q] = np.nan
            bad_rf_rate_unix[:, q] = np.copy(rf_evt_rate_unix).astype(float)
            bad_rf_rate_unix[bad_unix_qual == 0, q] = np.nan
            bad_cal_rate_unix[:, q] = np.copy(cal_evt_rate_unix).astype(float)
            bad_cal_rate_unix[bad_unix_qual == 0, q] = np.nan
            bad_soft_rate_unix[:, q] = np.copy(soft_evt_rate_unix).astype(float)
            bad_soft_rate_unix[bad_unix_qual == 0, q] = np.nan

            bad_rate_pps[:, q] = np.copy(evt_rate_pps).astype(float)
            bad_rate_pps[bad_pps_qual == 0, q] = np.nan
            bad_rf_rate_pps[:, q] = np.copy(rf_evt_rate_pps).astype(float)
            bad_rf_rate_pps[bad_pps_qual == 0, q] = np.nan
            bad_cal_rate_pps[:, q] = np.copy(cal_evt_rate_pps).astype(float)
            bad_cal_rate_pps[bad_pps_qual == 0, q] = np.nan
            bad_soft_rate_pps[:, q] = np.copy(soft_evt_rate_pps).astype(float)
            bad_soft_rate_pps[bad_pps_qual == 0, q] = np.nan
    del ara_uproot

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut,
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
            'soft_evt_rate_pps':soft_evt_rate_pps,
            'bad_rate_unix':bad_rate_unix,
            'bad_rf_rate_unix':bad_rf_rate_unix,
            'bad_cal_rate_unix':bad_cal_rate_unix,
            'bad_soft_rate_unix':bad_soft_rate_unix,
            'bad_rate_pps':bad_rate_pps,
            'bad_rf_rate_pps':bad_rf_rate_pps,
            'bad_cal_rate_pps':bad_cal_rate_pps,
            'bad_soft_rate_pps':bad_soft_rate_pps,
            'bad_total_rate_unix':bad_total_rate_unix,
            'bad_total_rf_rate_unix':bad_total_rf_rate_unix,
            'bad_total_cal_rate_unix':bad_total_cal_rate_unix,
            'bad_total_soft_rate_unix':bad_total_soft_rate_unix,
            'bad_total_rate_pps':bad_total_rate_pps,
            'bad_total_rf_rate_pps':bad_total_rf_rate_pps,
            'bad_total_cal_rate_pps':bad_total_cal_rate_pps,
            'bad_total_soft_rate_pps':bad_total_soft_rate_pps}





