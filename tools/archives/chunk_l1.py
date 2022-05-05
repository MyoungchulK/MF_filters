import numpy as np

def l1_collector(Data, Ped, Station, Run, Year, analyze_blind_dat = False):

    print('Collecting L1 info starts!')

    from tools.ara_constant import ara_const
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_eventHk_uproot_loader
    from tools.ara_quality_cut import qual_cut_loader
    
    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # load data 
    ara_eventHk_uproot = ara_eventHk_uproot_loader(Data)
    l1_rate, l1_thres = ara_eventHk_uproot.get_l1_info()
    unix_time = ara_eventHk_uproot.unix_time

    if ara_eventHk_uproot.empty_file_error: 
        ara_geom = ara_geom_loader(Station, Year, verbose = True)
    else:
        ara_geom = ara_geom_loader(ara_eventHk_uproot.station_id, ara_eventHk_uproot.year, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    del ara_geom

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    if ara_eventHk_uproot.empty_file_error:
        total_qual_cut = ara_qual.load_qual_cut_result(Station, Run)
    else:
        total_qual_cut = ara_qual.load_qual_cut_result(ara_eventHk_uproot.station_id, ara_eventHk_uproot.run)
    evt_unix_time = ara_qual.unix_time
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del ara_qual

    # l1 info.
    l1_rate_cut = np.copy(l1_rate).astype(float)
    l1_thres_cut = np.copy(l1_thres).astype(float)
    if ara_eventHk_uproot.empty_file_error:
        pass 
    else:
        unix_bins = np.append(unix_time, unix_time[-1]+1)
        clean_secs = np.histogram(evt_unix_time, bins = unix_bins, weights = qual_cut_sum)[0].astype(int)
        l1_rate_cut[clean_secs != 0] = np.nan
        l1_thres_cut[clean_secs != 0] = np.nan
        del clean_secs, unix_bins
    del ara_eventHk_uproot, qual_cut_sum, evt_unix_time

    # histogram
    l1_range = np.arange(0,100000,100)
    l1_bins = np.linspace(0,100000,1000+1)
    l1_bin_center = (l1_bins[1:] + l1_bins[:-1]) / 2
    l1_rate_hist = np.full((len(l1_bin_center), num_eles), 0, dtype = int)
    l1_thres_hist = np.copy(l1_rate_hist)
    l1_rate_cut_hist = np.copy(l1_rate_hist)
    l1_thres_cut_hist = np.copy(l1_rate_hist)
    unix_min = (unix_time - unix_time[0]).astype(float) / 60
    min_range = np.arange(0, 360)
    min_bins = np.linspace(0, 360, 360 + 1)
    min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2
    l1_rate_hist2d = np.full((len(min_bin_center), len(l1_bin_center), num_eles), 0, dtype = int)
    l1_thres_hist2d = np.copy(l1_rate_hist2d)
    l1_rate_cut_hist2d = np.copy(l1_rate_hist2d)
    l1_thres_cut_hist2d = np.copy(l1_rate_hist2d)
    for ch in range(num_eles):
        l1_rate_hist[:, ch] = np.histogram(l1_rate[:, ch], bins = l1_bins)[0].astype(int)
        l1_rate_cut_hist[:, ch] = np.histogram(l1_rate_cut[:, ch], bins = l1_bins)[0].astype(int)
        l1_thres_hist[:, ch] = np.histogram(l1_thres[:, ch], bins = l1_bins)[0].astype(int)
        l1_thres_cut_hist[:, ch] = np.histogram(l1_thres_cut[:, ch], bins = l1_bins)[0].astype(int)
        l1_rate_hist2d[:, :, ch] = np.histogram2d(unix_min, l1_rate[:, ch], bins = (min_bins, l1_bins))[0].astype(int)
        l1_rate_cut_hist2d[:, :, ch] = np.histogram2d(unix_min, l1_rate_cut[:, ch], bins = (min_bins, l1_bins))[0].astype(int)
        l1_thres_hist2d[:, :, ch] = np.histogram2d(unix_min, l1_thres[:, ch], bins = (min_bins, l1_bins))[0].astype(int)
        l1_thres_cut_hist2d[:, :, ch] = np.histogram2d(unix_min, l1_thres_cut[:, ch], bins = (min_bins, l1_bins))[0].astype(int)
    del num_eles

    def get_2d_max(dat_2d, use_min = False):
        dat_max = np.copy(dat_2d)
        dat_max[dat_max != 0] = 1
        dat_max *= l1_range[np.newaxis, :, np.newaxis]
        if use_min:
            dat_max = np.nanmin(dat_max, axis = 1)
        else:
            dat_max = np.nanmax(dat_max, axis = 1)
        return dat_max

    # roll max/min
    l1_rate_hist2d_max = get_2d_max(l1_rate_hist2d)
    l1_thres_hist2d_max = get_2d_max(l1_thres_hist2d)
    l1_rate_cut_hist2d_max = get_2d_max(l1_rate_cut_hist2d)
    l1_thres_cut_hist2d_max = get_2d_max(l1_thres_cut_hist2d)
    l1_rate_hist2d_min = get_2d_max(l1_rate_hist2d, use_min = True)
    l1_thres_hist2d_min = get_2d_max(l1_thres_hist2d, use_min = True)
    l1_rate_cut_hist2d_min = get_2d_max(l1_rate_cut_hist2d, use_min = True)
    l1_thres_cut_hist2d_min = get_2d_max(l1_thres_cut_hist2d, use_min = True)

    print('l1 info collecting is done!')

    return {'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'ele_ch':ele_ch,
            'trig_ch':trig_ch,
            'l1_rate':l1_rate,
            'l1_thres':l1_thres,
            'l1_rate_cut':l1_rate_cut,
            'l1_thres_cut':l1_thres_cut,
            'l1_range':l1_range,
            'l1_bins':l1_bins,
            'l1_bin_center':l1_bin_center,
            'l1_rate_hist':l1_rate_hist,
            'l1_thres_hist':l1_thres_hist,
            'l1_rate_cut_hist':l1_rate_cut_hist,
            'l1_thres_cut_hist':l1_thres_cut_hist,
            'unix_min':unix_min,
            'min_range':min_range,
            'min_bins':min_bins,
            'min_bin_center':min_bin_center,
            'l1_rate_hist2d':l1_rate_hist2d,
            'l1_thres_hist2d':l1_thres_hist2d,
            'l1_rate_cut_hist2d':l1_rate_cut_hist2d,
            'l1_thres_cut_hist2d':l1_thres_cut_hist2d,
            'l1_rate_hist2d_max':l1_rate_hist2d_max,
            'l1_thres_hist2d_max':l1_thres_hist2d_max,
            'l1_rate_cut_hist2d_max':l1_rate_cut_hist2d_max,
            'l1_thres_cut_hist2d_max':l1_thres_cut_hist2d_max,
            'l1_rate_hist2d_min':l1_rate_hist2d_min,
            'l1_thres_hist2d_min':l1_thres_hist2d_min,
            'l1_rate_cut_hist2d_min':l1_rate_cut_hist2d_min,
            'l1_thres_cut_hist2d_min':l1_thres_cut_hist2d_min}





