import numpy as np
from tqdm import tqdm

def blk_len_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting block length starts!')

    from tools.ara_constant import ara_const
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import qual_cut_loader

    ara_const = ara_const()
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    blk_len = (ara_uproot.read_win // num_ddas).astype(float)
    del num_ddas

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del ara_uproot, ara_qual
    
    #output array
    blk_len_cut = np.copy(blk_len)
    blk_len_cut[qual_cut_sum != 0] = np.nan
    rf_len = np.copy(blk_len)
    rf_len[trig_type != 0] = np.nan
    cal_len = np.copy(blk_len)
    cal_len[trig_type != 1] = np.nan
    soft_len = np.copy(blk_len)
    soft_len[trig_type != 2] = np.nan
    rf_len_cut = np.copy(blk_len)
    rf_len_cut[(trig_type != 0) | (qual_cut_sum != 0)] = np.nan
    cal_len_cut = np.copy(blk_len)
    cal_len_cut[(trig_type != 1) | (qual_cut_sum != 0)] = np.nan
    soft_len_cut = np.copy(blk_len)
    soft_len_cut[(trig_type != 2) | (qual_cut_sum != 0)] = np.nan
    del qual_cut_sum
    
    # hist output array
    blk_range = np.arange(50, dtype = int)
    blk_bins = np.linspace(0, 50, 50 + 1, dtype = int)
    blk_bin_center = (blk_bins[1:] + blk_bins[:-1]) / 2  
    blk_len_hist = np.histogram(blk_len, bins = blk_bins)[0].astype(int)
    rf_len_hist = np.histogram(rf_len, bins = blk_bins)[0].astype(int)
    cal_len_hist = np.histogram(cal_len, bins = blk_bins)[0].astype(int)
    soft_len_hist = np.histogram(soft_len, bins = blk_bins)[0].astype(int)
    blk_len_cut_hist = np.histogram(blk_len_cut, bins = blk_bins)[0].astype(int)
    rf_len_cut_hist = np.histogram(rf_len_cut, bins = blk_bins)[0].astype(int)
    cal_len_cut_hist = np.histogram(cal_len_cut, bins = blk_bins)[0].astype(int)
    soft_len_cut_hist = np.histogram(soft_len_cut, bins = blk_bins)[0].astype(int)

    unix_min = (unix_time - unix_time[0]).astype(float) / 60
    min_range = np.arange(0, 360)
    min_bins = np.linspace(0, 360, 360 + 1)
    min_bin_center = (min_bins[1:] + min_bins[:-1]) / 2
    blk_len_hist2d = np.histogram2d(unix_min, blk_len, bins = (min_bins, blk_bins))[0].astype(int)
    rf_len_hist2d = np.histogram2d(unix_min, rf_len, bins = (min_bins, blk_bins))[0].astype(int)
    cal_len_hist2d = np.histogram2d(unix_min, cal_len, bins = (min_bins, blk_bins))[0].astype(int)
    soft_len_hist2d = np.histogram2d(unix_min, soft_len, bins = (min_bins, blk_bins))[0].astype(int)
    blk_len_cut_hist2d = np.histogram2d(unix_min, blk_len_cut, bins = (min_bins, blk_bins))[0].astype(int)
    rf_len_cut_hist2d = np.histogram2d(unix_min, rf_len_cut, bins = (min_bins, blk_bins))[0].astype(int)
    cal_len_cut_hist2d = np.histogram2d(unix_min, cal_len_cut, bins = (min_bins, blk_bins))[0].astype(int)
    soft_len_cut_hist2d = np.histogram2d(unix_min, soft_len_cut, bins = (min_bins, blk_bins))[0].astype(int)

    print('Block length collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'blk_len':blk_len,
            'rf_len':rf_len,
            'cal_len':cal_len,
            'soft_len':soft_len,
            'blk_len_cut':blk_len_cut,
            'rf_len_cut':rf_len_cut,
            'cal_len_cut':cal_len_cut,
            'soft_len_cut':soft_len_cut,
            'blk_range':blk_range,
            'blk_bins':blk_bins,
            'blk_bin_center':blk_bin_center,
            'blk_len_hist':blk_len_hist,
            'rf_len_hist':rf_len_hist,
            'cal_len_hist':cal_len_hist,
            'soft_len_hist':soft_len_hist,
            'blk_len_cut_hist':blk_len_cut_hist,
            'rf_len_cut_hist':rf_len_cut_hist,
            'cal_len_cut_hist':cal_len_cut_hist,
            'soft_len_cut_hist':soft_len_cut_hist,
            'unix_min':unix_min,
            'min_range':min_range,
            'min_bins':min_bins,
            'min_bin_center':min_bin_center,
            'blk_len_hist2d':blk_len_hist2d,
            'rf_len_hist2d':rf_len_hist2d,
            'cal_len_hist2d':cal_len_hist2d,
            'soft_len_hist2d':soft_len_hist2d,
            'blk_len_cut_hist2d':blk_len_cut_hist2d,
            'rf_len_cut_hist2d':rf_len_cut_hist2d,
            'cal_len_cut_hist2d':cal_len_cut_hist2d,
            'soft_len_cut_hist2d':soft_len_cut_hist2d}





