import numpy as np
from tqdm import tqdm

def std_adc_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    ara_geom = ara_geom_loader(ara_uproot.station_id, ara_uproot.year, verbose = True)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    ele_ch = ara_geom.get_ele_ch_idx()
    del ara_geom

    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1)
    del ara_qual, ara_uproot

    rf_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    cal_idx = np.logical_and(qual_cut_sum == 0, trig_type == 1)
    soft_idx = np.logical_and(qual_cut_sum == 0, trig_type == 2)
    clean_evt = evt_num[rf_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum

    # output array
    std = np.full((num_eles, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib14to20_Bug)
        for ant in range(num_eles):
            raw_v = ara_root.get_ele_ch_wf(ant)[1]
            std[ant, evt] = np.nanstd(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_root, num_evts, daq_qual_sum, num_eles

    std_range = np.arange(0, 100, 0.1)
    std_bins = np.linspace(0, 100, 1000 + 1)
    ara_hist = hist_loader(std_bins)
    std_bin_center = ara_hist.bin_x_center

    std_hist = ara_hist.get_1d_hist(std)
    std_rf_hist = ara_hist.get_1d_hist(std, cut = trig_type != 0)
    std_cal_hist = ara_hist.get_1d_hist(std, cut = trig_type != 1)
    std_soft_hist = ara_hist.get_1d_hist(std, cut = trig_type != 2)
    std_rf_w_cut_hist  = ara_hist.get_1d_hist(std, cut = ~rf_idx)
    std_cal_w_cut_hist  = ara_hist.get_1d_hist(std, cut = ~cal_idx)
    std_soft_w_cut_hist  = ara_hist.get_1d_hist(std, cut = ~soft_idx)
    del ara_hist, rf_idx, cal_idx, soft_idx

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'ele_ch':ele_ch,
            'total_qual_cut':total_qual_cut,
            'clean_evt':clean_evt,
            'std':std,
            'std_range':std_range,
            'std_bins':std_bins,
            'std_bin_center':std_bin_center,
            'std_hist':std_hist,
            'std_rf_hist':std_rf_hist,
            'std_cal_hist':std_cal_hist,
            'std_soft_hist':std_soft_hist,
            'std_rf_w_cut_hist':std_rf_w_cut_hist,
            'std_cal_w_cut_hist':std_cal_w_cut_hist,
            'std_soft_w_cut_hist':std_soft_w_cut_hist}



