import numpy as np
from tqdm import tqdm

def std_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    ele_ch = ara_root.ara_geom.get_ele_ch_idx()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = ara_qual.total_qual_cut_sum
    daq_qual_sum = ara_qual.daq_qual_cut_sum
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum, ara_qual, ara_uproot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_ele_ch = True)

    # output array
    std = np.full((num_eles, num_evts), np.nan, dtype = float)
    std_bp = np.copy(std)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        for ant in range(num_eles):
            raw_t, raw_v = ara_root.get_ele_ch_wf(ant)
            std[ant, evt] = np.nanstd(raw_v)
            if clean_evt_idx[evt]:
               wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        if clean_evt_idx[evt]:
            bp_v = wf_int.pad_v
            std_bp[:, evt] = np.nanstd(bp_v, axis = 0)
            del bp_v
        ara_root.del_usefulEvt()
    del ara_root, num_evts, daq_qual_sum, num_eles, wf_int

    std_range = np.arange(0, 100, 0.1)
    std_bins = np.linspace(0, 100, 1000 + 1)
    ara_hist = hist_loader(std_bins)
    std_bin_center = ara_hist.bin_x_center

    std_hist = ara_hist.get_1d_hist(std)
    std_rf_hist = ara_hist.get_1d_hist(std, cut = trig_type != 0)
    std_rf_w_cut_hist  = ara_hist.get_1d_hist(std, cut = ~clean_evt_idx)
    std_bp_rf_w_cut_hist  = ara_hist.get_1d_hist(std_bp, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'ele_ch':ele_ch,
            'total_qual_cut':total_qual_cut,
            'clean_evt':clean_evt,
            'std':std,
            'std_bp':std_bp,
            'std_range':std_range,
            'std_bins':std_bins,
            'std_bin_center':std_bin_center,
            'std_hist':std_hist,
            'std_rf_hist':std_rf_hist,
            'std_rf_w_cut_hist':std_rf_w_cut_hist,
            'std_bp_rf_w_cut_hist':std_bp_rf_w_cut_hist}



