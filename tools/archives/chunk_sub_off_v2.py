import numpy as np
from tqdm import tqdm

def sub_off_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting sub off starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    unix_time = ara_uproot.unix_time
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)  
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1)
    
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}')
    del qual_cut_sum, ara_qual, ara_uproot

    # output arr
    sub_off = np.full((num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kJustPedWithOut1stBlockAndBadSamples)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            sub_off[ant, evt] = np.nanmedian(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt() 
    del ara_root, num_evts, daq_qual_sum, num_ants

    #std
    sub_std = np.nanstd(sub_off, axis = 1)
    sub_rf_copy = np.copy(sub_off)
    sub_rf_copy[:, trig_type != 0] = np.nan
    sub_rf_std = np.nanstd(sub_rf_copy, axis = 1)
    sub_rf_cut_copy = np.copy(sub_off)
    sub_rf_cut_copy[:, ~clean_evt_idx] = np.nan
    sub_rf_cut_std = np.nanstd(sub_rf_cut_copy, axis = 1)

    # first/end diff
    min_2nd_idx = np.logical_and(unix_time > unix_time[0] + 59, unix_time < unix_time[0] + 180)
    min_last_idx = unix_time > unix_time[-1] - 120

    def get_sub_diff(dat):
        sub_2nd_min = np.copy(dat)
        sub_last_min = np.copy(dat)
        sub_2nd_min[:, ~min_2nd_idx] = np.nan
        sub_last_min[:, ~min_last_idx] = np.nan
        sub_2nd_min_medi = np.nanmedian(sub_2nd_min, axis = 1)
        sub_last_min_medi = np.nanmedian(sub_last_min, axis = 1)
        sub_diff = np.abs(sub_2nd_min_medi - sub_last_min_medi)
        del sub_2nd_min, sub_last_min, sub_2nd_min_medi, sub_last_min_medi
        return sub_diff

    sub_diff = get_sub_diff(sub_off)
    sub_rf_diff = get_sub_diff(sub_rf_copy)
    sub_rf_cut_diff = get_sub_diff(sub_rf_cut_copy)
    del sub_rf_copy, sub_rf_cut_copy, min_2nd_idx, min_last_idx

    bit_range = np.arange(-200, 200)
    bit_bins = np.linspace(-200, 200, 200*2 + 1)
    ara_hist = hist_loader(bit_bins)
    bit_bin_center = ara_hist.bin_x_center    
    sub_hist = ara_hist.get_1d_hist(sub_off)
    sub_rf_hist = ara_hist.get_1d_hist(sub_off, cut = trig_type != 0)
    sub_rf_w_cut_hist = ara_hist.get_1d_hist(sub_off, cut = ~clean_evt_idx)
    del ara_hist
    
    unix_min = (unix_time - unix_time[0]).astype(float) / 60
    min_range = np.arange(0, 360)
    min_bins = np.linspace(0, 360, 360 + 1)
    ara_hist = hist_loader(min_bins, bit_bins)
    min_bin_center = ara_hist.bin_x_center
    sub_2d_hist = ara_hist.get_sub_off_2d_hist(unix_min, sub_off)
    sub_rf_2d_hist = ara_hist.get_sub_off_2d_hist(unix_min, sub_off, cut = trig_type != 0)
    sub_rf_w_cut_2d_hist = ara_hist.get_sub_off_2d_hist(unix_min, sub_off, cut = ~clean_evt_idx)
    del ara_hist, unix_min, clean_evt_idx
   
    print('Sub off collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'sub_off':sub_off,
            'sub_std':sub_std,
            'sub_rf_std':sub_rf_std,
            'sub_rf_cut_std':sub_rf_cut_std,
            'sub_diff':sub_diff,
            'sub_rf_diff':sub_rf_diff,
            'sub_rf_cut_diff':sub_rf_cut_diff,
            'bit_range':bit_range,
            'bit_bins':bit_bins,
            'bit_bin_center':bit_bin_center,
            'sub_hist':sub_hist,
            'sub_rf_hist':sub_rf_hist,
            'sub_rf_w_cut_hist':sub_rf_w_cut_hist,
            'min_range':min_range,
            'min_bins':min_bins,
            'min_bin_center':min_bin_center,
            'sub_2d_hist':sub_2d_hist,
            'sub_rf_2d_hist':sub_rf_2d_hist,
            'sub_rf_w_cut_2d_hist':sub_rf_w_cut_2d_hist}


