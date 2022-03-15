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
    num_bits = ara_const.BUFFER_BIT_RANGE
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
    clean_rf_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_cal_idx = np.logical_and(qual_cut_sum == 0, trig_type == 1)
    clean_soft_idx = np.logical_and(qual_cut_sum == 0, trig_type == 2)
    clean_evt = evt_num[clean_rf_idx]
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

    bit_bins = np.linspace(-num_bits, num_bits, num_bits*2 + 1)
    ara_hist = hist_loader(bit_bins)
    bit_bin_center = ara_hist.bin_x_center    

    sub_hist = ara_hist.get_1d_hist(sub_off)
    rf_sub_hist = ara_hist.get_1d_hist(sub_off, cut = trig_type != 0)
    cal_sub_hist = ara_hist.get_1d_hist(sub_off, cut = trig_type != 1)
    soft_sub_hist = ara_hist.get_1d_hist(sub_off, cut = trig_type != 2)
    rf_sub_w_cut_hist = ara_hist.get_1d_hist(sub_off, cut = ~clean_rf_idx)
    cal_sub_w_cut_hist = ara_hist.get_1d_hist(sub_off, cut = ~clean_cal_idx)
    soft_sub_w_cut_hist = ara_hist.get_1d_hist(sub_off, cut = ~clean_soft_idx)
    del num_bits, ara_hist, clean_rf_idx, clean_cal_idx, clean_soft_idx

    print('Sub off collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'sub_off':sub_off,
            'bit_bins':bit_bins,
            'bit_bin_center':bit_bin_center,
            'sub_hist':sub_hist,
            'rf_sub_hist':rf_sub_hist,
            'cal_sub_hist':cal_sub_hist,
            'soft_sub_hist':soft_sub_hist,
            'rf_sub_w_cut_hist':rf_sub_w_cut_hist,
            'cal_sub_w_cut_hist':cal_sub_w_cut_hist,
            'soft_sub_w_cut_hist':soft_sub_w_cut_hist}




