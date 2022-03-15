import numpy as np
from tqdm import tqdm

def peak_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting cw starts!')

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
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum, ara_qual, ara_uproot

    # output arr
    adc_max_min = np.full((2, num_ants, num_evts), np.nan, dtype = float)
    adc_abs_max = np.full((num_ants, num_evts), np.nan, dtype = float)
    mv_max_min = np.copy(adc_max_min)
    mv_abs_max = np.copy(adc_abs_max)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        

        if daq_qual_sum[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlockAndBadSamples)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]

            adc_max_min[0, ant, evt] = np.nanmax(raw_v)
            adc_max_min[1, ant, evt] = np.nanmin(raw_v)
            adc_abs_max[ant, evt] = np.nanmax(np.abs(raw_v))

            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        # loop over the antennas
        for ant in range(num_ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]

            mv_max_min[0, ant, evt] = np.nanmax(raw_v)
            mv_max_min[1, ant, evt] = np.nanmin(raw_v)
            mv_abs_max[ant, evt] = np.nanmax(np.abs(raw_v))

            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

    del ara_root, num_evts, daq_qual_sum, num_ants

    adc_bins = np.linspace(0, num_bits, num_bits + 1)
    ara_hist = hist_loader(adc_bins)
    adc_bin_center = ara_hist.bin_x_center
    adc_max_hist = ara_hist.get_1d_hist(adc_max_min[0])
    adc_max_rf_hist = ara_hist.get_1d_hist(adc_max_min[0], cut = trig_type != 0)
    adc_max_rf_w_cut_hist  = ara_hist.get_1d_hist(adc_max_min[0], cut = ~clean_evt_idx)
    adc_min_hist = ara_hist.get_1d_hist(adc_max_min[1])
    adc_min_rf_hist = ara_hist.get_1d_hist(adc_max_min[1], cut = trig_type != 0)
    adc_min_rf_w_cut_hist  = ara_hist.get_1d_hist(adc_max_min[1], cut = ~clean_evt_idx)
    adc_abs_max_hist = ara_hist.get_1d_hist(adc_abs_max)
    adc_abs_max_rf_hist = ara_hist.get_1d_hist(adc_abs_max, cut = trig_type != 0)
    adc_abs_max_rf_w_cut_hist  = ara_hist.get_1d_hist(adc_abs_max, cut = ~clean_evt_idx)
    del ara_hist

    mv_bins =  np.linspace(-num_bits//2, num_bits//2, num_bits + 1)
    ara_hist = hist_loader(mv_bins)
    mv_bin_center = ara_hist.bin_x_center
    mv_max_hist = ara_hist.get_1d_hist(mv_max_min[0])
    mv_max_rf_hist = ara_hist.get_1d_hist(mv_max_min[0], cut = trig_type != 0)
    mv_max_rf_w_cut_hist  = ara_hist.get_1d_hist(mv_max_min[0], cut = ~clean_evt_idx)
    mv_min_hist = ara_hist.get_1d_hist(mv_max_min[1])
    mv_min_rf_hist = ara_hist.get_1d_hist(mv_max_min[1], cut = trig_type != 0)
    mv_min_rf_w_cut_hist  = ara_hist.get_1d_hist(mv_max_min[1], cut = ~clean_evt_idx)
    mv_abs_max_hist = ara_hist.get_1d_hist(mv_abs_max)
    mv_abs_max_rf_hist = ara_hist.get_1d_hist(mv_abs_max, cut = trig_type != 0)
    mv_abs_max_rf_w_cut_hist  = ara_hist.get_1d_hist(mv_abs_max, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx, num_bits

    print('peak collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'adc_max_min':adc_max_min,
            'adc_abs_max':adc_abs_max,
            'mv_max_min':mv_max_min,
            'mv_abs_max':mv_abs_max,
            'adc_bins':adc_bins,
            'adc_bin_center':adc_bin_center,
            'adc_max_hist':adc_max_hist,
            'adc_max_rf_hist':adc_max_rf_hist,
            'adc_max_rf_w_cut_hist':adc_max_rf_w_cut_hist,
            'adc_min_hist':adc_min_hist,
            'adc_min_rf_hist':adc_min_rf_hist,
            'adc_min_rf_w_cut_hist':adc_min_rf_w_cut_hist,
            'adc_abs_max_hist':adc_abs_max_hist,
            'adc_abs_max_rf_hist':adc_abs_max_rf_hist,
            'adc_abs_max_rf_w_cut_hist':adc_abs_max_rf_w_cut_hist,
            'mv_bins':mv_bins,
            'mv_bin_center':mv_bin_center,
            'mv_max_hist':mv_max_hist,
            'mv_max_rf_hist':mv_max_rf_hist,
            'mv_max_rf_w_cut_hist':mv_max_rf_w_cut_hist,
            'mv_min_hist':mv_min_hist,
            'mv_min_rf_hist':mv_min_rf_hist,
            'mv_min_rf_w_cut_hist':mv_min_rf_w_cut_hist,
            'mv_abs_max_hist':mv_abs_max_hist,
            'mv_abs_max_rf_hist':mv_abs_max_rf_hist,
            'mv_abs_max_rf_w_cut_hist':mv_abs_max_rf_w_cut_hist}





