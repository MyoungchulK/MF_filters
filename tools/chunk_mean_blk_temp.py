import numpy as np
from tqdm import tqdm

def mean_blk_temp_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting mean block starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import hist_loader
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_blks = ara_const.BLOCKS_PER_DDA
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.run, ara_uproot.year, incl_cable_delay = True)
    buffer_info.get_int_time_info()
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
    del qual_cut_sum, ara_qual

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)

    # output array
    blk_est_range = 50
    blk_idx = np.full((blk_est_range, num_evts), np.nan, dtype = float)
    mean_blk = np.full((blk_est_range, num_ants, num_evts), np.nan, dtype = float)
    mean_blk_bp = np.copy(mean_blk)
    del blk_est_range

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        

        if daq_qual_sum[evt] != 0:
            continue
    
        # block index
        blk_idx_arr, blk_len = ara_uproot.get_block_idx(evt, trim_1st_blk = True)
        blk_idx[:blk_len, evt] = blk_idx_arr    
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        if clean_evt_idx[evt]:
            buffer_info.get_num_samp_in_blk(blk_idx_arr, use_int_dat = True)
        del blk_idx_arr

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            mean_blk[:blk_len, ant, evt] = buffer_info.get_mean_blk(ant, raw_v)

            if clean_evt_idx[evt]:
                wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
   
        if clean_evt_idx[evt]:
            bp_v = wf_int.pad_v
            bp_num = wf_int.pad_num
            for ant in range(num_ants):
                mean_blk_bp[:blk_len, ant, evt] = buffer_info.get_mean_blk(ant, bp_v[:bp_num[ant], ant], use_int_dat = True)
            del bp_v, bp_num
        del blk_len
    del ara_root, num_evts, daq_qual_sum, num_ants, ara_uproot, buffer_info, wf_int

    mean_range = np.arange(-30, 30, 1)
    mean_bins = np.linspace(-30, 30, 60 + 1)
    ara_hist = hist_loader(mean_bins)
    mean_bin_center = ara_hist.bin_x_center
    mean_blk_hist = ara_hist.get_flat_1d_hist(mean_blk)
    mean_blk_rf_hist = ara_hist.get_flat_1d_hist(mean_blk, cut = trig_type != 0)
    mean_blk_rf_w_cut_hist  = ara_hist.get_flat_1d_hist(mean_blk, cut = ~clean_evt_idx)
    mean_blk_bp_rf_w_cut_hist  = ara_hist.get_flat_1d_hist(mean_blk_bp, cut = ~clean_evt_idx)
    del ara_hist

    blk_range = np.arange(num_blks)
    blk_bins = np.linspace(0, num_blks, num_blks + 1) 
    ara_hist = hist_loader(blk_bins, mean_bins)
    blk_bin_center = ara_hist.bin_x_center
    mean_blk_2d_hist = ara_hist.get_mean_blk_2d_hist(blk_idx, mean_blk)
    mean_blk_2d_rf_hist = ara_hist.get_mean_blk_2d_hist(blk_idx, mean_blk, cut = trig_type != 0)
    mean_blk_2d_rf_w_cut_hist  = ara_hist.get_mean_blk_2d_hist(blk_idx, mean_blk, cut = ~clean_evt_idx)
    mean_blk_bp_2d_rf_w_cut_hist  = ara_hist.get_mean_blk_2d_hist(blk_idx, mean_blk_bp, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx, num_blks

    print('Mean blk collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'mean_blk':mean_blk,
            'mean_blk_bp':mean_blk_bp,
            'blk_idx':blk_idx,
            'mean_range':mean_range,
            'mean_bins':mean_bins,
            'mean_bin_center':mean_bin_center,
            'mean_blk_hist':mean_blk_hist,
            'mean_blk_rf_hist':mean_blk_rf_hist,
            'mean_blk_rf_w_cut_hist':mean_blk_rf_w_cut_hist,
            'mean_blk_bp_rf_w_cut_hist':mean_blk_bp_rf_w_cut_hist,
            'blk_range':blk_range,
            'blk_bins':blk_bins,
            'blk_bin_center':blk_bin_center,
            'mean_blk_2d_hist':mean_blk_2d_hist,
            'mean_blk_2d_rf_hist':mean_blk_2d_rf_hist,
            'mean_blk_2d_rf_w_cut_hist':mean_blk_2d_rf_w_cut_hist,
            'mean_blk_bp_2d_rf_w_cut_hist':mean_blk_bp_2d_rf_w_cut_hist}
