import numpy as np
from tqdm import tqdm

def cliff_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting cliff starts!')

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
    
    wo_1min = np.copy(total_qual_cut)
    wo_1min[:, 11] = (evt_num < 7).astype(int)
    wo_1min_sum = np.nansum(wo_1min, axis = 1)
    wo_1min_idx = np.logical_and(wo_1min_sum == 0, trig_type == 0) 
    del wo_1min, wo_1min_sum

    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum, ara_qual

     # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)

    # output arr
    cliff = np.full((num_ants, num_evts), np.nan, dtype = float)
    cliff_bp = np.copy(cliff)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        

        if daq_qual_sum[evt] != 0:
            continue

        # sample index
        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        buffer_info.get_num_samp_in_blk(blk_idx_arr)
        samp_in_blk = buffer_info.samp_in_blk
        if clean_evt_idx[evt]:
            buffer_info.get_num_samp_in_blk(blk_idx_arr, use_int_dat = True)
            int_samp_in_blk = buffer_info.int_samp_in_blk
        del blk_idx_arr

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)

            cliff[ant, evt] = np.nanmedian(raw_v[:samp_in_blk[0,ant]]) - np.nanmedian(raw_v[-samp_in_blk[-1,ant]:])

            if clean_evt_idx[evt]:
                wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)    

            del raw_v, raw_t
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    
        if clean_evt_idx[evt]:
            bp_v = wf_int.pad_v
            bp_num = wf_int.pad_num
            for ant in range(num_ants):
                bp_v_ant = bp_v[:bp_num[ant], ant]
                cliff_bp[ant, evt] = np.nanmedian(bp_v_ant[:int_samp_in_blk[0, ant]]) - np.nanmedian(bp_v_ant[-int_samp_in_blk[-1, ant]:])
                del bp_v_ant
            del int_samp_in_blk, bp_v, bp_num
        del samp_in_blk
    del ara_root, num_evts, daq_qual_sum, num_ants, ara_uproot, buffer_info, wf_int

    adc_bins =  np.linspace(-1200, 1200, 600 + 1)
    ara_hist = hist_loader(adc_bins)
    adc_bin_center = ara_hist.bin_x_center
    cliff_hist = ara_hist.get_1d_hist(cliff)
    cliff_rf_hist = ara_hist.get_1d_hist(cliff, cut = trig_type != 0)
    cliff_rf_wo_1min_cut_hist = ara_hist.get_1d_hist(cliff, cut = ~wo_1min_idx)
    cliff_rf_w_cut_hist  = ara_hist.get_1d_hist(cliff, cut = ~clean_evt_idx)
    cliff_bp_rf_w_cut_hist  = ara_hist.get_1d_hist(cliff_bp, cut = ~clean_evt_idx)
    del ara_hist, clean_evt_idx, wo_1min_idx

    print('peak collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'cliff':cliff,
            'cliff_bp':cliff_bp,
            'adc_bins':adc_bins,
            'adc_bin_center':adc_bin_center,
            'cliff_hist':cliff_hist,
            'cliff_rf_hist':cliff_rf_hist,
            'cliff_rf_wo_1min_cut_hist':cliff_rf_wo_1min_cut_hist,
            'cliff_rf_w_cut_hist':cliff_rf_w_cut_hist,
            'cliff_bp_rf_w_cut_hist':cliff_bp_rf_w_cut_hist}





