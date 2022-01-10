import numpy as np
from tqdm import tqdm

def samp_map_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.year, incl_cable_delay = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut results
    clean_evt = ara_uproot.evt_num
    clean_entry = ara_uproot.entry_num

    from tools.ara_quality_cut import pre_qual_cut_loader
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del pre_qual, pre_qual_cut

    # output array
    adc_medi = np.full((num_Ants, len(clean_entry)), np.nan, dtype = float)
    adc_max_min = np.full((2, num_Ants, len(clean_entry)), np.nan, dtype = float)
    ara_hist = hist_loader()
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    unix_time = ara_uproot.unix_time
    trig_type = ara_uproot.get_trig_type()

    from tools.ara_quality_cut import post_qual_cut_loader
    post_qual = post_qual_cut_loader(ara_uproot,ara_root)
    zero_adc_ratio = post_qual.zero_adc_ratio 

    # loop over the events
    for evt in tqdm(range(len(clean_evt))):
      #if evt < 100:
        
        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyGoodADC)

        # sample index
        blk_idx_arr = ara_uproot.get_block_idx(clean_entry[evt], trim_1st_blk = True)[0]
        samp_idx = buffer_info.get_samp_idx(blk_idx_arr, ch_shape = True)   
        del blk_idx_arr

        # loop over the antennas
        for ant in range(num_Ants):

            # stack in sample map
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            if len(raw_v) == 0:
                continue
            adc_medi[ant, evt] = np.nanmedian(raw_v)
            zero_adc_ratio[ant, evt] = post_qual.get_zero_adc_events(raw_v, len(raw_v))
            if pre_qual_cut_sum[evt] != 0:
                continue
            adc_max_min[0, ant, evt] = np.nanmax(raw_v)
            adc_max_min[1, ant, evt] = np.nanmin(raw_v)
            samp_idx_ant = samp_idx[:,ant][~np.isnan(samp_idx[:,ant])].astype(int)
            if trig_type[evt] == 0:
                ara_hist.stack_in_hist(samp_idx_ant, raw_v.astype(int), ant)
            del samp_idx_ant, raw_v
            ara_root.del_TGraph()
        del samp_idx
        ara_root.del_usefulEvt()
    del ara_const, ara_root, ara_uproot, buffer_info, clean_entry, num_Ants

    samp_medi = ara_hist.get_median_est()
    samp_map = ara_hist.hist_map
    buffer_bit_range = ara_hist.y_range
    buffer_sample_range = ara_hist.x_range
    del ara_hist

    print('WF collecting is done!')

    return {'buffer_bit_range':buffer_bit_range,
            'buffer_sample_range':buffer_sample_range,
            'samp_map':samp_map, 
            'samp_medi':samp_medi,
            'adc_medi':adc_medi,
            'adc_max_min':adc_max_min,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'unix_time':unix_time,
            'trig_type':trig_type,
            'zero_adc_ratio':zero_adc_ratio,
            'clean_evt':clean_evt}








