import os, sys
import numpy as np
from tqdm import tqdm

def samp_map_collector_dat(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import clean_event_loader
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    num_Strs = ara_const.DDA_PER_ATRI

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.year, incl_cable_delay = True)
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut results
    #cleaner = clean_event_loader(ara_uproot, trig_flag = [0,2], qual_flag = [0])    
    #clean_evt, clean_entry, clean_st, clean_ant = cleaner.get_qual_cut_results()
    #del cleaner

    clean_evt = ara_uproot.evt_num
    clean_entry = ara_uproot.entry_num
    clean_st = np.full((4,len(clean_evt)), 0, dtype = int)
    clean_ant = np.arange(16, dtype = int)

    # output array
    ara_hist = hist_loader()

    samp_peak = np.full((num_Ants, len(clean_evt)), np.nan, dtype = float)
    adc_medi = np.copy(samp_peak)
    from tools.ara_quality_cut import post_qual_cut_loader
    post_qual = post_qual_cut_loader(ara_uproot,ara_root)
    spikey_evts = post_qual.spikey_evts

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

            if clean_st[ant%num_Strs, evt] != 0:
                print('not good string!', ant%num_Strs)
                continue       
 
            # stack in sample map
            samp_idx_ant = samp_idx[:,ant][~np.isnan(samp_idx[:,ant])].astype(int)
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            adc_medi[ant, evt] = np.nanmedian(raw_v)
            ara_hist.stack_in_hist(samp_idx_ant, raw_v.astype(int), ant)
            del samp_idx_ant, raw_v
            ara_root.del_TGraph()
        del samp_idx
        ara_root.del_usefulEvt()
            
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        for ant in range(num_Ants):
            raw_v = ara_root.get_rf_ch_wf(ant)[1]

            if len(raw_v) == 0:
                continue

            samp_peak[ant, evt] = np.nanmax(np.abs(raw_v))
            spikey_evts[ant, evt] = np.nanmax(np.abs(raw_v))
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
        
    del ara_const, ara_root, ara_uproot, buffer_info, clean_ant, clean_entry, num_Strs, num_Ants, clean_st

    spikey_ratio = post_qual.get_spikey_ratio(apply_bad_ant = True)
    print(spikey_ratio)
    del post_qual, spikey_evts

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
            'samp_peak':samp_peak,
            'spikey_ratio':spikey_ratio,
            'adc_medi':adc_medi,
            'clean_evt':clean_evt}








