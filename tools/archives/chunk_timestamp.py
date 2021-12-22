import os, sys
import numpy as np
from tqdm import tqdm

def time_stamp_collector_dat(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.constant import ara_const
    from tools.qual import qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_Ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Ped, Station)
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    time_stamp = ara_uproot.time_stamp
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts

    # pre quality cut
    ara_qual = qual_cut_loader(Station, ara_uproot, trim_1st_blk = True)
    pre_qual_cut = ara_qual.run_pre_qual_cut() 
    del ara_qual

    # output array
    adc_mean = np.full((num_evts, num_Ants), np.nan, dtype = float)
    adc_median = np.copy(adc_mean)
    low_adc_ratio = np.copy(adc_mean)
    low_adc_limit = 8

    # loop over the events
    for evt in tqdm(range(num_evts)):
        
        # get entry and wf
        ara_root.get_entry(entry_num[evt])
        ara_root.get_useful_evt()

        if pre_qual_cut[evt,2] == 1 or pre_qual_cut[evt,4] == 1 or pre_qual_cut[evt,0] == 1:
            continue

        # loop over the antennas
        for ant in range(num_Ants):
        
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            wf_len = len(raw_v)
            
            adc_mean[evt, ant] = np.nanmean(raw_v)
            adc_median[evt, ant] = np.nanmedian(raw_v)
            low_adc_ratio[evt, ant] = np.count_nonzero(raw_v < low_adc_limit)/wf_len
            
            del raw_v, wf_len
            ara_root.del_TGraph()

        ara_root.del_usefulEvt()
    del ara_root, ara_uproot, num_Ants, entry_num, low_adc_limit, num_evts
  
    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'time_stamp':time_stamp,
            'pps_number':pps_number,
            'unix_time':unix_time,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'adc_mean':adc_mean,
            'adc_median':adc_median,
            'low_adc_ratio':low_adc_ratio}







