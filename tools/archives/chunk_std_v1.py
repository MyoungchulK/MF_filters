import os, sys
import numpy as np
from tqdm import tqdm

def std_collector_dat(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    entry_num = ara_uproot.entry_num

    #output array
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    unix_time = ara_uproot.unix_time
    wf_std = np.full((num_eles, len(entry_num)), np.nan, dtype = float) 

    # loop over the events
    for evt in tqdm(range(len(entry_num))):
      #if evt <100:        
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        # loop over the antennas
        for ant in range(num_eles):

            # stack in sample map
            raw_v = ara_root.get_ele_ch_wf(ant)[1]
            wf_std[ant, evt] = np.nanstd(raw_v)
            del raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_const, ara_root, ara_uproot, num_eles, entry_num

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'unix_time':unix_time,
            'wf_std':wf_std}







