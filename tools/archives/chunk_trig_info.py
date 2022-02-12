import os, sys
import numpy as np
from tqdm import tqdm

def trig_info_collector_dat(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()

    #output array
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    unix_time = ara_uproot.unix_time

    del ara_uproot

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'unix_time':unix_time}







