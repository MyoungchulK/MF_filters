import numpy as np

def trig_ratio_collector(Data, Ped):

    print('Collecting trig info starts!')

    from tools.ara_data_load import ara_uproot_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts

    #output array
    evt_num = ara_uproot.evt_num
    pps_number = ara_uproot.pps_number
    time_stamp = ara_uproot.time_stamp
    unix_time = ara_uproot.unix_time

    trig_type = ara_uproot.get_trig_type()
    trig_ratio = np.full((3), np.nan, dtype = float)
    trig_ratio[0] = np.count_nonzero(trig_type == 0)
    trig_ratio[1] = np.count_nonzero(trig_type == 1)
    trig_ratio[2] = np.count_nonzero(trig_type == 2)
    trig_ratio /= num_evts

    print()
    print(f'Quick ratio check! RF:{np.round(trig_ratio[0],2)}, Cal:{np.round(trig_ratio[1],2)}, Soft:{np.round(trig_ratio[2],2)}')
    print()

    del ara_uproot, num_evts

    print('Trig info collecting is done!')

    return {'evt_num':evt_num,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'unix_time':unix_time,
            'trig_type':trig_type,
            'trig_ratio':trig_ratio}







