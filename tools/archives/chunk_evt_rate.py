import numpy as np
from tqdm import tqdm

def evt_rate_collector(Data, Ped):

    print('Collecting event rate starts!')

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

    # event rate
    one_min = 60
    unix_norm = unix_time - unix_time[0]
    unix_norm_min = unix_norm//one_min
    min_arr = np.arange(unix_norm_min[-1]+1)
 
    num_evts_in_min = np.full((len(min_arr)), 0, dtype = float) 
    evt_rate = np.full((len(min_arr), 3), 0, dtype = float) 
    for t in tqdm(range(num_evts)):
        evt_rate[unix_norm_min[t], trig_type[t]] += 1
        num_evts_in_min[unix_norm_min[t]] += 1        
    print(num_evts_in_min)
    evt_rate /= num_evts_in_min[:, np.newaxis]
    evt_rate *= 10
    print(evt_rate[:,0])
    print(evt_rate[:,1])
    print(evt_rate[:,2])

    # event histogram
    rate_bins = np.linspace(0, 20, 2000 + 1)
    rate_bin_center = (rate_bins[1:] + rate_bins[:-1]) * 0.5
    rate_hist = np.full((len(rate_bin_center), 3), 0, dtype = int)

    del ara_uproot, num_evts

    print('Event rate collecting is done!')

    return {'evt_num':evt_num,
            'pps_number':pps_number,
            'time_stamp':time_stamp,
            'unix_time':unix_time,
            'trig_type':trig_type,
            'rate_bins':rate_bins,
            'rate_bin_center':rate_bin_center,
            'rate_hist':rate_hist,
            'evt_rate':evt_rate}







