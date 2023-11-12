import numpy as np
from tqdm import tqdm
import h5py

def l2_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting reco starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    evt_num = ara_uproot.evt_num
    num_evts = ara_uproot.num_evts
    trig_type = ara_uproot.get_trig_type()
    st = ara_uproot.station_id
    yr = ara_uproot.year
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, yr)
    ara_uproot.get_sub_info()
    read_win = np.nansum(ara_uproot.read_win // 4)
    read_win *= 64
    read_win *= num_ants
    read_10 = int(float(read_win) * 0.1)
    read_win += read_10 
    del ara_uproot, st, run, yr, read_10

    wf_len = np.full((num_evts * num_ants), 0, dtype = int)
    wf_v = np.full((read_win), 0, dtype = int)
    print(f'filtered wfs size: ~{np.round(wf_v.nbytes/1024/1024)} MB')
    del read_win

    counts = 0
    bin_conunts = 0
    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
        
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kJustPedWithOut1stBlockAndBadSamples)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            pad_num = len(raw_v)
            wf_len[counts] = pad_num
            wf_v[bin_conunts:bin_conunts + pad_num] = raw_v
            bin_conunts += pad_num
            counts += 1
            del raw_t, raw_v, pad_num
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del num_ants, num_evts, ara_root, counts 

    wf_v = wf_v[:bin_conunts]
    del bin_conunts

    print('l2 collecting is done!')

    return {'evt_num':evt_num, 
            'trig_type':trig_type,
            'wf_len':wf_len,
            'wf_v':wf_v}








