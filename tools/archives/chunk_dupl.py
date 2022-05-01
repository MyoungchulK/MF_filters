import numpy as np
from tqdm import tqdm

def dupl_collector(Data, Ped):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_bits = ara_const.BUFFER_BIT_RANGE
    num_samps = ara_const.SAMPLES_PER_BLOCK
    qua_num_samps = num_samps // 4
    del num_samps

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts

    #output array
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    dupl_hist = np.full((num_ants, num_bits*2), 0, dtype = int)
    dupl_rf_hist = np.copy(dupl_hist)
    dupl_rf_hist_w_cut = np.copy(dupl_hist)
    dupl_bins = np.linspace(-num_bits, num_bits, num_bits*2 + 1)
    dupl_range = np.arange(-num_bits, num_bits)
    del num_bits

    from tools.ara_quality_cut import pre_qual_cut_loader
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    del pre_qual, pre_qual_cut_temp, ara_uproot

    clean_rf_evt_idx = np.logical_and(pre_qual_cut_sum == 0, trig_type == 0)
    clean_rf_evt = evt_num[clean_rf_evt_idx]   
    print(f'Number of clean event is {len(clean_rf_evt)}') 
    del clean_rf_evt_idx

    def quater_samp_split(dat, modulo = qua_num_samps):
        dat_reshape = dat.reshape(-1,modulo)
        even_dat = dat_reshape[::2].reshape(-1)
        odd_dat = dat_reshape[1::2].reshape(-1)
        del dat_reshape
        return even_dat, odd_dat

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
    
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlock)

        # loop over the antennas
        for ant in range(num_ants):

            # stack in sample map
            raw_v = ara_root.get_rf_ch_wf(ant)[1]
            if len(raw_v) == 0:
                continue

            even_raw_v, odd_raw_v = quater_samp_split(raw_v)
            diff = even_raw_v - odd_raw_v
            del even_raw_v, odd_raw_v

            dupl_hist_wf = np.histogram(diff, bins = dupl_bins)[0].astype(int)
            dupl_hist[ant] += dupl_hist_wf
            if trig_type[evt] == 0:
                dupl_rf_hist[ant] += dupl_hist_wf
            if pre_qual_cut_sum[evt] == 0 and trig_type[evt] == 0:
                dupl_rf_hist_w_cut[ant] += dupl_hist_wf
            del raw_v, diff, dupl_hist_wf
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()
    del ara_const, ara_root, num_evts, pre_qual_cut_sum, num_ants, qua_num_samps

    print('WF collecting is done!')

    return {'evt_num':evt_num,
            'clean_rf_evt':clean_rf_evt,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'dupl_bins':dupl_bins,
            'dupl_range':dupl_range,
            'dupl_hist':dupl_hist,
            'dupl_rf_hist':dupl_rf_hist,
            'dupl_rf_hist_w_cut':dupl_rf_hist_w_cut}







