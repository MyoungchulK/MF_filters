import numpy as np
from scipy.stats import rayleigh
from tqdm import tqdm

def rayl_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting rayl. starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = ara_qual.total_qual_cut_sum
    clean_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_entry = entry_num[clean_idx]
    clean_evt = evt_num[clean_idx]
    #clean_entry = ara_qual.get_useful_events(use_entry = True, use_qual = True, trig_idx = 0)
    #clean_evt = ara_qual.get_useful_events(use_qual = True, trig_idx = 0)
    num_clean_evts = len(clean_evt)
    print(f'Number of clean event is {num_clean_evts}') 
    del ara_qual, ara_uproot, entry_num

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_band_pass = True)#, use_cw = True, cw_config = (3, 0.05, 0.13, 0.85))
    fft_len = wf_int.pad_fft_len
    freq_range = wf_int.pad_zero_freq

    # output
    clean_rffts = np.full((fft_len, num_ants, num_clean_evts), np.nan, dtype = float)
    print(f'fft array dim.: {clean_rffts.shape}')
    print(f'fft array size: ~{np.round(clean_rffts.nbytes/1024/1024)} MB')

    # loop over the events
    for evt in tqdm(range(num_clean_evts)):
      #if evt <100:        

        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)#, use_cw = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        clean_rffts[:, :, evt] = wf_int.pad_fft     
    del ara_root, num_clean_evts, wf_int, clean_entry 

    # rayl fit
    binning = 1000
    bin_edges = np.asarray([np.nanmin(clean_rffts, axis = 2), np.nanmax(clean_rffts, axis = 2)])
    rayl_params = np.full((2, fft_len, num_ants), np.nan, dtype = float)    
    for freq in tqdm(range(fft_len)):
        for ant in range(num_ants):
            amp_bins = np.linspace(bin_edges[0, freq, ant], bin_edges[1, freq, ant], binning + 1)
            amp_bins_center = (amp_bins[1:] + amp_bins[:-1]) / 2
            amp_hist = np.histogram(clean_rffts[freq, ant], bins = amp_bins)[0]
            mu_init_idx = np.nanargmax(amp_hist)
            if np.isnan(mu_init_idx):
                continue
            mu_init = amp_bins_center[mu_init_idx]
            del amp_bins, amp_bins_center, amp_hist, mu_init_idx

            try:
                rayl_params[:, freq, ant] = rayleigh.fit(clean_rffts[freq, ant], loc = bin_edges[0, freq, ant], scale = mu_init)
            except RuntimeError:
                print(f'Runtime Issue in {f} GHz!')
                pass
            del mu_init
    del binning, bin_edges
    del num_ants, fft_len

    print('Rayl. collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'total_qual_cut':total_qual_cut,
            'freq_range':freq_range,
            'clean_rffts':clean_rffts,
            'rayl_params':rayl_params}




