import numpy as np
from tqdm import tqdm
import h5py

def spec_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting spec starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True, use_band_pass = False)
    freq = wf_int.pad_zero_freq
    del ara_uproot

    # output array  
    freq_bins = np.linspace(0, 1, 500 + 1)
    freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2 
    log10_amp_bins = np.linspace(-5, 5, 500 + 1)
    log10_amp_bin_center = (log10_amp_bins[1:] + log10_amp_bins[:-1]) / 2 
    spec = np.full((len(freq_bin_center), len(log10_amp_bin_center), num_ants, 3), 0, dtype = int)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
  
        if trig_type[evt] != 1:
            continue
 
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            if len(raw_t) == 0:
                del raw_t, raw_v
                ara_root.del_TGraph()
                continue
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = False)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        # fft
        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True)       
        fft_evt = np.log10(wf_int.pad_fft)
        for ant in range(num_ants):
            spec[:, :, ant, trig_type[evt]] += np.histogram2d(freq, fft_evt[:, ant], bins = (freq_bins, log10_amp_bins))[0].astype(int)
        del fft_evt
    del ara_root, num_evts, num_ants, wf_int

    print('Reco collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'log10_amp_bins':log10_amp_bins,
            'log10_amp_bin_center':log10_amp_bin_center,
            'freq':freq,
            'spec':spec}
    








