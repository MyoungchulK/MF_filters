import os
import numpy as np
from tqdm import tqdm
import h5py
from scipy.signal import medfilt

def medi_filt_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False):

    print('Collecting medi filt starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import run_info_loader
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
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    qual_hf = h5py.File(qual_dat, 'r')
    qual_evt = qual_hf['evt_num'][:]
    qual_daq = qual_hf['daq_qual_cut_sum'][:] != 0
    qual_tot = qual_hf['tot_qual_cut_sum'][:] != 0
    daq_cut = np.in1d(evt_num, qual_evt[qual_daq]).astype(int)
    tot_cut = np.in1d(evt_num, qual_evt[qual_tot]).astype(int)
    del ara_uproot, run_info, qual_dat, qual_hf, qual_evt, qual_daq, qual_tot

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_band_pass = True)
    freq_width = 0.04 # 40 MHz

    # output
    power = np.full((num_ants, num_evts), np.nan, dtype = float)
    power_medi = np.copy(power)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:
 
        if daq_cut[evt]:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
            
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        wfs = wf_int.pad_v
        power[:, evt] = np.nanmean(wfs ** 2, axis = 0)

        wf_int.get_fft_wf(use_abs = True, use_norm = True)
        ffts = wf_int.pad_fft / wf_int.sqrt_dt
        freqs = wf_int.pad_freq
        filt_size = (freq_width / (freqs[1] - freqs[0])).astype(int)
        for ant in range(num_ants):
            if filt_size[ant] % 2 == 0:
                filt_size[ant] += 1
            smooith_ffts = medfilt(ffts[:, ant], kernel_size = filt_size[ant])
            power_medi[ant, evt] = np.nanmean(smooith_ffts ** 2)
            del smooith_ffts
        del wfs, ffts, freqs, filt_size
    del ara_root, num_evts, num_ants, wf_int, daq_cut, freq_width 
  
    print('Medi filt collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'tot_cut':tot_cut,
            'power':power,          
            'power_medi':power_medi}        





      

