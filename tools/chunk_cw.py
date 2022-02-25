import numpy as np
from tqdm import tqdm

def cw_collector(Data, Ped, analyze_blind_dat = False, sel_evts = None):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
  
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
 
    print(f'Example event number: {evt_num[:20]}')
    print(f'Example trigger number: {trig_type[:20]}')
    if sel_evts is not None:
        sel_evt_idx = np.in1d(evt_num, sel_evts)
        sel_entries = entry_num[sel_evt_idx]
        sel_evts = evt_num[sel_evt_idx]
    else:
        sel_entries = entry_num[:20]
        sel_evts = evt_num[sel_entries]
    print(f'Selected events are {sel_evts}')
    print(f'Selected entries are {sel_entries}')
    sel_evt_len = len(sel_entries)

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)
    dt = wf_int.dt

    # output array
    wf_all = np.full((wf_int.pad_len, 2, num_ants, sel_evt_len), np.nan, dtype=float)
    int_wf_all = np.copy(wf_all)
    bp_wf_all = np.copy(wf_all)
    freq = np.full((wf_int.pad_len, num_ants, sel_evt_len), np.nan, dtype=float)
    int_fft = np.full(freq.shape, np.nan, dtype=complex)
    bp_fft = np.copy(int_fft)
    cw_out_all = np.copy(wf_all)

    import os
    import ROOT
    #ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")
    #ROOT.gInterpreter.ProcessLine('#include "/cvmfs/ara.opensciencegrid.org/trunk/centos7/misc_build/include/FFTtools.h"')
    ROOT.gSystem.Load('/cvmfs/ara.opensciencegrid.org/trunk/centos7/misc_build/lib/libRootFftwWrapper.so')
   
    sinsub = ROOT.FFTtools.SineSubtract(3, 0.05, False)  # optional arguments in constructing
    sinsub.setVerbose(False)
    #sinsub.setFreqLimits(0.13, 0.85)
    sinsub.setFreqLimits(0.2, 0.3)


    # loop over the events
    for evt in tqdm(range(sel_evt_len)):
       
        # get entry and wf
        ara_root.get_entry(sel_entries[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
 
        # loop over the antennas
        for ant in range(num_ants):        

            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_len = len(raw_t)
            wf_all[:wf_len, 0, ant, evt] = raw_t
            wf_all[:wf_len, 1, ant, evt] = raw_v
            
            int_t, int_v = wf_int.get_int_wf(raw_t, raw_v, ant)
            int_wf_len = len(int_t)
             
            cw_out = np.full((int_wf_len), 0, dtype = np.double)
            sinsub.subtractCW(int_wf_len, int_v, dt, cw_out)

            #cw_freq = sinsub.storedSpectra(0).GetX()
            #cw_fft = sinsub.storedSpectra(0).GetY()
            num_cw_sols = sinsub.getNSines()
            print(num_cw_sols)
            cw_sol_freq_num = np.frombuffer(sinsub.getFreqs(), dtype = float, count=num_cw_sols)
            print(cw_sol_freq_num)
            if num_cw_sols > 0:
                cw_out_all[:int_wf_len, 0, ant, evt] = int_t 
                cw_out_all[:int_wf_len, 1, ant, evt] = cw_out.astype(float) 
                    
    

            int_wf_all[:int_wf_len, 0, ant, evt] = int_t
            int_wf_all[:int_wf_len, 1, ant, evt] = int_v
           
            bp_v = wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True)[1]
            bp_wf_all[:int_wf_len, 0, ant, evt] = int_t
            bp_wf_all[:int_wf_len, 1, ant, evt] = bp_v     

            int_freq = np.fft.rfftfreq(int_wf_len, dt)
            fft_len = len(int_freq)
            int_fft_evt = np.fft.rfft(int_v)
            bp_fft_evt = np.fft.rfft(bp_v)
            freq[:fft_len, ant, evt] = int_freq
            int_fft[:fft_len, ant, evt] = int_fft_evt
            bp_fft[:fft_len, ant, evt] = bp_fft_evt
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

    print('WF collecting is done!')

    #output
    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'sel_entries':sel_entries,
            'sel_evts':sel_evts,
            'wf_all':wf_all,
            'int_wf_all':int_wf_all,
            'bp_wf_all':bp_wf_all,
            'cw_out_all':cw_out_all,
            'freq':freq,
            'int_fft':int_fft,
            'bp_fft':bp_fft}























