import numpy as np
from tqdm import tqdm
import h5py

def cw_flag_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False):

    print('Collecting cw flag starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_cw_filters import py_phase_variance
    from tools.ara_cw_filters import py_testbed
    from tools.ara_run_manager import run_info_loader
    from tools.ara_known_issue import known_issue_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    trig_type = ara_uproot.get_trig_type()
    evt_num = ara_uproot.evt_num
    num_evts = ara_uproot.num_evts
    st = ara_uproot.station_id
    yr = ara_uproot.year
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, yr)
    del ara_uproot

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True, force_blind = True)
    daq_hf = h5py.File(daq_dat, 'r')
    daq_evt = daq_hf['evt_num'][:]
    daq_qual_cut = daq_hf['daq_qual_cut_sum'][:] != 0
    daq_qual_cut_sum = np.in1d(evt_num, daq_evt[daq_qual_cut]).astype(int)
    del run_info, daq_dat, daq_hf, daq_evt, daq_qual_cut

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)
    freq_range = wf_int.pad_zero_freq

    # cw class
    cw_phase = py_phase_variance(st, run, freq_range)
    evt_len = cw_phase.evt_len
    start_evt = int(evt_len - 1)
    cw_testbed = py_testbed(st, run, freq_range, 6, 5.5, 3, analyze_blind_dat = analyze_blind_dat, verbose = True)
    del st, run

    # output array  
    sigma = []
    phase_idx = []
    testbed_idx = []
    empty = np.full((0), np.nan, dtype = float)

    # loop over the events
    evt_counts = 0
    for evt in tqdm(range(num_evts)):
      if evt_num[evt] == 84329:        
        
        if daq_qual_cut_sum[evt] or trig_type[evt] == 1:
            sigma.append(empty)
            phase_idx.append(empty)
            testbed_idx.append(empty)
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_phase = True, use_abs = True, use_norm = True, use_dbmHz = True)
        rfft_dbmhz = wf_int.pad_fft
        rfft_phase = wf_int.pad_phase
        
        cw_testbed.get_bad_magnitude(rfft_dbmhz)
        testbed_idxs = cw_testbed.bad_idx
        testbed_idx.append(testbed_idxs)
        print('testbed: ', freq_range[testbed_idxs])       
        print(len(testbed_idxs))
        print(len(freq_range))
 
        cw_phase.get_phase_differences(rfft_phase, evt_counts % evt_len)
        del rfft_dbmhz, rfft_phase
        if evt_counts < start_evt:
            sigma.append(empty)
            phase_idx.append(empty)
            evt_counts += 1 
            continue
        cw_phase.get_bad_phase()
        sigmas = cw_phase.bad_sigma 
        phase_idxs = cw_phase.bad_idx
        sigma.append(sigmas)
        phase_idx.append(phase_idxs)
        print('phase: ', freq_range[phase_idxs])
        print(evt_counts)        
        print(evt_num[evt])
   
        if evt_counts == 14: 
            sys.exit(1)
        evt_counts += 1
    del ara_root, num_evts, num_ants, wf_int, cw_phase, cw_testbed, daq_qual_cut_sum

    # to numpy array
    sigma = np.asarray(sigma)
    phase_idx = np.asarray(phase_idx)
    testbed_idx = np.asarray(testbed_idx)

    print('CW flag collecting is done!')

    return {'evt_num':evt_num,
            'bad_ant':bad_ant,
            'freq_range':freq_range,
            'sigma':sigma,
            'phase_idx':phase_idx,
            'testbed_idx':testbed_idx,
            'evt_len':np.array([evt_len], dtype = int)}









