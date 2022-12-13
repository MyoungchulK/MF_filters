import numpy as np
from tqdm import tqdm
import h5py

def cw_flag_debug_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False):

    print('Collecting cw flag starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_cw_filters_debug import py_phase_variance
    from tools.ara_cw_filters_debug import py_testbed
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
    useful_freq_range_phase = freq_range[cw_phase.useful_freq_idx]
    evt_len = cw_phase.evt_len
    start_evt = int(evt_len - 1)
    cw_testbed = py_testbed(st, run, freq_range, 6, 5.5, 3, analyze_blind_dat = analyze_blind_dat, verbose = True)
    baseline = cw_testbed.baseline_copy
    useful_freq_range = cw_testbed.useful_freq_range
    freq_bins = np.linspace(0, 1, 200 + 1)
    freq_bin_center = (freq_bins[1:] + freq_bins[:-1]) / 2
    dbm_bins = np.linspace(-240, -40, 200 + 1)
    dbm_bin_center = (dbm_bins[1:] + dbm_bins[:-1]) / 2
    del_bins = np.linspace(-50, 50, 200 + 1)
    del_bin_center = (del_bins[1:] + del_bins[:-1]) / 2 
    sig_bins = np.linspace(-2, 18, 200 + 1)
    sig_bin_center = (sig_bins[1:] + sig_bins[:-1]) / 2
    del st, run

    # output array  
    sigma = []
    phase_idx = []
    testbed_idx = []
    empty = np.full((0), np.nan, dtype = float)
    dbm_map = np.full((len(freq_bin_center), len(dbm_bin_center), num_ants, 2), 0, dtype = int)
    del_map = np.full((len(freq_bin_center), len(del_bin_center), num_ants, 2), 0, dtype = int)
    sig_map = np.full((len(freq_bin_center), len(sig_bin_center), 2, 2), 0, dtype = int)

    # loop over the events
    evt_counts = 0
    for evt in tqdm(range(num_evts)):
        
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
        
        if trig_type[evt] == 0:
            arr_idx = 0
        if trig_type[evt] == 2:
            arr_idx = 1
        for ant in range(num_ants): 
            dbm_map[:, :, ant, arr_idx] += np.histogram2d(freq_range, rfft_dbmhz[:, ant], bins = (freq_bins, dbm_bins))[0].astype(int)
        
        cw_testbed.get_bad_magnitude(rfft_dbmhz)
        delta_mag = cw_testbed.delta_mag
        for ant in range(num_ants):
            del_map[:, :, ant, arr_idx] += np.histogram2d(useful_freq_range, delta_mag[:, ant], bins = (freq_bins, del_bins))[0].astype(int)

        testbed_idxs = cw_testbed.bad_idx
        testbed_idx.append(testbed_idxs)
        cw_phase.get_phase_differences(rfft_phase, evt_counts % evt_len)
        del rfft_dbmhz, rfft_phase, delta_mag
        if evt_counts < start_evt:
            sigma.append(empty)
            phase_idx.append(empty)
            evt_counts += 1 
            continue
        cw_phase.get_bad_phase()
        sigma_variance_avg = cw_phase.sigma_variance_avg
        for pol in range(2):
            sig_map[:, :, pol, arr_idx] += np.histogram2d(useful_freq_range_phase, sigma_variance_avg[:, pol], bins = (freq_bins, sig_bins))[0].astype(int)
        del sigma_variance_avg
        sigmas = cw_phase.bad_sigma 
        phase_idxs = cw_phase.bad_idx
        sigma.append(sigmas)
        phase_idx.append(phase_idxs)
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
            'useful_freq_range':useful_freq_range,
            'useful_freq_range_phase':useful_freq_range_phase,
            'sigma':sigma,
            'phase_idx':phase_idx,
            'testbed_idx':testbed_idx,
            'evt_len':np.array([evt_len], dtype = int),
            'baseline':baseline,
            'freq_bins':freq_bins,
            'freq_bin_center':freq_bin_center,
            'dbm_bins':dbm_bins,
            'dbm_bin_center':dbm_bin_center,
            'del_bins':del_bins,
            'del_bin_center':del_bin_center,
            'sig_bins':sig_bins,
            'sig_bin_center':sig_bin_center,
            'dbm_map':dbm_map,
            'del_map':del_map,
            'sig_map':sig_map}




