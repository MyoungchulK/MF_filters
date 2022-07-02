import numpy as np
from scipy.stats import rayleigh
from tqdm import tqdm

def rayl_lite_collector(Data, Ped, Station, Run, Year, use_araroot_cut = False, use_mf_qual_cut = False, analyze_blind_dat = False):

    print('Collecting rayl. starts!')

    from tools.ara_data_load_lite import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer_lite import wf_analyzer

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Ped, Station, Year)
    num_evts = ara_root.num_evts

    # quality cuts
    if use_araroot_cut:
        print('AraRoot quality cut class!')
        ara_root.get_qual_cut()
    if use_mf_qual_cut:
        import h5py
        from tools.ara_run_manager_lite import run_info_loader
        run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
        qual_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
        qual_hf = h5py.File(qual_dat, 'r')
        bad_evt = qual_hf['total_qual_cut_sum'][:]
        del run_info, qual_dat, qual_hf
        print(f'Number of clean event: {np.count_nonzero(bad_evt == 0)}')

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)
    fft_len = wf_int.pad_fft_len
    freq_range = wf_int.pad_zero_freq

    # output
    rffts = []
    used_entries = []

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        

        # mf quality cut
        if use_mf_qual_cut and bad_evt[evt] != 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)

        # trigger filtering
        if ara_root.get_trig_type() != 0: # only rf. trust issue with software...
            continue
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # araroot quality cuts
        if use_araroot_cut and ara_root.get_qual_cut_result() != 0:
            continue

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            if len(raw_v) == 0:
                ara_root.del_TGraph()
                continue

            # interpolation and zero padding
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v 
            ara_root.del_TGraph()
        ara_root.del_usefulEvt(use_araroot_cut)

        # rfft
        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True)
        rfft_evt = wf_int.pad_fft
        rffts.append(rfft_evt)  
        used_entries.append(evt)
    del ara_root, num_evts, wf_int

    # size check
    rffts = np.asarray(rffts)
    print(f'fft array dim.: {rffts.shape}')
    print(f'fft array size: ~{np.round(rffts.nbytes/1024/1024)} MB')
    used_entries = np.asarray(used_entries)

    # rayl fit
    binning = 1000
    bin_edges = np.asarray([np.nanmin(rffts, axis = 0), np.nanmax(rffts, axis = 0)])
    rayl_params = np.full((2, fft_len, num_ants), np.nan, dtype = float)    
    for freq in tqdm(range(fft_len)):
        for ant in range(num_ants):
            amp_bins = np.linspace(bin_edges[0, freq, ant], bin_edges[1, freq, ant], binning + 1)
            amp_bins_center = (amp_bins[1:] + amp_bins[:-1]) / 2
            amp_hist = np.histogram(rffts[:, freq, ant], bins = amp_bins)[0]
            mu_init_idx = np.nanargmax(amp_hist)
            if np.isnan(mu_init_idx):
                continue
            mu_init = amp_bins_center[mu_init_idx]
            del amp_bins, amp_bins_center, amp_hist, mu_init_idx

            try:
                # unbinned fitting
                rayl_params[:, freq, ant] = rayleigh.fit(rffts[:, freq, ant], loc = bin_edges[0, freq, ant], scale = mu_init)
            except RuntimeError:
                print(f'Runtime Issue in {f} GHz!')
                pass
            del mu_init
    del binning, bin_edges
    del num_ants, fft_len, rffts

    print('Rayl. collecting is done!')

    return {'used_entries':used_entries,
            'freq_range':freq_range,
            'rayl_params':rayl_params}




