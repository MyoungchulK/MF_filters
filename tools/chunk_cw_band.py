import numpy as np
import h5py
from tqdm import tqdm

def cw_band_collector(Data, Station, Run, analyze_blind_dat = False, no_tqdm = False):

    print('Collecting cw band starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_cw_filters import group_bad_frequency   
    from tools.ara_quality_cut import get_bad_events 

    # load cw flag
    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw_flag', verbose = True, force_blind = True) # get the h5 file path
    cw_hf = h5py.File(cw_dat, 'r')
    cw_sigma = cw_hf['sigma'][:] # sigma value for phase variance
    cw_phase = cw_hf['phase_idx'][:] # bad frequency indexs by phase variance
    cw_testbed = cw_hf['testbed_idx'][:] # bad freqency indexs by testbed method
    freq_range = cw_hf['freq_range'][:] # frequency array that uesd for identification
    evt_num = cw_hf['evt_num'][:]
    num_evts = len(evt_num)
    del cw_dat, cw_hf

    if num_evts != len(cw_sigma):
        print('Wrong!!!!!:', num_evts - len(cw_sigma), num_evts, len(cw_sigma), Station, Run)

    if analyze_blind_dat == False:
        evt_num_full = np.copy(evt_num)
        r_dat = run_info.get_result_path(file_type = 'reco', verbose = True) # get the h5 file path
        r_hf = h5py.File(r_dat, 'r')
        evt_num = r_hf['evt_num'][:]
        num_evts = len(evt_num)
        del r_dat, r_hf
        #from tools.ara_data_load import ara_uproot_loader
        #ara_uproot = ara_uproot_loader(Data)
        #num_evts = ara_uproot.num_evts
        #evt_num = ara_uproot.evt_num
        #del ara_uproot
    del run_info

    # pre quality cut
    daq_qual_cut = get_bad_events(Station, Run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, use_1st = True)[0]

    # group bad frequency
    cw_freq = group_bad_frequency(Station, Run, freq_range, verbose = True) # constructor for bad frequency grouping function
    del freq_range

    # output array  
    bad_range = []
    empty_float = np.full((0), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
     
        # quality cut
        if daq_qual_cut[evt]:
            bad_range.append(empty_float)
            continue

        if analyze_blind_dat == False:
            entry_idx = np.where(evt_num_full == evt_num[evt])[0]
            if len(entry_idx) == 0:
                bad_range.append(empty_float)
                continue
            else:
                entry_idx = entry_idx[0]
        else:
            entry_idx = evt
 
        bad_range_evt = cw_freq.get_pick_freqs_n_bands(cw_sigma[entry_idx], cw_phase[entry_idx], cw_testbed[entry_idx]).flatten()
        bad_range.append(bad_range_evt)
    del cw_sigma, cw_phase, cw_testbed, num_evts, daq_qual_cut, cw_freq

    # to numpy array
    bad_range = np.asarray(bad_range)

    print('cw band collecting is done!')

    return {'evt_num':evt_num,
            'bad_range':bad_range}







