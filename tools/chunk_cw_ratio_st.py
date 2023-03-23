import numpy as np
import h5py
from tqdm import tqdm

def cw_ratio_st_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting cw ratio st starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_quality_cut import get_bad_events
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
    del ara_uproot

    # pre quality cut
    daq_qual_cut = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, use_1st = True)[0]

    # bad antenna
    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_cw = True, st = st, run = run, analyze_blind_dat = analyze_blind_dat)

    # output array  
    cw_ratio = np.full((num_ants, num_evts), np.nan, dtype = float)

    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw_check', verbose = True, force_blind = True) # get the h5 file path
    cw_hf = h5py.File(cw_dat, 'r')
    evt_check = cw_hf['evt_check'][:]    
    del run_info, cw_dat, cw_hf

    if num_evts != len(evt_check):
        print('Wrong!!!!!:', num_evts, len(evt_check), st, run)
        sys.exit(1)
    else:
        print(f'tot_evt: {num_evts}, bad_evt: {np.sum(evt_check)}, bad_ratio{np.round(np.sum(evt_check)/num_evts, 2)}')    

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
    
        if evt_check[evt] == 0:
            continue      
 
        # quality cut
        if daq_qual_cut[evt]:
            continue
 
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_cw = True, use_cw_ratio = True, evt = evt)
            cw_ratio[ant, evt] = wf_int.cw_ratio
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()  
    del ara_root, num_evts, num_ants, wf_int, daq_qual_cut

    print('cw ratio st collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'bad_ant':bad_ant,
            'cw_ratio':cw_ratio} 








