import os
import numpy as np
from tqdm import tqdm

def ped_collector(Data, Ped, analyze_blind_dat = False):

    print('Ped starts!')

    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.ara_quality_cut import post_qual_cut_loader
    from tools.ara_utility import size_checker
    from tools.ara_constant import ara_const
    
    # geom. info.
    ara_const = ara_const()
    num_blks = ara_const.BLOCKS_PER_DDA
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    unix_time = ara_uproot.unix_time
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)

    # quality cut config
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut(use_for_ped_qual = True)
    daq_qual_sum = np.nansum(pre_qual_cut[:, :9], axis = 1)
    post_qual = post_qual_cut_loader(ara_uproot, ara_root, verbose = True)
    del pre_qual

    # loop over the events
    if st == 3 and (run > 1124 and run < 1429):
        for evt in tqdm(range(num_evts)):
          #if evt<100:

            if daq_qual_sum[evt] != 0:
                continue

            # post quality cut
            post_qual.run_post_qual_cut(evt)
    del ara_root, daq_qual_sum

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    del post_qual

    # total quality cut
    total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    total_qual_cut_2nd = np.copy(total_qual_cut)
    total_qual_cut_2nd[:, 9] = 0 # use bad events number flag
    total_qual_cut_2nd[:, 10] = 0 # use bad unix time flag
    total_qual_cut_2nd[:, 13] = 0 # use cal ratio flag
    del pre_qual_cut, post_qual_cut
 
    # clean evts for repeder
    clean_evt_idx = np.logical_and(np.nansum(total_qual_cut, axis = 1) == 0, trig_type != 1)
    clean_evt_idx_2nd = np.logical_and(np.nansum(total_qual_cut_2nd, axis = 1) == 0, trig_type != 1)
    clean_num_evts = np.full((2), 0, dtype = int)
    clean_num_evts[0] = np.count_nonzero(clean_evt_idx)
    clean_num_evts[1] = np.count_nonzero(clean_evt_idx_2nd)
    ped_qualities = np.copy(clean_evt_idx)
    if clean_num_evts[0] == 0:
        ped_qualities = np.copy(clean_evt_idx_2nd)
    del total_qual_cut_2nd, clean_evt_idx, clean_evt_idx_2nd
    print(f'total uesful events for ped: {clean_num_evts}')

    # ped counter
    ped_counts = np.full((num_blks), 0, dtype = int)
    for evt in tqdm(range(num_evts)):

        if ped_qualities[evt] == False:
            continue

        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        ped_counts[blk_idx_arr] += 1
        del blk_idx_arr
    del num_blks, num_evts, ara_uproot

    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{st}/ped{blind_type}/'
    if not os.path.exists(Output):
        os.makedirs(Output)

    txt_file_name = f'{Output}ped{blind_type}_qualities_A{st}_R{run}.dat'
    np.savetxt(txt_file_name, ped_qualities.astype(int), fmt='%i')
    print(f'output is {txt_file_name}')
    size_checker(txt_file_name)
    del st, run

    print('Ped is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'clean_num_evts':clean_num_evts,
            'ped_counts':ped_counts,
            'ped_qualities':ped_qualities}

