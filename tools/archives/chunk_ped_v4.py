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
    daq_qual_sum = np.nansum(pre_qual_cut[:, :6], axis = 1)
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
    del ara_root

    # post quality cut
    post_qual_cut = post_qual.get_post_qual_cut()
    del post_qual

    # total quality cut
    total_qual_cut = np.append(pre_qual_cut, post_qual_cut, axis = 1)
    del pre_qual_cut, post_qual_cut

    # clean event type
    num_qual_type = 4
    clean_evts = np.full((num_evts, num_qual_type), 0, dtype = int)
    clean_evts_qual_type = np.full((total_qual_cut.shape[1], num_qual_type), 0, dtype = int)

    qual_type = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,17], dtype = int) # turn on all cuts
    clean_evts_qual_type[qual_type, 0] = 1
    clean_evts[:, 0] = np.logical_and(np.nansum(total_qual_cut, axis = 1) == 0, trig_type != 1).astype(int)
    qual_type = np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,17], dtype = int) # not use bad event and no calpulser ratio cut
    clean_evts_qual_type[qual_type, 1] = 1
    clean_evts[:, 1] = np.logical_and(np.nansum(total_qual_cut[:, qual_type], axis = 1) == 0, trig_type != 1).astype(int)
    qual_type = np.array([0,1,2,3,4,5,6,7,8,11,12,14,17], dtype = int) # hardware only
    clean_evts_qual_type[qual_type, 2] = 1
    clean_evts[:, 2] = np.logical_and(np.nansum(total_qual_cut[:, qual_type], axis = 1) == 0, trig_type != 1).astype(int)
    qual_type = np.array([0,1,2,3,4,5], dtype = int)  # only rf/software
    clean_evts_qual_type[qual_type, 3] = 1
    clean_evts[:, 3] = np.logical_and(np.nansum(total_qual_cut[:, qual_type], axis = 1) == 0, trig_type != 1).astype(int)
    del qual_type
 
    # clean evts for repeder
    clean_num_evts = np.nansum(clean_evts, axis = 0)
    print(f'total uesful events for ped: {clean_num_evts}')

    # ped counter
    block_usage = np.full((num_blks, num_qual_type), 0, dtype = int)
    for evt in tqdm(range(num_evts)):

        if daq_qual_sum[evt] != 0:
            continue

        blk_idx_arr = ara_uproot.get_block_idx(evt, trim_1st_blk = True)[0]
        if clean_evts[evt, 0] == 1:
            block_usage[blk_idx_arr, 0] += 1
        if clean_evts[evt, 1] == 1:
            block_usage[blk_idx_arr, 1] += 1
        if clean_evts[evt, 2] == 1:
            block_usage[blk_idx_arr, 2] += 1
        if clean_evts[evt, 3] == 1:
            block_usage[blk_idx_arr, 3] += 1
        del blk_idx_arr
    del num_blks, num_evts, ara_uproot, daq_qual_sum

    # select final type
    low_block_usage = np.any(block_usage < 2, axis = 0).astype(int)
    print(f'low_block_usage flag: {low_block_usage}')
    final_type = 0
    ped_counts = np.copy(block_usage[:, 3])
    ped_qualities = np.copy(clean_evts[:, 3])
    for t in range(num_qual_type):
        if low_block_usage[t] == 0:
            ped_counts = np.copy(block_usage[:, t])
            ped_qualities = np.copy(clean_evts[:, t])
            break
        final_type += 1
    del num_qual_type
    print(f'type {final_type} was chosen for ped!')
    final_type = np.asarray([final_type]) 

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
            'clean_evts':clean_evts,
            'clean_evts_qual_type':clean_evts_qual_type,
            'clean_num_evts':clean_num_evts,
            'block_usage':block_usage,
            'low_block_usage':low_block_usage,
            'ped_counts':ped_counts,
            'ped_qualities':ped_qualities,
            'final_type':final_type}

