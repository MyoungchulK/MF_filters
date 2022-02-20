import os
import numpy as np
from tqdm import tqdm

def ped_collector(Data, Ped, analyze_blind_dat = False):

    print('Ped starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import pre_qual_cut_loader
    from tools.utility import size_checker
    from tools.ara_constant import ara_const
    
    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    num_blks = ara_const.BLOCKS_PER_DDA
    num_chs = ara_const.RFCHAN_PER_DDA
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    st = ara_uproot.station_id
    run = ara_uproot.run
   
    # quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, analyze_blind_dat = analyze_blind_dat, verbose = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut(use_for_ped_qual = True)
    pre_qual_cut_temp1 = np.full(pre_qual_cut.shape, -1, dtype = int)
    pre_qual_cut_temp2 = np.full(pre_qual_cut.shape, -1, dtype = int)
    pre_qual_cut_temp3 = np.full(pre_qual_cut.shape, -1, dtype = int)
    pre_qual_cut_temp4 = np.full(pre_qual_cut.shape, -1, dtype = int)
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del pre_qual

    # clean evts for repeder
    clean_num_evts = np.full((5), -1, dtype = int)
    ped_qualities = np.logical_and(pre_qual_cut_sum == 0, trig_type != 1)
    clean_evt = np.count_nonzero(ped_qualities)
    clean_num_evts[0] = np.copy(clean_evt)
    print(f'total uesful events for ped: {clean_evt}')
    if clean_evt == 0:
        print('There is no passed events! Use daq error, first few, and bias voltage for filter')
        pre_qual_cut_temp1 = np.copy(pre_qual_cut)
        pre_qual_cut_temp1[:, 5:9] = 0
        pre_qual_cut_temp1[:, 9] = 0
        pre_qual_cut_temp1[:, 10] = 0
        pre_qual_cut_temp1[:, 13:] = 0
        pre_qual_cut_sum_temp1 = np.nansum(pre_qual_cut_temp1, axis = 1)
        ped_qualities = np.logical_and(pre_qual_cut_sum_temp1 == 0, trig_type != 1)
        clean_evt = np.count_nonzero(ped_qualities)    
        clean_num_evts[1] = np.copy(clean_evt)
        print(f'total uesful events for ped: {clean_evt}')
    if clean_evt == 0:
        print('There is still no passed events! Use daq error, and first few for filter')
        pre_qual_cut_temp2 = np.copy(pre_qual_cut)
        pre_qual_cut_temp2[:, 5:9] = 0
        pre_qual_cut_temp2[:, 9] = 0
        pre_qual_cut_temp2[:, 10] = 0
        pre_qual_cut_temp2[:, 12] = 0
        pre_qual_cut_temp2[:, 13:] = 0
        pre_qual_cut_sum_temp2 = np.nansum(pre_qual_cut_temp2, axis = 1)
        ped_qualities = np.logical_and(pre_qual_cut_sum_temp2 == 0, trig_type != 1)
        clean_evt = np.count_nonzero(ped_qualities)
        clean_num_evts[2] = np.copy(clean_evt)
        print(f'total uesful events for ped: {clean_evt}')
    if clean_evt == 0:
        print('There is still x 2 no passed events! Use only daq error for filter')
        pre_qual_cut_temp3 = np.copy(pre_qual_cut)
        pre_qual_cut_temp3[:, 5:9] = 0
        pre_qual_cut_temp3[:, 9] = 0
        pre_qual_cut_temp3[:, 10] = 0
        pre_qual_cut_temp3[:, 11] = 0
        pre_qual_cut_temp3[:, 12] = 0
        pre_qual_cut_temp3[:, 13:] = 0
        pre_qual_cut_sum_temp3 = np.nansum(pre_qual_cut_temp3, axis = 1)
        ped_qualities = np.logical_and(pre_qual_cut_sum_temp3 == 0, trig_type != 1)
        clean_evt = np.count_nonzero(ped_qualities)
        clean_num_evts[3] = np.copy(clean_evt)
        print(f'total uesful events for ped: {clean_evt}')
    if clean_evt == 0:
        print('There is still x 3 no passed events! Use them all...')
        pre_qual_cut_temp4 = np.copy(pre_qual_cut)
        pre_qual_cut_temp4[:] = 0
        pre_qual_cut_sum_temp4 = np.nansum(pre_qual_cut_temp4, axis = 1)
        ped_qualities = np.logical_and(pre_qual_cut_sum_temp4 == 0, trig_type != 1)
        clean_evt = np.count_nonzero(ped_qualities)
        clean_num_evts[4] = np.copy(clean_evt)
        print(f'total uesful events for ped: {clean_evt}')
    del pre_qual_cut_sum

    # ped counter
    ped_counts = np.full((num_blks, num_eles), 0, dtype = int)
   
    irs_block_number = ara_uproot.irs_block_number & 0x1ff
    channel_mask = ara_uproot.channel_mask 
    dda_number = ((channel_mask & 0x300) >> 8) * num_chs
    bi_ch_mask = 1 << np.arange(num_chs, dtype = int)
    x_bins = np.linspace(0, num_blks, num_blks + 1, dtype = int)
    y_bins = np.linspace(0, num_eles, num_eles + 1, dtype = int)
    del ara_uproot
    ped_evts = entry_num[ped_qualities]
    trim_1st_blk = num_ddas

    for evt in tqdm(ped_evts):
        blk_idx = np.asarray(irs_block_number[int(evt)][trim_1st_blk:], dtype = int)
        ch_mask = np.asarray(channel_mask[int(evt)][trim_1st_blk:], dtype = int)
        dda_idx = np.asarray(dda_number[int(evt)][trim_1st_blk:], dtype = int)
        
        blk_idx_expand = np.repeat(blk_idx[:, np.newaxis], num_chs, axis = 1).flatten()
        ele_ch = np.repeat(ch_mask[:, np.newaxis], num_chs, axis = 1) 
        ele_ch = ele_ch & bi_ch_mask[np.newaxis, :]
        bad_ch_idx = (ele_ch == bi_ch_mask[np.newaxis, :]).flatten()
        ele_ch = np.log2(ele_ch).astype(int) + dda_idx[:, np.newaxis]
        ele_ch = ele_ch.flatten()

        x_hist = blk_idx_expand[bad_ch_idx]
        y_hist = ele_ch[bad_ch_idx]
        ped_counts += np.histogram2d(x_hist, y_hist, bins = (x_bins, y_bins))[0].astype(int)
        del blk_idx, ch_mask, dda_idx, blk_idx_expand, ele_ch, bad_ch_idx, x_hist, y_hist
    del irs_block_number, channel_mask, dda_number, bi_ch_mask, x_bins, y_bins, entry_num, ped_evts, num_ddas, trim_1st_blk

    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{st}/ped/'
    if not os.path.exists(Output):
        os.makedirs(Output)

    txt_file_name = f'{Output}ped_qualities_A{st}_R{run}.dat'
    np.savetxt(txt_file_name, ped_qualities.astype(int), fmt='%i')
    print(f'output is {txt_file_name}')
    size_checker(txt_file_name)

    txt_file_name = f'{Output}ped_counts_A{st}_R{run}.dat'
    np.savetxt(txt_file_name, ped_counts, fmt='%i')
    print(f'output is {txt_file_name}')
    size_checker(txt_file_name)

    print('Ped is done!')

    return {'evt_num':evt_num,
            'clean_num_evts':clean_num_evts,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'pre_qual_cut_temp1':pre_qual_cut_temp1,
            'pre_qual_cut_temp2':pre_qual_cut_temp2,
            'pre_qual_cut_temp3':pre_qual_cut_temp3,
            'pre_qual_cut_temp4':pre_qual_cut_temp4,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts}




