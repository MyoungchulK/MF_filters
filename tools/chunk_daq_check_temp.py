import numpy as np
from tqdm import tqdm

def daq_check_temp_collector(Data, Ped, analyze_blind_dat = False):

    print('DAQ checking starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import quick_qual_check

    ara_const = ara_const()
    num_ddas = ara_const.DDA_PER_ATRI
    num_blks = ara_const.BLOCKS_PER_DDA
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()

    irs_block_number = ara_uproot.irs_block_number
    channel_mask = ara_uproot.channel_mask

    blk_len = np.full((num_evts), 0, dtype = int)
    blk_num = np.copy(blk_len)
    dda_num = np.copy(blk_len)
    ch_num = np.copy(blk_len)
    blk_gap = np.copy(blk_len)

    ch_num_by = 1 << np.arange(8, dtype = int) 
    ch_num_len = len(ch_num_by)

    for evt in tqdm(range(num_evts)):

      if evt_num[evt] == 33488:

        blk_idx_evt = np.asarray(irs_block_number[evt], dtype = int)
        blk_len[evt] = len(blk_idx_evt) % num_ddas

        print(irs_block_number[evt])

        blk_num_reshape = np.reshape(blk_idx_evt, (-1,4))
        blk_num[evt] = int(np.any(blk_num_reshape != blk_num_reshape[:,0][:, np.newaxis]))

        ch_mask_evt = np.asarray(channel_mask[evt], dtype = int)
    
        dda_num_evt = (ch_mask_evt&0x300)>>8
        dda_num_reshape = np.reshape(dda_num_evt, (-1,4))
        dda_num[evt] = int(np.any(dda_num_reshape != dda_num_reshape[0][np.newaxis, :]))

        print(dda_num_evt)

        ch_num_reshape = np.repeat(ch_mask_evt[:, np.newaxis], ch_num_len, axis = 1)
        ch_num_bit = ch_num_reshape & ch_num_by[np.newaxis, :]
        ch_num[evt] = int(np.any(ch_num_bit != ch_num_by[np.newaxis, :]))

        print(ch_num_reshape)
        print(ch_num_bit)

        for dda in range(num_ddas):
            blk_num_evt = blk_num_reshape[:, dda]
            first_block_idx = blk_num_evt[0]
            last_block_idx = blk_num_evt[-1]
            block_diff = len(blk_num_evt) - 1

            if first_block_idx + block_diff != last_block_idx:
                if num_blks - first_block_idx + last_block_idx != block_diff:
                    blk_gap[evt] += 1
            del first_block_idx, last_block_idx, block_diff, blk_num_evt
        del blk_idx_evt, blk_num_reshape, dda_num_evt, dda_num_reshape, ch_mask_evt, ch_num_reshape, ch_num_bit

    quick_qual_check(blk_len != 0, evt_num, 'blk_length flag!')
    quick_qual_check(blk_num != 0, evt_num, 'blk_number flag!')
    quick_qual_check(dda_num != 0, evt_num, 'dda_number flag!')
    quick_qual_check(ch_num != 0, evt_num, 'ch_mask flag!')
    quick_qual_check(blk_gap != 0, evt_num, 'blk_gap flag!')
    del num_ddas, num_blks, ara_uproot, num_evts, irs_block_number, channel_mask, ch_num_by, ch_num_len

    print('DAQ checking is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'blk_len':blk_len,
            'blk_num':blk_num,
            'dda_num':dda_num,
            'ch_num':ch_num,
            'blk_gap':blk_gap}




