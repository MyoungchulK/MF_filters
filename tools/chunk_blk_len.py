import numpy as np
from tqdm import tqdm

def blk_len_collector(Data, Ped):

    print('Blk length starts!')

    from tools.ara_constant import ara_const
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import pre_qual_cut_loader

    ara_const = ara_const()
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()

    read_win = ara_uproot.read_win
    blk_len = read_win//num_ddas - 1
    del read_win, num_ddas

    # quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    del pre_qual, ara_uproot, pre_qual_cut_temp

    blk_range = np.arange(0, 100, dtype = int)
    blk_bins = np.linspace(0, 100, 100+1)
    rf_blk_hist = np.histogram(blk_len[trig_type == 0], bins = blk_bins)[0].astype(int)
    cal_blk_hist = np.histogram(blk_len[trig_type == 1], bins = blk_bins)[0].astype(int)
    soft_blk_hist = np.histogram(blk_len[trig_type == 2], bins = blk_bins)[0].astype(int)
    rf_blk_hist_w_cut = np.histogram(blk_len[(trig_type == 0) & (pre_qual_cut_sum == 0)], bins = blk_bins)[0].astype(int)
    cal_blk_hist_w_cut = np.histogram(blk_len[(trig_type == 1) & (pre_qual_cut_sum == 0)], bins = blk_bins)[0].astype(int)
    soft_blk_hist_w_cut = np.histogram(blk_len[(trig_type == 2) & (pre_qual_cut_sum == 0)], bins = blk_bins)[0].astype(int)
    del pre_qual_cut_sum
    
    print('Blk length is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'blk_len':blk_len,
            'blk_range':blk_range,
            'blk_bins':blk_bins,
            'rf_blk_hist':rf_blk_hist,
            'cal_blk_hist':cal_blk_hist,
            'soft_blk_hist':soft_blk_hist,
            'rf_blk_hist_w_cut':rf_blk_hist_w_cut,
            'cal_blk_hist_w_cut':cal_blk_hist_w_cut,
            'soft_blk_hist_w_cut':soft_blk_hist_w_cut}




