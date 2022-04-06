import numpy as np
from tqdm import tqdm

def blk_len_collector(Data, Ped):

    print('Blk length starts!')

    from tools.ara_constant import ara_const
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import analog_buffer_info_loader
    from tools.ara_quality_cut import pre_qual_cut_loader

    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION    
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts
    buffer_info = analog_buffer_info_loader(ara_uproot.station_id, ara_uproot.run, ara_uproot.year, incl_cable_delay = True)
    buffer_info.get_int_time_info()

    read_win = ara_uproot.read_win
    blk_len = read_win//num_ddas - 1
    del read_win, num_ddas

    # quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    del pre_qual, pre_qual_cut_temp

    wf_len = np.full((2, num_ants, num_evts), np.nan, dtype = float)
    int_wf_len = np.copy(wf_len)

    for evt in tqdm(range(num_evts)):

        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(evt, trim_1st_blk = True)
        if blk_idx_len == 0:
            continue
        del blk_idx_len
        time_arr = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True, ch_shape = True) 
        int_time_arr = buffer_info.get_time_arr(blk_idx_arr, trim_1st_blk = True, ch_shape = True, use_int_dat = True)         
        
        wf_len[0, :, evt] = np.nanmin(time_arr, axis = 0)
        wf_len[1, :, evt] = np.nanmax(time_arr, axis = 0)
        int_wf_len[0, :, evt] = np.nanmin(int_time_arr, axis = 0)
        int_wf_len[1, :, evt] = np.nanmax(int_time_arr, axis = 0)
        del blk_idx_arr, time_arr, int_time_arr
    del ara_uproot, num_evts, buffer_info

    blk_range = np.arange(0, 100, dtype = int)
    blk_bins = np.linspace(0, 100, 100+1)
    rf_blk_hist = np.histogram(blk_len[trig_type == 0], bins = blk_bins)[0].astype(int)
    cal_blk_hist = np.histogram(blk_len[trig_type == 1], bins = blk_bins)[0].astype(int)
    soft_blk_hist = np.histogram(blk_len[trig_type == 2], bins = blk_bins)[0].astype(int)
    rf_blk_hist_w_cut = np.histogram(blk_len[(trig_type == 0) & (pre_qual_cut_sum == 0)], bins = blk_bins)[0].astype(int)
    cal_blk_hist_w_cut = np.histogram(blk_len[(trig_type == 1) & (pre_qual_cut_sum == 0)], bins = blk_bins)[0].astype(int)
    soft_blk_hist_w_cut = np.histogram(blk_len[(trig_type == 2) & (pre_qual_cut_sum == 0)], bins = blk_bins)[0].astype(int)
   
    wf_range = np.arange(-500,1500,0.5) 
    wf_bins = np.linspace(-500,1500,4000+1)
    def hist_maker(dat):
        hist = np.full((2, num_ants, len(wf_range)), 0, dtype = int)
        for ant in range(num_ants):
            hist[0, ant] = np.histogram(dat[0, ant], bins = wf_bins)[0].astype(int)
            hist[1, ant] = np.histogram(dat[1, ant], bins = wf_bins)[0].astype(int)
        return hist
    rf_wf_hist = hist_maker(wf_len[:,:,trig_type == 0])
    cal_wf_hist = hist_maker(wf_len[:,:,trig_type == 1])
    soft_wf_hist = hist_maker(wf_len[:,:,trig_type == 2])
    int_rf_wf_hist = hist_maker(int_wf_len[:,:,trig_type == 0])
    int_cal_wf_hist = hist_maker(int_wf_len[:,:,trig_type == 1])
    int_soft_wf_hist = hist_maker(int_wf_len[:,:,trig_type == 2])

    rf_wf_hist_w_cut = hist_maker(wf_len[:,:,(trig_type == 0) & (pre_qual_cut_sum == 0)])
    cal_wf_hist_w_cut = hist_maker(wf_len[:,:,(trig_type == 1) & (pre_qual_cut_sum == 0)])
    soft_wf_hist_w_cut = hist_maker(wf_len[:,:,(trig_type == 2) & (pre_qual_cut_sum == 0)])
    int_rf_wf_hist_w_cut = hist_maker(int_wf_len[:,:,(trig_type == 0) & (pre_qual_cut_sum == 0)])
    int_cal_wf_hist_w_cut = hist_maker(int_wf_len[:,:,(trig_type == 1) & (pre_qual_cut_sum == 0)])
    int_soft_wf_hist_w_cut = hist_maker(int_wf_len[:,:,(trig_type == 2) & (pre_qual_cut_sum == 0)])
    del num_ants, pre_qual_cut_sum

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
            'soft_blk_hist_w_cut':soft_blk_hist_w_cut,
            'wf_len':wf_len,
            'int_wf_len':int_wf_len,
            'wf_range':wf_range,
            'wf_bins':wf_bins,
            'rf_wf_hist':rf_wf_hist,
            'cal_wf_hist':cal_wf_hist,
            'soft_wf_hist':soft_wf_hist,
            'rf_wf_hist_w_cut':rf_wf_hist_w_cut,
            'cal_wf_hist_w_cut':cal_wf_hist_w_cut,
            'soft_wf_hist_w_cut':soft_wf_hist_w_cut,
            'int_rf_wf_hist':int_rf_wf_hist,
            'int_cal_wf_hist':int_cal_wf_hist,
            'int_soft_wf_hist':int_soft_wf_hist,
            'int_rf_wf_hist_w_cut':int_rf_wf_hist_w_cut,
            'int_cal_wf_hist_w_cut':int_cal_wf_hist_w_cut,
            'int_soft_wf_hist_w_cut':int_soft_wf_hist_w_cut}




