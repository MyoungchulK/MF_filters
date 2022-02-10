import numpy as np
from tqdm import tqdm

def repeder_collector(Data, Ped = '0', debug = False):

    print('Making pedestal starts!')

    from tools.ara_data_load import ara_root_loader
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import repeder_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import hist_loader
    from tools.ara_quality_cut import pre_qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_eles = ara_const.CHANNELS_PER_ATRI
    num_buffers = ara_const.SAMPLES_PER_DDA
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut results
    trig_type = ara_uproot.get_trig_type()
    entry_num = ara_uproot.entry_num
    evt_num = ara_uproot.evt_num

    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True, analyze_blind_dat = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut[:,-1] = 0 # exclude sensor file cut
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del pre_qual, pre_qual_cut

    clean_idx = np.logical_and(trig_type != 1, pre_qual_cut_sum == 0)
    clean_entry = entry_num[clean_idx]
    clean_evt = evt_num[clean_idx]
    if len(clean_entry) == 0:
        print('There is no passed events! Use all events...')
        clean_entry = entry_num[trig_type != 1]
        clean_evt = evt_num[trig_type != 1]
    print(f'Number of clean event is {len(clean_entry)}')
    del entry_num, evt_num, trig_type, pre_qual_cut_sum

    # output array
    ara_repeder = repeder_loader(ara_uproot, trim_1st_blk = True)
    ele_ch = ara_root.ara_geom.get_ele_ch_idx()
    samp_medi_int = np.full((num_buffers, num_eles), 0, dtype = int)
    if debug:
        print('Debug mode!')
        ara_hist = hist_loader(chs = num_eles)
        buffer_bit_range = ara_hist.y_range
        buffer_sample_range = ara_hist.x_range
        samp_map = ara_hist.hist_map
        samp_medi = np.full((num_buffers, num_eles), np.nan, dtype = float)
        del ara_hist
    else:
        print('Lite mode!')
    del ara_uproot, num_buffers

    for ant in tqdm(range(num_eles)):
      #if ant < 1:

        # sample map
        ara_hist = hist_loader(chs = 1)

        # loop over the events
        for evt in clean_entry:
          #if evt < 10:
       
            # get entry and wf
            ara_root.get_entry(int(evt))
            ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlock)

            # sample index
            samp_idx = ara_repeder.get_samp_idx(int(evt))

            # stack in sample map
            raw_v = ara_root.get_ele_ch_wf(ant)[1].astype(int)
            ara_hist.stack_in_hist(samp_idx, raw_v)
            del raw_v, samp_idx 
            ara_root.del_TGraph()
            ara_root.del_usefulEvt()

        samp_medi_ant = ara_hist.get_median_est(nan_to_zero = True)[:, 0]
        samp_medi_ant_int = samp_medi_ant.astype(int)
        samp_medi_int[:, ant] = samp_medi_ant_int
        if debug:
            samp_map[:, :, ant] = ara_hist.hist_map[:, :, 0]
            samp_medi[:, ant] = samp_medi_ant
        ara_hist.del_hist_map()
        del ara_hist

        ara_repeder.get_pedestal_foramt(samp_medi_ant_int, ant)
        del samp_medi_ant, samp_medi_ant_int
    del ara_root, clean_entry, num_eles

    ped_arr = ara_repeder.ped_arr
    del ara_repeder

    if debug:
        repeder_dict = {'ped_arr':ped_arr,'ele_ch':ele_ch,'clean_evt':clean_evt,'buffer_bit_range':buffer_bit_range,'buffer_sample_range':buffer_sample_range,'samp_medi':samp_medi,'samp_medi_int':samp_medi_int,'samp_map':samp_map}
    else:
        repeder_dict = {'ped_arr':ped_arr,'ele_ch':ele_ch,'clean_evt':clean_evt,'samp_medi_int':samp_medi_int}

    print('Pedestal making is done!')

    return repeder_dict








