import numpy as np
from tqdm import tqdm

def repeder_collector(Data, Ped, debug = False):

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
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut results
    trig_type = ara_uproot.get_trig_type()
    entry_num = ara_uproot.entry_num
    evt_num = ara_uproot.evt_num

    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    pre_qual_cut[:,7] = 0 # exclude sensor file cut
    pre_qual_cut_sum = np.nansum(pre_qual_cut, axis = 1)
    del pre_qual, pre_qual_cut

    clean_idx = np.logical_and(trig_type != 1, pre_qual_cut_sum == 0)
    clean_entry = entry_num[clean_idx]
    clean_evt = evt_num[clean_idx]
    if len(clean_entry) == 0:
        print('There is no passed events! Use all events...')
        clean_entry = entry_num[trig_type != 1]
        clean_evt = evt_num[trig_trig != 1]
    print(f'Number of clean event is {len(clean_entry)}')
    del entry_num, evt_num, trig_type, pre_qual_cut_sum

    # output array
    ara_hist = hist_loader(chs = num_eles)
    ara_repeder = repeder_loader(ara_uproot, trim_1st_blk = True)
    del ara_uproot

    # loop over the events
    for evt in tqdm(clean_entry):
      #if evt < 10:
       
        # get entry and wf
        ara_root.get_entry(int(evt))
        ara_root.get_useful_evt(ara_root.cal_type.kOnlyADCWithOut1stBlock)

        # sample index
        samp_idx = ara_repeder.get_samp_idx(evt)

        # loop over the antennas
        for ant in range(num_eles):

            # stack in sample map
            raw_v = ara_root.get_ele_ch_wf(ant)[1].astype(int)
            ara_hist.stack_in_hist(samp_idx, raw_v, ant)
            del raw_v 
            ara_root.del_TGraph()
        del samp_idx
        ara_root.del_usefulEvt()
    del ara_root, clean_entry, num_eles

    samp_medi = ara_hist.get_median_est()
    samp_medi_int = samp_medi.astype(int)
    if debug:
        samp_map = ara_hist.hist_map
        buffer_bit_range = ara_hist.y_range
        buffer_sample_range = ara_hist.x_range
    else:
        del clean_evt, samp_medi
    del ara_hist

    ped_arr = ara_repeder.get_pedestal_foramt(samp_medi_int)
    del ara_repeder

    if debug:
        repeder_dict = {'ped_arr':ped_arr,'clean_evt':clean_evt,'samp_medi':samp_medi,'samp_medi_int':samp_medi_int,'samp_map':samp_map,'buffer_bit_range':buffer_bit_range,'buffer_sample_range':buffer_sample_range}
    else:
        repeder_dict = {'ped_arr':ped_arr}

    print('Pedestal making is done!')

    return repeder_dict








