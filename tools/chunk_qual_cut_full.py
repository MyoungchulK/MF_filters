import numpy as np
from tqdm import tqdm

def qual_cut_full_collector(Data, Ped):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import pre_qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()

    # quality cut
    pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = True, analyze_blind_dat = True)
    pre_qual_cut = pre_qual.run_pre_qual_cut()
    del pre_qual, ara_uproot

    # clean evts for repeder
    pre_qual_cut_temp = np.copy(pre_qual_cut)
    pre_qual_cut_temp[:, -1] = 0
    pre_qual_cut_sum = np.nansum(pre_qual_cut_temp, axis = 1)
    clean_repeder_evt = np.logical_and(pre_qual_cut_sum == 0, trig_type != 1).astype(int)
    if np.count_nonzero(clean_repeder_evt) == 0:
        print('There is no passed events! Use all events...')    
        clean_repeder_evt = (trig_type != 1).astype(int)
    del pre_qual_cut_temp, pre_qual_cut_sum

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'clean_repeder_evt':clean_repeder_evt}




