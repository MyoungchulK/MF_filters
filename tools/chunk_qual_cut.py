import numpy as np
from tqdm import tqdm

def qual_cut_collector(Data, Ped):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_quality_cut import qual_cut_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    # quality cut config
    ara_qual = qual_cut_loader(ara_root, ara_uproot, trim_1st_blk = True)

    # pre quality cut
    pre_qual_cut = ara_qual.pre_qual.run_pre_qual_cut() 

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt<100:
        # post quality cut
        ara_qual.post_qual.get_post_qual_cut(evt)

    # post quality cut
    post_qual_cut = ara_qual.post_qual.run_post_qual_cut()  

    num_evts = ara_uproot.num_evts
    trig_type = ara_uproot.get_trig_type()
    if num_evts != 0:
        trig_ratio = np.full((3), np.nan, dtype = float)
        trig_ratio[0] = np.count_nonzero(trig_type == 0)
        trig_ratio[1] = np.count_nonzero(trig_type == 1)
        trig_ratio[2] = np.count_nonzero(trig_type == 2)
        trig_ratio /= num_evts
    else:
        trig_ratio = np.full((3), np.nan, dtype = float)
    bv_idx = pre_qual_cut[:,6] == 0
    if np.any(bv_idx):
        trig_type_wo_bv = trig_type[bv_idx]
        num_evts_wo_bv = len(trig_type_wo_bv)
        trig_ratio_wo_bv = np.full((3), np.nan, dtype = float)
        trig_ratio_wo_bv[0] = np.count_nonzero(trig_type_wo_bv == 0)
        trig_ratio_wo_bv[1] = np.count_nonzero(trig_type_wo_bv == 1)
        trig_ratio_wo_bv[2] = np.count_nonzero(trig_type_wo_bv == 2)
        trig_ratio_wo_bv /= num_evts_wo_bv
        del num_evts_wo_bv, trig_type_wo_bv
    else:
        trig_ratio_wo_bv = np.full((3), np.nan, dtype = float)
    del trig_type, bv_idx

    del ara_root, ara_uproot, num_evts, ara_qual

    print(f'trig ratio:',trig_ratio)
    print(f'trig ratio wo bv:',trig_ratio_wo_bv)
 
    print('Quality cut is done!')

    return {'pre_qual_cut':pre_qual_cut,
            'post_qual_cut':post_qual_cut,
            'trig_ratio':trig_ratio,
            'trig_ratio_wo_bv':trig_ratio_wo_bv}





