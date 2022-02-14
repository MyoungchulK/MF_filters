import numpy as np
from tqdm import tqdm

def arr_table_collector(Data, Ped):

    print('Collecting arrival time starts!')

    from tools.ara_sim_load import ara_raytrace_loader

    ara_ray = ara_raytrace_loader(1.35,1.78,0.0132)
    ara_ray.get_src_trg_position(3, 2018)

    arr_time_table = ara_ray.get_arrival_time_table()








    print('Arrival time collecting is done!')
    """
    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'pre_qual_cut':pre_qual_cut,
            'corr_v':corr_v,
            'corr_h':corr_h}
    """







