import numpy as np
from tqdm import tqdm

def aeff_sim_collector(Data, Station, Year):

    print('Collecting aeff starts!')

    from tools.ara_sim_load import ara_root_loader

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts
    evt_num = np.arange(num_evts, dtype = int)
    ara_root.get_sub_info(Data)
    time_step = ara_root.time_step
    posnu_radius = ara_root.posnu_radius
    pnu = ara_root.pnu
    nuflavorint = ara_root.nuflavorint
    nu_nubar = ara_root.nu_nubar
    inu_thrown = ara_root.inu_thrown
    weight = ara_root.weight
    probability = ara_root.probability
    currentint = ara_root.currentint
    elast_y = ara_root.elast_y
    posnu = ara_root.posnu
    nnu = ara_root.nnu

    print('aeff collecting is done!')

    return {'evt_num':evt_num,
            'time_step':time_step,
            'posnu_radius':posnu_radius,
            'pnu':pnu,
            'nuflavorint':nuflavorint,
            'nu_nubar':nu_nubar,
            'inu_thrown':inu_thrown,
            'weight':weight,
            'probability':probability,
            'currentint':currentint,
            'elast_y':elast_y,
            'nnu':nnu}

