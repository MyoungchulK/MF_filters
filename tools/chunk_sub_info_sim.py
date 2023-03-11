import os
import h5py
import numpy as np
from tqdm import tqdm

def sub_info_sim_collector(Data, Station, Year):

    print('Collecting sim sub info starts!')

    from tools.ara_sim_load import ara_root_loader

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = True)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    dt = ara_root.time_step
    wf_len = np.array([ara_root.waveform_length], dtype = int)
    wf_time = ara_root.wf_time    
    radius = ara_root.posnu_radius
    pnu = ara_root.pnu
    inu_thrown = ara_root.inu_thrown
    weight = ara_root.weight
    probability = ara_root.probability
    nuflavorint = ara_root.nuflavorint
    nu_nubar = ara_root.nu_nubar
    currentint = ara_root.currentint
    elast_y = ara_root.elast_y
    posnu = ara_root.posnu
    nnu = ara_root.nnu
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    arrival_time = ara_root.arrival_time
    del ara_root
    
    print('Sub info sim collecting is done!')

    return {'entry_num':entry_num,
            'dt':dt,
            'wf_len':wf_len,        
            'wf_time':wf_time,
            'radius':radius,
            'pnu':pnu,
            'inu_thrown':inu_thrown,
            'weight':weight,
            'probability':probability,
            'nuflavorint':nuflavorint,
            'nu_nubar':nu_nubar,
            'currentint':currentint,
            'elast_y':elast_y,
            'posnu':posnu,
            'nnu':nnu,
            'rec_ang':rec_ang,    
            'view_ang':view_ang,
            'arrival_time':arrival_time}
