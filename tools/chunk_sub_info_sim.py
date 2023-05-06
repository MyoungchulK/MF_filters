import os
import h5py
import numpy as np
from tqdm import tqdm

def sub_info_sim_collector(Data, Station, Year):

    print('Collecting sim sub info starts!')

    from tools.ara_sim_load import ara_root_loader

    if Data.find('signal') != -1:
        print('Data is signal sim!')
        get_angle_info = True
    else:
        print('Data is noise sim!')
        get_angle_info = False

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = get_angle_info)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    dt = ara_root.time_step
    wf_len = np.array([ara_root.waveform_length], dtype = int)
    wf_time = ara_root.wf_time    
    radius = ara_root.posnu_radius
    pnu = ara_root.pnu
    exponent_range = ara_root.exponent_range
    inu_thrown = ara_root.inu_thrown
    weight = ara_root.weight
    probability = ara_root.probability
    nuflavorint = ara_root.nuflavorint
    nu_nubar = ara_root.nu_nubar
    currentint = ara_root.currentint
    elast_y = ara_root.elast_y
    posnu = ara_root.posnu
    nnu = ara_root.nnu
    nnu_tot = ara_root.nnu_tot
    rec_ang = ara_root.rec_ang
    view_ang = ara_root.view_ang
    arrival_time = ara_root.arrival_time
    del ara_root

    st_center = np.array([10000, 10000, 6.35944e6], dtype = float)
    posnu_vec = posnu[:3] - st_center[:, np.newaxis] 

    theta_unit_x = 0
    theta_unit_y = 0
    theta_unit_z = 1
    AB = theta_unit_x * posnu_vec[0] + theta_unit_y * posnu_vec[1] + theta_unit_z * posnu_vec[2] # A.B
    ABabs = np.sqrt(theta_unit_x**2 + theta_unit_y**2 + theta_unit_z**2) * np.sqrt(posnu_vec[0]**2 + posnu_vec[1]**2 + posnu_vec[2]**2) # |AB|
    elevation_ang = 90 - np.degrees(np.arccos(AB / ABabs))
    del theta_unit_x, theta_unit_y, theta_unit_z, AB, ABabs

    phi_unit_x = 1
    phi_unit_y = 0
    AD = phi_unit_x * posnu_vec[0] + phi_unit_y * posnu_vec[1]
    ADabs = np.sqrt(phi_unit_x**2 + phi_unit_y**2) * np.sqrt(posnu_vec[0]**2 + posnu_vec[1]**2)
    azimuth_ang = np.degrees(np.arccos(AD / ADabs))
    azimuth_ang[posnu_vec[1] < 0] *= -1
    del phi_unit_x, phi_unit_y, AD, ADabs

    radius_ang = np.sqrt((posnu_vec[0])**2 + (posnu_vec[1])**2)
 
    print('Sub info sim collecting is done!')

    return {'entry_num':entry_num,
            'dt':dt,
            'wf_len':wf_len,        
            'wf_time':wf_time,
            'radius':radius,
            'pnu':pnu,
            'exponent_range':exponent_range,
            'inu_thrown':inu_thrown,
            'weight':weight,
            'probability':probability,
            'nuflavorint':nuflavorint,
            'nu_nubar':nu_nubar,
            'currentint':currentint,
            'elast_y':elast_y,
            'posnu':posnu,
            'nnu':nnu,
            'nnu_tot':nnu_tot,
            'rec_ang':rec_ang,    
            'view_ang':view_ang,
            'arrival_time':arrival_time,
            'st_center':st_center,
            'posnu_vec':posnu_vec,
            'elevation_ang':elevation_ang,
            'azimuth_ang':azimuth_ang,
            'radius_ang':radius_ang}
