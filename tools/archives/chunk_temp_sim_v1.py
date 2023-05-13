import numpy as np
from tqdm import tqdm

def temp_sim_collector(Data, Station, Year):

    print('Collecting template starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    dt = ara_root.time_step
    print(dt)
    wf_len = ara_root.waveform_length
    wf_time = ara_root.wf_time    
    #print(wf_len)   
    #print(wf_time)
 
    # template parameter
    nu_elst = np.array([0.1, 0.9])
    off_cone = np.arange(0, 4.1, 0.5)
    off_int = (off_cone * 2).astype(int)
    #print(off_int)
    ant_res = np.arange(0, -61, -10, dtype = int)    
    ant_ch = np.arange(num_ants, dtype = int)

    # load path
    param_path = f'../sim/sim_temp/temp_A{Station}_setup_parameter.txt'
    param = open(param_path, 'r')
    p_lines = param.readlines()
    print(p_lines[0])

    # wf arr
    temp = np.full((wf_len, num_ants, len(ant_res), len(off_cone), len(nu_elst)), 0, dtype = float)
    temp_ori = np.copy(temp)
    print(temp.shape)   
 
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        temp_idx = p_lines[evt+1].split(' ')
        if str(temp_idx[1]) == 'NuE':
            elst_idx = 0
        if str(temp_idx[1]) == 'NuMu':
            elst_idx = 1
        res_idx = np.where(ant_res == int(temp_idx[-3]))[0][0]
        ant_idx = int(temp_idx[-1])
        off_idx = np.where(off_int == int(float(temp_idx[-2])*2))[0][0]

        #print(evt, str(temp_idx[1]), elst_idx, int(temp_idx[-3]), res_idx, float(temp_idx[-2]), off_idx, ant_idx)

        temp_wf = ara_root.get_rf_wfs(evt)[:, ant_idx]
        temp_wf_len = np.count_nonzero(temp_wf)
        if temp_wf_len < 1:
            print('zero!:',evt)
        temp_wf_nonzero = temp_wf[temp_wf != 0]

        temp[:temp_wf_len, ant_idx, res_idx, off_idx, elst_idx] = temp_wf_nonzero
        temp_ori[:, ant_idx, res_idx, off_idx, elst_idx] = temp_wf

    param.close()
    del ara_root, num_evts

    print('Template collecting is done!')

    return {'dt':dt,
            'wf_time':wf_time,
            'temp':temp,
            'temp_ori':temp_ori,
            'nu_elst':nu_elst,
            'off_cone':off_cone,
            'ant_res':ant_res,
            'ant_ch':ant_ch}
    
    

