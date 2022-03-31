import numpy as np
from tqdm import tqdm

def temp_sim_collector(Data, Station, Year):

    print('Collecting template starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_sim_matched_filter import ara_sim_matched_filter
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
    wf_len = ara_root.waveform_length
    wf_time = ara_root.wf_time    
    
    # template parameter
    nu_flavor = ['NuE', 'NuMu']
    nu_current = ['CC', 'NC']
    nu_elst = np.array([np.array([0.1, 0.4, 0.7]), np.array([0.9, 0.6, 0.3])])
    nu_elst_int = (nu_elst * 10).astype(int)
    off_cone = np.arange(0, 5, 1, dtype = int)
    ant_res = np.arange(0, -76, -15, dtype = int)    
    ant_ch = np.arange(num_ants, dtype = int)

    # load path
    param_path = f'../sim/temp_A{Station}_setup_parameter.txt'
    param = open(param_path, 'r')
    p_lines = param.readlines()
    print(p_lines[0])

    # wf arr
    temp = np.full((wf_len, num_ants, len(ant_res), len(off_cone), len(nu_flavor), len(nu_elst[0])), 0, dtype = float)
    
    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        temp_idx = p_lines[evt+1].split(' ')
        if str(temp_idx[1]) == 'NuE':
            fla_idx = 0
        if str(temp_idx[1]) == 'NuMu':
            fla_idx = 1
        elst_idx = np.where(nu_elst_int[fla_idx] == int(float(temp_idx[-4])*10))[0][0]
        res_idx = np.where(ant_res == int(temp_idx[-3]))[0][0]

        temp_wf = ara_root.get_rf_wfs(evt)[:, int(temp_idx[-1])]
        temp_wf_len = np.count_nonzero(temp_wf)
        temp_wf_nonzero = temp_wf[temp_wf != 0]

        temp[:temp_wf_len, int(temp_idx[-1]), res_idx, int(temp_idx[-2]), fla_idx, elst_idx] = temp_wf_nonzero

    param.close()
    del ara_root, num_evts

    print('Template collecting is done!')

    return {'dt':np.asarray([dt]),
            'time_pad':time_pad,
            'temp':temp,
            'nu_flavor':np.array([0,1], dtype = int),
            'nu_elst':nu_elst,
            'off_cone':off_cone,
            'ant_res':ant_res,
            'ant_ch':ant_ch}
    
    

