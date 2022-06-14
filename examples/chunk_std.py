import numpy as np
from tqdm import tqdm
import os, sys
import h5py

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')

def std_collector(Data, Ped, Station, Year):

    print('Collecting wf starts!')

    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import hist_loader

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Ped, Station, Year)
    num_evts = ara_root.num_evts

    # output array
    std_rf = []
    std_cal = []
    std_soft = []

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)

        # trigger info
        trig_type = ara_root.get_trig_type()

        # loop over the antennas
        std = np.full((num_ants), np.nan, dtype = float)
        for ant in range(num_ants):
            wf_v = ara_root.get_rf_ch_wf(ant)[1]
            std[ant] = np.nanstd(wf_v)
            del wf_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        if trig_type == 0:
            std_rf.append(std)
        elif trig_type == 1:
            std_cal.append(std)
        else:
            std_soft.append(std)
        del trig_type
    del ara_root, num_evts, num_ants

    # to numpy
    std_rf = np.asarray(std_rf)
    std_cal = np.asarray(std_cal)
    std_soft = np.asarray(std_soft)

    print('WF collecting is done!')

    return {'std_rf':std_rf,
            'std_cal':std_cal,
            'std_soft':std_soft}

def script_loader(Data, Ped, Station, Run, Year):

    from tools.ara_utility import size_checker

    # run the chunk code
    results = std_collector(Data, Ped, Station, Year)

    # create output dir
    Output = '/home/mkim/analysis/MF_filters/examples/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}std_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name}')

    # quick size check
    size_checker(h5_file_name)

if __name__ == "__main__":

    if len (sys.argv) < 5:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Data path ex)/data/exp/ARA/2014/unblinded/L1/ARA02/1027/run004434/event004434.root>
    <Ped ex)/misc/disk19/users/mkim/OMF_filter/ARA02/ped_full/ped_full_values_A2_R4434.dat>
    <Station ex)2>
    <Run ex)4434>
    <Year ex)2015>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    data = str(sys.argv[1])
    ped = str(sys.argv[2])
    station = int(sys.argv[3])
    run = int(sys.argv[4])
    year = int(sys.argv[5])
    
    script_loader(Data = data, Ped = ped, Station = station, Run = run, Year = year)

