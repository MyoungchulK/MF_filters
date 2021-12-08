import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_root import ara_root_lib
from tools.ara_root import ara_raw_to_qual
import tools.antenna as ant
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
from tools.run import data_info_reader
from tools.chunk import wf_collector

def ms_filter(Data, Ped, Output):

    # read data info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)

    # import root and ara root lib
    ROOT = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, eventTree, rawEvent, num_events, calibrator, qual = ara_raw_to_qual(ROOT, Data, Ped, Station)
    del Data, Ped

    # known configuration. Probably can call from actual data file through the AraRoot in future....
    #antenna
    num_Antennas = ant.antenna_info()[2]
    # interpolation time width
    time_width_ns = interpolation_bin_width()[0]

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(time_width_ns)[1:]

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)

    # collecting WF (chunk analysis way)
    wf_all, int_wf_len_all = wf_collector(ROOT, eventTree, rawEvent, 0, num_events, calibrator, qual # ara root
                                                        , num_Antennas, time_width_ns # known config
                                                        , time_pad_len, time_pad_i, time_pad_f # time
                                                        , trig_set = 2, qual_set = 1, wf_len = True)
    print(wf_all.shape)
    print(int_wf_len_all.shape)

    # create output file
    os.chdir(Output)
    h5_file_name=f'WF_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('time', data=np.arange(time_pad_i, time_pad_f+time_width_ns, time_width_ns), compression="gzip", compression_opts=9)
    hf.create_dataset('volt', data=wf_all, compression="gzip", compression_opts=9)
    hf.create_dataset('int_wf_len', data=int_wf_len_all, compression="gzip", compression_opts=9)
    hf.close()
    #del hf, freq, soft_psd, soft_psd_wo_band, Config

    print(f'output is {Output}{h5_file_name}')
    del Output, h5_file_name
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) !=4:
        Usage = """
    This is designed to analyze all events in the run. You have to choose specific run.
    Depending on DMode, It will save just psd or all middle step information.
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/PSD/>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Data=str(sys.argv[1])
    Ped=str(sys.argv[2])
    Output=str(sys.argv[3])
    ms_filter(Data, Ped, Output)

del curr_path













    
