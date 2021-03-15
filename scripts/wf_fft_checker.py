import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_root import ara_root_lib
from tools.ara_root import ara_raw_to_qual
from tools.ara_root import trig_checker
from tools.ara_root import qual_checker
import tools.antenna as ant
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
from tools.wf import TGraph_to_raw
from tools.wf import akima_interp
from tools.fft import freq_pad_maker
from tools.run import data_info_reader
from tools.ara_root import useful_evt_maker
from tools.array import arr_1d
from tools.plot import plot_16_indi

def wf_fft_checker(Data, Ped, Output, Sel_evt = None):

    # read data info
    Station, Run, Config, Year, MD = data_info_reader(Data)

    # import root and ara root lib
    ROOT = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, eventTree, rawEvent, num_events, calibrator, qual = ara_raw_to_qual(ROOT, Data, Ped, Station)
    del Data, Ped

    if Sel_evt is not None:
        pass
    else:
        Sel_evt = np.random.choice(np.arange(10,num_events),1)[0] 
    print('selected event:', Sel_evt)

    # known configuration. Probably can call from actual data file through the AraRoot in future....
    #antenna
    Antenna, ant_index, num_Antennas = ant.antenna_info()
    # masked antenna
    bad_ant_index = ant.bad_antenna(Station, Run)
    # interpolation time width
    time_width_ns, time_width_s, Ndf =interpolation_bin_width()

    # make wf pad
    time_pad, time_pad_len, time_pad_i, time_pad_f = time_pad_maker(time_width_ns)

    # make freq pad
    freq = freq_pad_maker(time_pad_len, time_width_s)[0]

    # make a useful event
    usefulEvent = useful_evt_maker(ROOT, eventTree, rawEvent, Sel_evt, calibrator)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
   
    h5_file_name=f'WF_FFT_A{Station}_R{Run}_E{Sel_evt}.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    g0 = hf.create_group('Evt_info')
    g1 = hf.create_group('Raw_WF')
    g2 = hf.create_group('Int_WF')
    g3 = hf.create_group('Pad_WF')
    g4 = hf.create_group('Int_FFT')
    g5 = hf.create_group('Pad_FFT')

    # trigger type check
    trig_name = ['RF', 'Cal', 'Soft', 'Unknown']
    trig_num = trig_checker(rawEvent)
    trig = trig_name[trig_num]
    print('Trigger:', trig)
    del trig_name

    # quality tyep check
    qual_name = ['Cut','Pass','Unknown']
    qual_num = qual_checker(qual, usefulEvent)
    quality = qual_name[qual_num]
    print('Quality:', quality)
    del qual_name    

    # save event info
    g0.create_dataset('Config', data=np.array([Config]), compression="gzip", compression_opts=9)
    g0.create_dataset('Year', data=np.array([Year]), compression="gzip", compression_opts=9)
    g0.create_dataset('MD', data=np.array([int(MD)]), compression="gzip", compression_opts=9)
    g0.create_dataset('Ant_index', data=ant_index, compression="gzip", compression_opts=9)
    g0.attrs['Antenna_label'] = Antenna
    g0.create_dataset('Trig', data=np.array([trig_num]), compression="gzip", compression_opts=9)
    g0.attrs['Trigger'] = trig
    g0.create_dataset('Qual', data=np.array([qual_num]), compression="gzip", compression_opts=9)
    g0.attrs['Quality'] = quality
    g0.create_dataset('Bad_Ant', data=np.array(bad_ant_index), compression="gzip", compression_opts=9)
    bad_ant_label = Antenna[bad_ant_index[0]]
    g0.attrs['Bad_Antenna'] = bad_ant_label
    del g0

    # list for plot
    raw_wf_t = []
    raw_wf_v = []
    int_wf_t = []
    int_wf_v = []
    pad_wf_t = []
    pad_wf_v = []
    int_fft_f = []
    int_fft_v = []
    pad_fft_f = []
    pad_fft_v = []
    
    # loop over the antennas
    for ant_i in range(num_Antennas):

        # TGraph
        graph = usefulEvent.getGraphFromRFChan(ant_i)

        # raw wf
        raw_t, raw_v = TGraph_to_raw(graph)
        raw_wf_t.append(raw_t)
        raw_wf_v.append(raw_v)

        # int wf 
        int_ti, int_tf, int_v, int_len = akima_interp(raw_t, raw_v, time_width_ns)
        int_t = np.arange(int_ti, int_tf + time_width_ns, time_width_ns)
        int_wf_t.append(int_t)
        int_wf_v.append(int_v)

        # pad wf
        pad_v = arr_1d(time_pad_len, 0, float)
        pad_v[int((int_ti - time_pad_i) / time_width_ns):-int((time_pad_f - int_tf) / time_width_ns)] = int_v
        pad_wf_t.append(time_pad)
        pad_wf_v.append(pad_v)
    
        # save wf
        g1.create_dataset(f'Ch{ant_i}', data=np.stack([raw_t, raw_v], axis=-1), compression="gzip", compression_opts=9)
        g2.create_dataset(f'Ch{ant_i}', data=np.stack([int_t, int_v], axis=-1), compression="gzip", compression_opts=9)
        g3.create_dataset(f'Ch{ant_i}', data=np.stack([time_pad, pad_v], axis=-1), compression="gzip", compression_opts=9)

        # int fft
        int_freq = freq_pad_maker(int_len, time_width_s)[0]
        int_fft = np.fft.fft(int_v / 1e3) / Ndf
        int_fft_f.append(int_freq/1e9)
        int_fft_v.append(int_fft)

        # pad fft
        pad_fft = np.fft.fft(pad_v / 1e3) / Ndf
        pad_fft_f.append(freq/1e9)
        pad_fft_v.append(pad_fft)

        # save fft
        g4.create_dataset(f'Ch{ant_i}', data=np.stack([int_freq, int_fft], axis=-1), compression="gzip", compression_opts=9)
        g5.create_dataset(f'Ch{ant_i}', data=np.stack([freq, pad_fft], axis=-1), compression="gzip", compression_opts=9)

    hf.close()
    del g1, g2, g3, g4, g5, hf, usefulEvent
   
    print(f'output is {Output}{h5_file_name}')
    
    # raw wf plot
    plot_16_indi(r'Time [ $ns$ ]',r'Amplitude [ $mV$ ]',f'Raw WF, A{Station}, Run{Run}, Evt{Sel_evt}, \n Y{Year}, MD{MD}, Config:{Config}, Trig:{trig}, Qual:{quality}, Bad Ant:{bad_ant_label}'
            ,raw_wf_t, raw_wf_v
            ,Output,f'Raw_WF_A{Station}_R{Run}_E{Sel_evt}.png'
            ,f'Event# {Sel_evt} Raw WF plot was generated!'
            ,vpeak = 'on')

    # int wf plot
    plot_16_indi(r'Time [ $ns$ ]',r'Amplitude [ $mV$ ]',f'Int WF, A{Station}, Run{Run}, Evt{Sel_evt}, \n Y{Year}, MD{MD}, Config:{Config}, Trig:{trig}, Qual:{quality}, Bad Ant:{bad_ant_label}'
            ,int_wf_t, int_wf_v
            ,Output,f'Int_WF_A{Station}_R{Run}_E{Sel_evt}.png'
            ,f'Event# {Sel_evt} Int WF plot was generated!'
            ,vpeak = 'on')

    # pad wf plot
    plot_16_indi(r'Time [ $ns$ ]',r'Amplitude [ $mV$ ]',f'Pad WF, A{Station}, Run{Run}, Evt{Sel_evt}, \n Y{Year}, MD{MD}, Config:{Config}, Trig:{trig}, Qual:{quality}, Bad Ant:{bad_ant_label}'
            ,pad_wf_t, pad_wf_v
            ,Output,f'Pad_WF_A{Station}_R{Run}_E{Sel_evt}.png'
            ,f'Event# {Sel_evt} Pad WF plot was generated!'
            ,vpeak = 'on')

    # int fft plot
    plot_16_indi(r'Frequency [ $GHz$ ]',r'Amplitude [ $V/Hz$ ]',f'Int FFT, A{Station}, Run{Run}, Evt{Sel_evt}, \n Y{Year}, MD{MD}, Config:{Config}, Trig:{trig}, Qual:{quality}, Bad Ant:{bad_ant_label}'
            ,int_fft_f, int_fft_v
            ,Output,f'Int_FFT_A{Station}_R{Run}_E{Sel_evt}.png'
            ,f'Event# {Sel_evt} Int FFT plot was generated!'
            ,xmin = 0, xmax = 1
            ,y_scale = 'log'
            ,absol = 'abs')
    
    # pad fft plot
    plot_16_indi(r'Frequency [ $GHz$ ]',r'Amplitude [ $V/Hz$ ]',f'Pad FFT, A{Station}, Run{Run}, Evt{Sel_evt}, \n Y{Year}, MD{MD}, Config:{Config}, Trig:{trig}, Qual:{quality}, Bad Ant:{bad_ant_label}'
            ,pad_fft_f, pad_fft_v
            ,Output,f'Pad_FFT_A{Station}_R{Run}_E{Sel_evt}.png'
            ,f'Event# {Sel_evt} Pad FFT plot was generated!'
            ,xmin = 0, xmax = 1
            ,y_scale = 'log'
            ,absol = 'abs')

    del Output, h5_file_name
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) !=4 and len (sys.argv) !=5:
        Usage = """
    This is designed to output wf and fft of single event.  You have to choose specific run and event.
    It will be output 1) trig, qual, bad ant, antenna 2) raw wf 3) int wf 4) pad wf 5) int fft 6) pad fft
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Run2898_WF_FFT/>
    if you want choose specific event, 
        <Sel_evt ex) 10(Soft) 11(Cal) or 13(RF)>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Data=str(sys.argv[1])
    Ped=str(sys.argv[2])
    Output=str(sys.argv[3])
    if len (sys.argv) == 5:   
        sel = int(sys.argv[4])
        wf_fft_checker(Data, Ped, Output, Sel_evt = sel)
        del sel

    else:
        wf_fft_checker(Data, Ped, Output)

del curr_path













    
