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
from tools.fft import freq_pad_maker
from tools.temp import temp_loader
#from tools.mf import soft_psd_maker
#from tools.mf import soft_psd_maker_debug
#from tools.mf import evt_snr_maker
#from tools.mf import evt_snr_maker_debug
from tools.arr_table import table_loader
from tools.array import arr_2d

def ms_filter(Data, Ped, Station, Run, Output, DMode, CPath = curr_path):

    if DMode == 'normal':
        print('DMode is normal! Final result will be only event-wise SNR!')
    elif DMode == 'debug':
        print('DMode is debug! It will save all middle progress by h5 and png!')
    else:
        print('DMode is not set. Choose 1) normal or 2) debug')
        sys.exit(1)

    # import root and ara root lib
    ROOT = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, eventTree, rawEvent, num_events, calibrator, qual = ara_raw_to_qual(ROOT, Data, Ped, Station)
    del Data, Ped

    # known configuration. Probably can call from actual data file through the AraRoot in future....
    #antenna
    num_Antennas = ant.antenna_info()[2]
    # masked antenna
    bad_ant_index = ant.bad_antenna(Station)
    # interpolation time width
    time_width_ns, time_width_s =interpolation_bin_width()

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(time_width_ns)[1:]

    # make freq pad
    freq, freq_w = freq_pad_maker(time_pad_len, time_width_s)

    # template loader
    temp_vol, num_theta, theta_w = temp_loader(CPath, 'EM')[:-1]

    # array for 16 wf
    ant_arr = arr_2d(time_pad_len, num_Antennas, 0, float)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)

    # psd maker
    if DMode == 'normal':

        from tools.mf import soft_psd_maker

        soft_psd = soft_psd_maker(ant_arr, time_pad_len, num_Antennas, num_events
                                , ROOT, eventTree, rawEvent, calibrator, qual
                                , time_width_ns, time_pad_i, time_pad_f, time_width_s
                                , num_theta, freq, bad_ant_index)

    elif DMode == 'debug':
    
        # selected event
        soft_sel_evt = 679

        #summon time pad
        time_pad = time_pad_maker(time_width_ns)[0]
    
        from tools.mf import soft_psd_maker_debug

        soft_psd, soft_psd_wo_band, soft_evt_num, soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band, soft_indi_event = soft_psd_maker_debug(ant_arr, time_pad_len, num_Antennas, num_events
                                , ROOT, eventTree, rawEvent, calibrator, qual
                                , time_width_ns, time_pad_i, time_pad_f, time_width_s
                                , num_theta, freq, bad_ant_index
                                , Output, Station, Run, time_pad, soft_sel_evt)

    else:
        print('DMode is not set. Choose 1) normal or 2) debug')    
        sys.exit(1)  
    del time_width_s, time_pad_len

    # table_loader
    peak_w = 50
    mov_index, pad_t_len, pad_len_front, pad_len_end, ps_len_index = table_loader(CPath, Station, theta_w, peak_w)[:-2]

    # event-wise snr maker
    if DMode == 'normal':

        from tools.mf import evt_snr_maker

        evt_w_snr, evt_w_snr_v, evt_w_snr_h, evts_num, trig = evt_snr_maker(ant_arr, pad_t_len, num_Antennas, num_theta, mov_index, num_events 
                                                                            , ROOT, eventTree, rawEvent, calibrator, qual
                                                                            , time_width_ns, time_pad_i, time_pad_f, freq
                                                                            , freq_w, soft_psd, temp_vol, pad_len_front, pad_len_end, ps_len_index, bad_ant_index)
        del soft_psd

    elif DMode == 'debug':

        # selected event
        #sel_evt = 11
        sel_evt = 13

        #summon arr table
        mov_t, pad_t = table_loader(CPath, Station, theta_w, peak_w)[-2:]

        from tools.mf import evt_snr_maker_debug

        evt_w_snr, evt_w_snr_v, evt_w_snr_h, evts_num, trig, trig_index, event_copy, ant_arr_copy, ant_arr_fft, ant_arr_fft_band, snr_wf_copy, snr_wf_r_max_copy, v_match, h_match, v_sum_match, h_sum_match, v_sum_match_01, h_sum_match_01, v_avg_match, h_avg_match, evt_snr_v_sky_2d, evt_snr_h_sky_2d, nadir_range, phi_range = evt_snr_maker_debug(ant_arr, pad_t_len, num_Antennas, num_theta, mov_index, num_events
                                                                            , ROOT, eventTree, rawEvent, calibrator, qual
                                                                            , time_width_ns, time_pad_i, time_pad_f, freq
                                                                            , freq_w, soft_psd, temp_vol, pad_len_front, pad_len_end, ps_len_index, bad_ant_index
                                                                            , Output, Station, Run, time_pad, pad_t, mov_t, sel_evt, peak_w, theta_w)

    else:
        print('DMode is not set. Choose 1) normal or 2) debug')
        sys.exit(1)

    del CPath, ROOT, file, eventTree, rawEvent, num_events, calibrator, qual, num_Antennas, time_width_ns, time_pad_i, time_pad_f, freq_w, temp_vol, num_theta, ant_arr, mov_index, pad_t_len, pad_len_front, pad_len_end, ps_len_index 

    # create output file
    os.chdir(Output)
    if DMode == 'normal':

        h5_file_name='Evt-Wise_SNR_A'+str(Station)+'_R'+str(Run)+'.h5'
        hf = h5py.File(h5_file_name, 'w')

        #saving result
        hf.create_dataset('evt-w_snr', data=evt_w_snr, compression="gzip", compression_opts=9)
        hf.create_dataset('evt-w_snr_v', data=evt_w_snr_v, compression="gzip", compression_opts=9)
        hf.create_dataset('evt-w_snr_h', data=evt_w_snr_h, compression="gzip", compression_opts=9)
        hf.create_dataset('evts_num', data=evts_num, compression="gzip", compression_opts=9)
        hf.create_dataset('trig', data=trig, compression="gzip", compression_opts=9)

        hf.close()
        del hf

    elif DMode == 'debug':

        h5_file_name='Evt-Wise_SNR_A'+str(Station)+'_R'+str(Run)+'_debug.h5'
        hf = h5py.File(h5_file_name, 'w')

        #saving result
        g0 = hf.create_group('Station_info')
        g0.create_dataset('Bad_ant', data=np.array([bad_ant_index]), compression="gzip", compression_opts=9)
        del g0

        g1 = hf.create_group('PSD')
        g1.create_dataset('Freq', data=freq, compression="gzip", compression_opts=9)
        g1.create_dataset('Time', data=time_pad, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD', data=soft_psd_wo_band, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD_band', data=soft_psd, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_evt_num', data=np.array([soft_evt_num]), compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_WF_evt'+str(soft_indi_event), data=soft_wf, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_FFT_evt'+str(soft_indi_event), data=soft_fft, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_FFT_band_evt'+str(soft_indi_event), data=soft_fft_band, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD_evt'+str(soft_indi_event), data=soft_indi_psd, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD_band_evt'+str(soft_indi_event), data=soft_indi_psd_band, compression="gzip", compression_opts=9)
        del g1, soft_sel_evt, soft_psd, soft_psd_wo_band, soft_evt_num, soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band, soft_indi_event

        g2 = hf.create_group('Evt-wise_SNR')
        g2.create_dataset('Trig_index', data=np.array([trig_index]), compression="gzip", compression_opts=9)
        g2.create_dataset('Freq', data=freq, compression="gzip", compression_opts=9)
        g2.create_dataset('Time', data=time_pad, compression="gzip", compression_opts=9)
        g2.create_dataset('WF_evt'+str(event_copy), data=ant_arr_copy, compression="gzip", compression_opts=9)
        g2.create_dataset('FFT_evt'+str(event_copy), data=ant_arr_fft, compression="gzip", compression_opts=9)
        g2.create_dataset('FFT_band_evt'+str(event_copy), data=ant_arr_fft_band, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_time', data=pad_t, compression="gzip", compression_opts=9)
        g2.create_dataset('Peak_search_width', data=np.array([peak_w]), compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_evt'+str(event_copy), data=snr_wf_copy, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_roll_max_evt'+str(event_copy), data=snr_wf_r_max_copy, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_time', data=mov_t, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_evt'+str(event_copy)+'Vpol', data=v_match, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_evt'+str(event_copy)+'Hpol', data=h_match, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_sum_evt'+str(event_copy)+'Vpol', data=v_sum_match, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_sum_evt'+str(event_copy)+'Hpol', data=h_sum_match, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_sum_count_evt'+str(event_copy)+'Vpol', data=v_sum_match_01, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_sum_count_evt'+str(event_copy)+'Hpol', data=h_sum_match_01, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_avg_evt'+str(event_copy)+'Vpol', data=v_avg_match, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_roll_max_avg_evt'+str(event_copy)+'Hpol', data=h_avg_match, compression="gzip", compression_opts=9)
        g2.create_dataset('Nadir_range', data=nadir_range, compression="gzip", compression_opts=9)
        g2.create_dataset('Phi_range', data=phi_range, compression="gzip", compression_opts=9)
        g2.create_dataset('Grid_width', data=np.array([theta_w]), compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_2d_evt'+str(event_copy)+'Vpol', data=evt_snr_v_sky_2d, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_2d_evt'+str(event_copy)+'Hpol', data=evt_snr_h_sky_2d, compression="gzip", compression_opts=9)
        del g2, sel_evt, mov_t, pad_t,time_pad, trig_index, event_copy, ant_arr_copy, ant_arr_fft, ant_arr_fft_band, snr_wf_copy, snr_wf_r_max_copy, v_match, h_match, v_sum_match, h_sum_match, v_sum_match_01, h_sum_match_01, v_avg_match, h_avg_match, evt_snr_v_sky_2d, evt_snr_h_sky_2d, nadir_range, phi_range

        gf = hf.create_group('Event-wise_SNR')
        gf.create_dataset('evt-w_snr', data=evt_w_snr, compression="gzip", compression_opts=9)
        gf.create_dataset('evt-w_snr_v', data=evt_w_snr_v, compression="gzip", compression_opts=9)
        gf.create_dataset('evt-w_snr_h', data=evt_w_snr_h, compression="gzip", compression_opts=9)
        gf.create_dataset('evts_num', data=evts_num, compression="gzip", compression_opts=9)
        gf.create_dataset('trig', data=trig, compression="gzip", compression_opts=9)
        del gf

        hf.close()
        del hf

    else:
        print('DMode is not set. Choose 1) normal or 2) debug')
        sys.exit(1)
    del DMode, Station, Run, evt_w_snr, evt_w_snr_v, evt_w_snr_h, evts_num, trig, freq, bad_ant_index, peak_w, theta_w

    print('output is',Output+h5_file_name)
    del Output, h5_file_name
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) !=7:
        Usage = """
    This is designed to analyze all events in the run. You have to choose specific run.
    Depending on DMode, It will save just event-wise SNR or all middle step information.
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Station ex)2>
    <Run ex)2898>
    <Output path ex)/data/user/mkim/OMF_sky/ARA02/>
    <DMode ex) normal or debug>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Data=str(sys.argv[1])
    Ped=str(sys.argv[2])
    Station=int(sys.argv[3])
    Run=int(sys.argv[4])
    Output=str(sys.argv[5])
    DMode=str(sys.argv[6])

    ms_filter(Data, Ped, Station, Run, Output, DMode, CPath = curr_path+'/..')

del curr_path













    
