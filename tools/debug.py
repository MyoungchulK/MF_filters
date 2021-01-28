import numpy as np
from tools.plot import plot_16
from tools.plot import plot_16_log_theta
from tools.plot import plot_16_3
from tools.fft import psd_maker
from tools.plot import plot_16_overlap
from tools.plot import plot_1
from tools.plot import sky_map
from tools.plot import hist_map
from tools.array import arr_4d

def Band_Square_debug(freq,amp):

    band_amp = np.copy(amp)

    #Bothpol
    band_amp[(freq>=430e6) & (freq<=480e6)]=1e-50
    band_amp[(freq<=-430e6) & (freq>=-480e6)]=1e-50

    #Vpol
    band_amp[(freq>=-150e6) & (freq<=150e6),:8]=1e-50
    band_amp[(freq>=680e6) | (freq<=-680e6),:8,1]=1e-50
    band_amp[(freq>=665e6) | (freq<=-665e6),:8,2]=1e-50

    #Hpol
    band_amp[(freq>=-217e6) & (freq<=217e6),8:]=1e-50
    band_amp[(freq>=630e6) | (freq<=-630e6),8:,1]=1e-50
    band_amp[(freq>=530e6) | (freq<=-530e6),8:,2]=1e-50

    #Both theta
    band_amp[(freq>=680e6) | (freq<=-680e6),:,0]=1e-50

    return(band_amp)

def psd_indi_debug(Station, Run, Output, event
                    , t_pad_len, time_pad_i, time_pad_f, t_width_ns, ndf
                    , f
                    , n_theta
                    , ant_arr, int_time_len):

    # wf plot
    ant_arr_copy = np.copy(ant_arr)

    plot_16(r'Time [ $ns$ ]',r'Amplitude [ $V$ ]',f'Soft WF, A{Station}, Run{Run}, Evt{event}'
            ,np.arange(time_pad_i, time_pad_f+t_width_ns, t_width_ns),ant_arr_copy
            ,np.round(np.nanmax(np.abs(ant_arr_copy),axis=0),2)
            ,-200,200
            ,Output,f'Soft_WF_A{Station}_Run{Run}_Evt{event}.png'
            ,f'Event# {event} Soft WF plot was generated!')

    # fft plot
    ant_arr_fft = np.fft.fft(ant_arr_copy, axis=0) / ndf
    ant_arr_fft_band = Band_Square_debug(f, np.repeat(ant_arr_fft[:,:,np.newaxis], n_theta, axis=2))

    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Amplitude [ $V$ ]', f'Soft FFT, A{Station}, Run{Run}, Evt{event}'
            ,f/1e9,ant_arr_fft
            ,f/1e9,ant_arr_fft_band[:,:,0]
            ,f/1e9,ant_arr_fft_band[:,:,1]
            ,f/1e9,ant_arr_fft_band[:,:,2]
            ,1e-13,1e-7
            ,Output,f'Soft_FFT_A{Station}_Run{Run}_Evt{event}.png'
            ,f'Event# {event} Soft FFT plot was generated!')

    # psd indi plot
    indi_psd = psd_maker(ant_arr_copy, ndf, t_pad_len, int_time_len)
    indi_psd_band = Band_Square_debug(f, np.repeat(indi_psd[:,:,np.newaxis], n_theta, axis=2))

    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Power [ $V^2$ ]', f'Soft PSD, A{Station}, Run{Run}, Evt{event}'
            ,f/1e9,indi_psd
            ,f/1e9,indi_psd_band[:,:,0]
            ,f/1e9,indi_psd_band[:,:,1]
            ,f/1e9,indi_psd_band[:,:,2]
            ,1e-17,1e-9
            ,Output,f'Soft_PSD_A{Station}_Run{Run}_Evt{event}.png'
            ,f'Event# {event} Soft PSD indi plot was generated!')

    return ant_arr_copy, ant_arr_fft, ant_arr_fft_band, indi_psd, indi_psd_band

def psd_debug(Station, Run, Output, event
                , f
                , n_theta
                , psd, num_psd):

    # psd plot
    psd_copy = np.copy(psd)
    psd_band = Band_Square_debug(f, np.repeat(psd_copy[:,:,np.newaxis], n_theta, axis=2))

    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Power [ $V^2$ ]', f'Soft Avg PSD, A{Station}, Run{Run}, Avg of {num_psd}Evts'
            ,f/1e9,psd_copy
            ,f/1e9,psd_band[:,:,0]
            ,f/1e9,psd_band[:,:,1]
            ,f/1e9,psd_band[:,:,2]
            ,1e-17,1e-9
            ,Output,f'Soft_Avg_PSD_A{Station}_Run{Run}.png'
            ,'Soft avg PSD plot was generated!')
    del psd_band

    return psd_copy

def evt_snr_indi_debug_0(Station, Run, Output, event, trig_type
                    , time_pad_i, time_pad_f, t_width_ns, ndf
                    , f
                    , n_theta, pad_t
                    , ant_arr, snr_wf, snr_wf_nor):

    # wf plot
    ant_arr_copy = np.copy(ant_arr)

    plot_16(r'Time [ $ns$ ]',r'Amplitude [ $V$ ]',f'{trig_type} WF, A{Station}, Run{Run}, Evt{event}'
            ,np.arange(time_pad_i, time_pad_f+t_width_ns, t_width_ns),ant_arr_copy
            ,np.round(np.nanmax(np.abs(ant_arr_copy),axis=0),2)
            ,time_pad_i,time_pad_f
            ,Output,f'{trig_type}_WF_A{Station}_Run{Run}_Evt{event}.png'
            ,f'Event# {event} {trig_type} WF plot was generated!')

    # fft plot
    ant_arr_fft = np.fft.fft(ant_arr_copy, axis=0) / ndf
    ant_arr_fft_band = Band_Square_debug(f, np.repeat(ant_arr_fft[:,:,np.newaxis], n_theta, axis=2))

    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Amplitude [ $V$ ]', f'{trig_type} FFT, A{Station}, Run{Run}, Evt{event}'
            ,f/1e9,ant_arr_fft
            ,f/1e9,ant_arr_fft_band[:,:,0]
            ,f/1e9,ant_arr_fft_band[:,:,1]
            ,f/1e9,ant_arr_fft_band[:,:,2]
            ,1e-13,1e-7
            ,Output,f'{trig_type}_FFT_A{Station}_Run{Run}_Evt{event}.png'
            ,f'Event# {event} {trig_type} FFT plot was generated!')

    #snr plot
    snr_wf_copy = np.sqrt(np.copy(snr_wf)**2 / np.copy(snr_wf_nor))

    plot_16_3(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', f'{trig_type} SNR, A{Station}, Run{Run}, Evt{event}, w/ tale'
            ,pad_t,snr_wf_copy[:,:,0],np.round(np.nanmax(snr_wf_copy[:,:,0],axis=0),2)
            ,pad_t,snr_wf_copy[:,:,1],np.round(np.nanmax(snr_wf_copy[:,:,1],axis=0),2)
            ,pad_t,snr_wf_copy[:,:,2],np.round(np.nanmax(snr_wf_copy[:,:,2],axis=0),2)
            ,Output,f'{trig_type}_SNR_A{Station}_Run{Run}_Evt{event}_w_tale.png'
            ,f'Event# {event} {trig_type} SNR w/ tale plot was generated!')

    return ant_arr_copy, ant_arr_fft, ant_arr_fft_band, snr_wf_copy 

def evt_snr_indi_debug_1(Station, Run, Output, event, trig_type
                    , pad_t
                    , snr_wf, snr_wf_nor):

    #snr plot
    snr_wf_copy_1 = np.sqrt(np.copy(snr_wf)**2 / np.copy(snr_wf_nor))
  
    peak = np.nanmax(np.abs(snr_wf[:,:8]),axis=0)
    peak_sum = np.nansum(peak,axis=0)**2
    nor = np.nanmax(snr_wf_nor[:,:8],axis=0)
    nor_sum = np.nansum(nor,axis=0)
    v_expected = np.round(np.copy(np.sqrt(peak_sum/nor_sum)),2)
            
    peak = np.nanmax(np.abs(snr_wf[:,8:]),axis=0)
    peak_sum = np.nansum(peak,axis=0)**2
    nor = np.nanmax(snr_wf_nor[:,8:],axis=0)
    nor_sum = np.nansum(nor,axis=0)
    h_expected = np.round(np.copy(np.sqrt(peak_sum/nor_sum)),2)
    del peak, peak_sum, nor, nor_sum

    plot_16_3(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', f'{trig_type} SNR, A{Station}, Run{Run}, Evt{event}, w/o tale, \n , Vex:{v_expected[0]},{v_expected[1]},{v_expected[2]}, Hex:{h_expected[0]},{h_expected[1]},{h_expected[2]}'
            ,pad_t,snr_wf_copy_1[:,:,0],np.round(np.nanmax(snr_wf_copy_1[:,:,0],axis=0),2)
            ,pad_t,snr_wf_copy_1[:,:,1],np.round(np.nanmax(snr_wf_copy_1[:,:,1],axis=0),2)
            ,pad_t,snr_wf_copy_1[:,:,2],np.round(np.nanmax(snr_wf_copy_1[:,:,2],axis=0),2)
            ,Output,f'{trig_type}_SNR_A{Station}_Run{Run}_Evt{event}_wo_tale.png'
            ,f'Event# {event} {trig_type} SNR w/o tale plot was generated!')
    del v_expected, h_expected

    return snr_wf_copy_1

def evt_snr_indi_debug_2(Station, Run, Output, event, trig_type
                    , pad_t, peak_w
                    , snr_wf, snr_wf_nor):

    #snr roll max plot
    snr_wf_r_max_copy = np.sqrt(np.copy(snr_wf)**2 / np.copy(snr_wf_nor))

    plot_16_3(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', f'{trig_type} Roll Max SNR(pw:{peak_w}ns), A{Station}, Run{Run}, Evt{event}'
            ,pad_t,snr_wf_r_max_copy[:,:,0],np.round(np.nanmax(snr_wf_r_max_copy[:,:,0],axis=0),2)
            ,pad_t,snr_wf_r_max_copy[:,:,1],np.round(np.nanmax(snr_wf_r_max_copy[:,:,1],axis=0),2)
            ,pad_t,snr_wf_r_max_copy[:,:,2],np.round(np.nanmax(snr_wf_r_max_copy[:,:,2],axis=0),2)
            ,Output,f'{trig_type}_Roll_Max_SNR_pw{peak_w}_A{Station}_Run{Run}_Evt{event}.png'
            ,f'Event# {event} {trig_type} Roll Max SNR plot was generated!')

    return snr_wf_r_max_copy

def evt_snr_indi_debug_3(Station, Run, Output, event, trig_type
                    , num_Ants, half_ant                
                    , mov_i, mov_t, theta_w, peak_w
                    , snr_wf, snr_wf_nor):

    #event-wise plot
    snr_wf_2d_ex = arr_4d(mov_i.shape[0], mov_i.shape[1], mov_i.shape[2], mov_i.shape[3], np.nan, float)
    snr_wf_2d_01_ex = np.copy(snr_wf_2d_ex)

    for n_ant in range(num_Ants):
        snr_wf_2d_ex[:,n_ant,2] = snr_wf[:,n_ant,0][mov_i[:,n_ant,2,:]]
        snr_wf_2d_ex[:,n_ant,3] = snr_wf[:,n_ant,1][mov_i[:,n_ant,3,:]]
        snr_wf_2d_ex[:,n_ant,1] = snr_wf[:,n_ant,1][mov_i[:,n_ant,1,:]]
        snr_wf_2d_ex[:,n_ant,0] = snr_wf[:,n_ant,2][mov_i[:,n_ant,0,:]]
        snr_wf_2d_ex[:,n_ant,4] = snr_wf[:,n_ant,2][mov_i[:,n_ant,4,:]]

        snr_wf_2d_01_ex[:,n_ant,2] = snr_wf_nor[:,n_ant,0][mov_i[:,n_ant,2,:]]
        snr_wf_2d_01_ex[:,n_ant,3] = snr_wf_nor[:,n_ant,1][mov_i[:,n_ant,3,:]]
        snr_wf_2d_01_ex[:,n_ant,1] = snr_wf_nor[:,n_ant,1][mov_i[:,n_ant,1,:]]
        snr_wf_2d_01_ex[:,n_ant,0] = snr_wf_nor[:,n_ant,2][mov_i[:,n_ant,0,:]]
        snr_wf_2d_01_ex[:,n_ant,4] = snr_wf_nor[:,n_ant,2][mov_i[:,n_ant,4,:]]

    v_sum = np.nansum(snr_wf_2d_ex[:,:half_ant], axis=1)**2
    v_sum_01 = np.nansum(snr_wf_2d_01_ex[:,:half_ant], axis=1)
    h_sum = np.nansum(snr_wf_2d_ex[:,half_ant:], axis=1)**2
    h_sum_01 = np.nansum(snr_wf_2d_01_ex[:,half_ant:], axis=1)

    v_avg = np.sqrt(v_sum / v_sum_01)
    h_avg = np.sqrt(h_sum / h_sum_01)

    evt_snr_v_sky = np.nanmax(v_avg)
    evt_snr_h_sky = np.nanmax(h_avg)

    evt_snr_v_loc = np.where(v_avg == evt_snr_v_sky)
    evt_snr_h_loc = np.where(h_avg == evt_snr_h_sky)
    del evt_snr_v_sky, evt_snr_h_sky

    evt_snr_v_sky_2d = np.nanmax(v_avg,axis=0)
    evt_snr_h_sky_2d = np.nanmax(h_avg,axis=0)

    v_match = np.sqrt(snr_wf_2d_ex[:,:half_ant,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]**2 / snr_wf_2d_01_ex[:,:half_ant,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]])
    h_match = np.sqrt(snr_wf_2d_ex[:,half_ant:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]**2 / snr_wf_2d_01_ex[:,half_ant:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]])
    del snr_wf_2d_ex, snr_wf_2d_01_ex

    v_sum_match = v_sum[:,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
    h_sum_match = h_sum[:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
    del v_sum, h_sum

    v_sum_match_01 = v_sum_01[:,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
    h_sum_match_01 = h_sum_01[:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
    del v_sum_01, h_sum_01

    v_avg_match = v_avg[:,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
    h_avg_match = h_avg[:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
    del v_avg, h_avg

    v_nadir = theta_w/2 + evt_snr_v_loc[1][0] * theta_w
    v_phi = theta_w/2 + evt_snr_v_loc[2][0] * theta_w
    h_nadir = theta_w/2 + evt_snr_h_loc[1][0] * theta_w
    h_phi = theta_w/2 + evt_snr_h_loc[2][0] * theta_w
    v_opt_angle = np.array([v_nadir,v_phi])
    h_opt_angle = np.array([h_nadir,h_phi])
    del evt_snr_v_loc, evt_snr_h_loc

    nadir_range = np.arange(0+theta_w/2, 180, theta_w)
    phi_range = np.arange(0+theta_w/2, 360, theta_w)

    plot_16_overlap(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', f'{trig_type} Roll Max SNR(pw:{peak_w}ns) w/o ArrT, N:{v_nadir}deg, P:{v_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Vpol'
            ,mov_t,v_match
            ,Output,f'{trig_type}_Roll_Max_SNR_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Vpol_N{v_nadir}_P{v_phi}.png'
            ,'Vpol'
            ,f'Event# {event} {trig_type} Roll Max SNR plot w/o ArrT Vpol was generated!')

    plot_16_overlap(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', f'{trig_type} Roll Max SNR(pw:{peak_w}ns) w/o ArrT, N:{h_nadir}deg, P:{h_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Hpol'
            ,mov_t,h_match
            ,Output,f'{trig_type}_Roll_Max_SNR_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Hpol_N{h_nadir}_P{h_phi}.png'
            ,'Hpol'
            ,f'Event# {event} {trig_type} Roll Max SNR plot w/o ArrT Hpol was generated!')

    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', f'{trig_type} Roll Max Sum SNR(pw:{peak_w}ns) w/o ArrT, N:{v_nadir}deg, P:{v_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Vpol'
            ,mov_t, v_sum_match
            ,Output,f'{trig_type}_Roll_Max_Sum_SNR_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Vpol_N{v_nadir}_P{v_phi}.png'
            ,f'Event# {event} {trig_type} Roll Max Sum SNR plot w/o ArrT Vpol was generated!')

    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', f'{trig_type} Roll Max Sum SNR(pw:{peak_w}ns) w/o ArrT, N:{h_nadir}deg, P:{h_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Hpol'
            ,mov_t, h_sum_match
            ,Output,f'{trig_type}_Roll_Max_Sum_SNR_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Hpol_N{h_nadir}_P{h_phi}.png'
            ,f'Event# {event} {trig_type} Roll Max Sum SNR plot w/o ArrT Hpol was generated!')

    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', f'{trig_type} Roll Max Sum SNR(pw:{peak_w}ns) Counts w/o ArrT, N:{v_nadir}deg, P:{v_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Vpol'
            ,mov_t, v_sum_match_01
            ,Output,f'{trig_type}_Roll_Max_Sum_SNR_Counts_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Vpol_N{v_nadir}_P{v_phi}.png'
            ,f'Event# {event} {trig_type} Roll Max Sum SNR Counts plot w/o ArrT Vpol was generated!')

    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', f'{trig_type} Roll Max Sum SNR(pw:{peak_w}ns) Counts w/o ArrT, N:{h_nadir}deg, P:{h_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Hpol'
            ,mov_t, h_sum_match_01
            ,Output,f'{trig_type}_Roll_Max_Sum_SNR_Counts_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Hpol_N{h_nadir}_P{h_phi}.png'
            ,f'Event# {event} {trig_type} Roll Max Sum SNR Counts plot w/o ArrT Hpol was generated!')

    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', f'{trig_type} Roll Max Avg SNR(pw:{peak_w}ns) w/o ArrT, N:{v_nadir}deg, P:{v_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Vpol'
            ,mov_t, v_avg_match
            ,Output,f'{trig_type}_Roll_Max_Avg_SNR_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Vpol_N{v_nadir}_P{v_phi}.png'
            ,f'Event# {event} {trig_type} Roll Max Avg SNR plot w/o ArrT Vpol was generated!')

    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', f'{trig_type} Roll Max Avg SNR(pw:{peak_w}ns) w/o ArrT, N:{h_nadir}deg, P:{h_phi}deg, \n A{Station}, Run{Run}, Evt{event}, Hpol'
            ,mov_t, h_avg_match
            ,Output,f'{trig_type}_Roll_Max_Avg_SNR_pw{peak_w}_wo_ArrT_A{Station}_Run{Run}_Evt{event}_Hpol_N{h_nadir}_P{h_phi}.png'
            ,f'Event# {event} {trig_type} Roll Max Avg SNR plot w/o ArrT Hpol was generated!')

    sky_map(evt_snr_v_sky_2d,nadir_range, phi_range, theta_w
            , f'{trig_type}, Evt-Wise SNR(pw:{peak_w}ns) 2d sky, A{Station}, Run{Run}, Evt{event}, Vpol'
            ,Output, f'{trig_type}_Evt_SNR_pw{peak_w}_2d_sky_A{Station}_Run{Run}_Evt{event}_Vpol.png'
            ,f'Event# {event} {trig_type} Event-wise SNR 2d sky map Vpol was generated!')

    sky_map(evt_snr_h_sky_2d,nadir_range, phi_range, theta_w
            , f'{trig_type}, Evt-Wise SNR(pw:{peak_w}ns) 2d sky, A{Station}, Run{Run}, Evt{event}, Hpol'
            ,Output, f'{trig_type}_Evt_SNR_pw{peak_w}_2d_sky_A{Station}_Run{Run}_Evt{event}_Hpol.png'
            ,f'Event# {event} {trig_type} Event-wise SNR 2d sky map Hpol was generated!')

    return v_match, h_match, v_sum_match, h_sum_match, v_sum_match_01, h_sum_match_01, v_avg_match, h_avg_match, evt_snr_v_sky_2d, evt_snr_h_sky_2d, nadir_range, phi_range, v_opt_angle, h_opt_angle 

def evt_snr_debug(Station, Run, Output
        , trigger, evt_snr_v, evt_snr_h):

    # hist map
    hist_map(r'Averaged & event-wise SNRs [ $V/RMS$ ]',r'# of events', f'Evt-wise SNR Hist., A{Station}, Run{Run}'
                ,trigger,evt_snr_v,evt_snr_h
                ,Output,f'Evt_SNR_hist_A{Station}_Run{Run}.png'
                ,'Event-wisc SNR histogram was generated!')










