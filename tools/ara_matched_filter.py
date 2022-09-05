import os
import numpy as np
import h5py
from scipy.signal import butter, filtfilt, hilbert, fftconvolve, correlation_lags
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

class ara_matched_filter:

    def __init__(self, st, config, year, dt, wf_len, bad_ant):

        self.st = st
        self.config = config
        self.year = year
        self.dt = dt
        self.bad_ant = bad_ant
        self.wf_len = wf_len
        self.lag = correlation_lags(self.wf_len, self.wf_len, 'full') * self.dt
        self.lag_len = len(self.lag)
        print(self.lag_len)
        self.half_pad = 500
        self.sum_lag_pad = self.lag_len + self.half_pad * 2
        self.wf_freq = np.fft.fftfreq(self.wf_len, self.dt)


        if self.st == 2:
            if self.config < 4:
                self.true_wf_len = int(22*20/0.5)
            elif self.config > 3 and self.config < 6:
                self.true_wf_len = int(26*20/0.5)
            else:
                self.true_wf_len = int(28*20/0.5)
        if self.st == 3:
            if self.config < 3:
                self.true_wf_len = int(22*20/0.5)
            elif self.config > 2 and self.config < 5:
                self.true_wf_len = int(26*20/0.5)
            elif self.config == 5:
                self.true_wf_len = int(22*20/0.5)
            else:
                self.true_wf_len = int(28*20/0.5) 
        print(self.true_wf_len)

        self.theta_width = 30
        self.theta = np.arange(30, 150 + 1, self.theta_width, dtype = int)
        self.theta_len = len(self.theta)
        self.phi_width = 60
        self.phi = np.arange(30, 330 + 1, self.phi_width, dtype = int)
        self.phi_len = len(self.phi)
        self.theta_idx = np.array([2, 1, 0, 1, 2], dtype = int)

        self.ant_res_width = np.copy(self.theta_width)
        self.ant_res = np.arange(0, -60 - 1, -self.ant_res_width, dtype = int)
        self.ant_res_len = len(self.ant_res)
        self.off_cone_width = 2
        self.off_cone = np.arange(0, 4+1, self.off_cone_width, dtype = int)
        self.off_cone_len = len(self.off_cone)

        self.roll_win = int(100/self.dt) + 1
        self.em_had_thres = int((num_ants - np.count_nonzero(self.bad_ant)) * self.ant_res_len * self.off_cone_len / 2)
        self.num_sols = 2
        self.num_fin_temps = self.theta_len * self.phi_len * self.num_sols

    def get_temp_wfs(self):
    
        t_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/temp_sim/temp_AraOut.A{self.st}_C{self.config}_temp_rayl.txt.run0.h5'
        print('temp_path', t_path)
        hf_t = h5py.File(t_path, 'r')
        off_len = len(hf_t['off_cone'][:])
        test_temp0 = hf_t['temp'][:]
        temp_len = len(test_temp0[:,0,0,0,0])
        test_temp1 = np.full((temp_len, num_ants, self.ant_res_len, off_len, self.num_sols), np.nan, dtype = float)
        test_temp1[:,:,0] = test_temp0[:, :, int(self.ant_res[0]/10)]
        test_temp1[:,:,1] = test_temp0[:, :, int(-self.ant_res[1]/10)]
        test_temp1[:,:,2] = test_temp0[:, :, int(-self.ant_res[2]/10)]
        test_temp = np.full((temp_len, num_ants, self.ant_res_len, self.off_cone_len, self.num_sols), np.nan, dtype = float)
        test_temp[:,:,:,0] = test_temp1[:,:,:,int(self.off_cone[0]/self.dt)]
        test_temp[:,:,:,1] = test_temp1[:,:,:,int(self.off_cone[1]/self.dt)]
        test_temp[:,:,:,2] = test_temp1[:,:,:,int(self.off_cone[2]/self.dt)]
        
        nu, de = butter(10, [0.13, 0.85], btype = 'band', fs = 1 / self.dt)
        temp = filtfilt(nu, de, test_temp, axis = 0)
        print(temp.shape)
        if self.wf_len != temp_len:
            temp = np.pad(temp, [(0, self.wf_len - len(temp[:,0,0,0,0])), (0, 0), (0, 0), (0, 0), (0, 0)], 'constant', constant_values = 0)
        del t_path, hf_t, test_temp0, test_temp1, test_temp, nu, de, off_len
        print(temp.shape)

        return temp

    def get_psd(self, d_path, corr_fac = None):

        print('psd_path', d_path)
        hf_n = h5py.File(d_path, 'r')
        try:
            rayl = np.nansum(hf_n['rayl'][:], axis = 0)
            if corr_fac is not None:
                rayl *= corr_fac
            psd = (rayl / np.sqrt(self.dt / self.wf_len))**2
        except KeyError:
            rayl = np.nansum(hf_n['soft_rayl'][:], axis = 0)
            print(rayl.shape)
            rayl = np.append(rayl[:-1], rayl[1:][::-1], axis = 0)
            if corr_fac is not None:
                rayl *= corr_fac
            psd = (rayl / np.sqrt(self.dt / self.true_wf_len))**2
        del hf_n, rayl

        return psd

    def get_arr_table(self):

        a_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/arr_time_table/arr_time_table_A{self.st}_Y{self.year}.h5'
        print('arr_table_path', a_path) 
        hf_a = h5py.File(a_path, 'r')
        arr_time_table = hf_a['arr_time_table'][:]
        arr_delay_table = arr_time_table - np.nanmean(arr_time_table, axis = 3)[:, :, :, np.newaxis]
        arr_delay_table = arr_delay_table[:,:,1]
        arr_del_idx = np.full((self.theta_len, self.phi_len, num_ants, self.num_sols), np.nan, dtype = float)
        for t in range(self.theta_len):
            for p in range(self.phi_len):
                arr_del_idx[t, p] = arr_delay_table[self.theta[t], self.phi[p]] 
        self.arr_nan = np.isnan(arr_del_idx)
        arr_del_idx = (arr_del_idx / self.dt).astype(int) + self.half_pad 
        print(arr_del_idx.shape)
        del a_path, hf_a, arr_time_table, arr_delay_table           

        return arr_del_idx 

    def get_template(self, d_path, corr_fac = None):

        self.temp = self.get_temp_wfs()
        self.psd = self.get_psd(d_path, corr_fac = corr_fac)
        self.arr_del_idx = self.get_arr_table() 

        # normalization factor
        temp_fft = np.fft.fft(self.temp, axis = 0)
        nor_fac = np.abs(temp_fft)**2 / self.psd[:, :, np.newaxis, np.newaxis, np.newaxis]
        nor_fac = np.sqrt(np.nansum(nor_fac, axis = 0) / (self.dt * self.wf_len))
        self.temp /= nor_fac[np.newaxis, :, :, :, :]
        del nor_fac

    def get_band_pass_filter(self, amp, val = 1e-100): # for temp, lets use brutal method.... for now....

        #notch filter
        amp[(self.wf_freq >= 0.43) & (self.wf_freq <= 0.48)] *= val
        amp[(self.wf_freq <= -0.43) & (self.wf_freq >= -0.48)] *= val

        # front/back band
        amp[(self.wf_freq >= -0.14) & (self.wf_freq <= 0.14)] *= val
        amp[(self.wf_freq >= 0.75) | (self.wf_freq <= -0.75)] *= val

        return amp

    def get_mf_wfs(self, wf_v):

        # fft and deconvolve psd
        wf_fft = np.fft.fft(wf_v, axis = 0) / self.psd
        wf_fft = self.get_band_pass_filter(wf_fft)        
        wf_v_w = np.real(np.fft.ifft(wf_fft, axis = 0))
        del wf_fft

        # correlation
        corr = fftconvolve(self.temp, wf_v_w[:, :, np.newaxis, np.newaxis, np.newaxis], 'full', axes = 0)
        corr = np.abs(hilbert(corr, axis = 0))
        del wf_v_w

        return corr

    def get_evt_wise_corr(self, corr, use_max = False):

        # select max corr
        corr_max = np.nanmax(corr, axis = 0)
        em_had_sel = int(np.nansum(np.argmax(corr_max, axis = 3)) > self.em_had_thres) # 1st choose em or had
        corr_off_max = np.argmax(corr_max[:, :, :, em_had_sel], axis = 2) # 2nd choose max off-cone

        corr_sort = np.full((self.lag_len, num_ants, self.ant_res_len), np.nan, dtype = float)
        for ant in range(num_ants):
            for res in range(self.ant_res_len):
                corr_sort[:, ant, res] = corr[:, ant, res, corr_off_max[ant, res], em_had_sel]
        del em_had_sel, corr_off_max, corr_max

        # rolling max
        corr_sort_roll_max = maximum_filter1d(corr_sort, axis = 0, size = self.roll_win, mode='constant')
        del corr_sort

        # sum up by arrival time delay
        evt_wize_corr = np.full((2, self.sum_lag_pad, self.num_fin_temps), 0, dtype = float)
        evt_wize_corr_max_ant = np.full((num_ants, self.num_fin_temps), np.nan, dtype = float)
        count = 0 
        for t in range(self.theta_len):
            for p in range(self.phi_len):
                for s in range(self.num_sols):
                    for ant in range(num_ants):
                        if self.arr_nan[t, p, ant, s]:
                            continue
                        if self.bad_ant[ant]:
                            continue
                        pol_idx = int(ant > 7)
                        arr_idx = self.arr_del_idx[t, p, ant, s]
                        corr_ch = corr_sort_roll_max[:, ant, self.theta_idx[t]]
                        evt_wize_corr[pol_idx, arr_idx:arr_idx + self.lag_len, count] += corr_ch
                        evt_wize_corr_max_ant[ant, count] = np.nanmax(corr_ch)
                        del arr_idx, pol_idx, corr_ch
                    count += 1
        del corr_sort_roll_max

        evt_wize_corr_max = np.nanmax(evt_wize_corr, axis = (1,2))
        evt_wize_corr_max_ant = evt_wize_corr_max_ant[:, np.where(evt_wize_corr == np.nanmax(evt_wize_corr_max))[2][0]]
        del evt_wize_corr

        return evt_wize_corr_max, evt_wize_corr_max_ant 

    def get_evt_wise_snr(self, wf_v, snr = None):

        corr = self.get_mf_wfs(wf_v) 

        if snr is not None:
            corr *= snr[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        corr[:, self.bad_ant] = np.nan

        evt_wize_corr, evt_wize_corr_max_ant = self.get_evt_wise_corr(corr, use_max = True)
        del corr

        return evt_wize_corr, evt_wize_corr_max_ant














