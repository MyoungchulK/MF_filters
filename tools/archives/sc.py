import numpy as np
import h5py

from tools.fft import vsqphz_maker
from tools.fft import db_sq_lin_maker
from tools.fft import db_log_maker

def ntot_loader(c_path, Station, oneside = True, dbmphz_scale = True, real_fft = True, fft_norm = True):

    # load ntot file
    dat_path = f'{c_path}/data/'
    if real_fft == True:
        if fft_norm == True:
            ntot_name = f'A{Station}_Ntot_Lab_real_norm.txt'
        else:
            ntot_name = f'A{Station}_Ntot_Lab_real.txt'
    else:
        if fft_norm == True:
            ntot_name = f'A{Station}_Ntot_Lab_norm.txt'
        else:
            ntot_name = f'A{Station}_Ntot_Lab.txt'
    print('Ntot file:',ntot_name)
    ntot_file = np.loadtxt(dat_path + ntot_name)
    ntot = ntot_file[:,1:]

    if dbmphz_scale == True:
        pass
    else:
        ntot = vsqphz_maker(ntot) 

    if oneside == True:
        pass
    else:
        ntot = np.append(ntot,ntot[::-1],axis=0)

    del dat_path, ntot_name, ntot_file

    return ntot

def sc_loader(Station, Run, oneside = True, db_scale = True):

    # load rayleigh file
    sc_path =f'/data/user/mkim/OMF_filter/ARA0{Station}/Rayl/'
    sc_name = f'Rayleigh_Fit_A{Station}_R{Run}.h5'
    sc_file = h5py.File(sc_path + sc_name, 'r')
    sc_arr = sc_file['sc'][:]

    if db_scale == True:
        sc_arr = db_log_maker(sc_arr**2)   
    else:
        pass

    if oneside == False:
        sc_arr = np.append(sc_arr,sc_arr[::-1],axis=0)
    else:
        pass

    del sc_path, sc_name, sc_file

    print('Dynamic Signal Chain loading is done!')

    return sc_arr

def sc_maker(psd, c_path, Station, db_scale = False, rfft = False, fft_norm = False):#, oneside = True):

    # load ntot file
    ntot = ntot_loader(c_path, Station, real_fft = rfft, fft_norm = fft_norm)

    # make sc
    sc = psd - ntot
    del ntot

    if db_scale == True:
        pass
    else:
        sc = db_sq_lin_maker(sc)

    print('In-situ signal chain making is done!')

    return sc
    
