import numpy as np
from itertools import combinations
import h5py

# information that can call directly data through AraRoot... later.....

def antenna_info():

    ant_name =['D1TV', 'D2TV', 'D3TV','D4TV','D1BV', 'D2BV', 'D3BV','D4BV','D1TH', 'D2TH', 'D3TH','D4TH','D1BH', 'D2BH', 'D3BH','D4BH']
    ant_index = np.arange(0,len(ant_name))
    num_ant = len(ant_name)

    return ant_name, ant_index, num_ant

def pol_type(pol_t = 2):

    return pol_t

def bad_antenna(st,Run):
    
    # masked antenna
    if st==2:
        #if Run>120 and Run<4028:
        #if Run>4027:
        bad_ant = np.array([15]) #D4BH
    
    elif st==3:
        #if Run<3014:
        #if Run>3013:
    
        if Run>1901:
            bad_ant = np.array([3,7,11,15])# all D4 antennas
        else:
            bad_ant = np.array([])

    else:
        bad_ant = np.array([2,6,10,14])
    
    return bad_ant

def antenna_combination_maker(arr = antenna_info()[1], mask=-1, com=2, pol=True):
    
    # antenna info
    ant_index, num_ant = antenna_info()[1:]
    num_pol_ant = num_ant//2
    
    # make combination w/ mixed or w/o mixed polarization
    # when do combination, remove masked antenna
    if pol == True:
        v_pairs = list(combinations(arr[:num_pol_ant][ant_index[:num_pol_ant] != mask], com))
        h_pairs = list(combinations(arr[num_pol_ant:][ant_index[num_pol_ant:] != mask], com))
        pairs = v_pairs + h_pairs
        return np.array(pairs), np.array(v_pairs), np.array(h_pairs)
    else:
        paris = list(combinations(arr[ant_index != mask], com))
        return np.array(pairs)

def com_dt_arr_table_maker(file_name, pairs): # need to change npz to root or h5py!

    # number of total antenna
    pairs_len = len(pairs)

    #load file
    table = h5py.File(file_name, 'r')

    # get theta, phi infomation
    axis = table['Table_Axis']
    theta = axis['Thata(rad)'][:] - np.radians(90) # zenith to elevation angle
    phi = axis['Phi(rad)'][:]

    # remove bad antenna
    arr_table = table['Arr_Table']
    re_table = arr_table['Arr_Table(ns)'][:]
    re_table = re_table[:,:,0,:,0]
    #re_table = np.delete(re_table,[mask],2)

    # pre-dt table making
    dt_pairs = np.full((len(theta),len(phi),pairs_len),np.nan)
    arr_pair1 = np.full((len(theta),len(phi),pairs_len),np.nan)
    arr_pair2 = np.full((len(theta),len(phi),pairs_len),np.nan)
    for a in range(pairs_len):
        dt_pairs[:,:,a] = re_table[:,:,pairs[a][0]] - re_table[:,:,pairs[a][1]]
        arr_pair1[:,:,a] = re_table[:,:,pairs[a][0]]
        arr_pair2[:,:,a] = re_table[:,:,pairs[a][1]]
    del pairs_len, table, axis, arr_table

    return dt_pairs, re_table, arr_pair1, arr_pair2, theta, phi



















