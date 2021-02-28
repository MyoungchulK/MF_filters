import numpy as np
import h5py

# custom lib
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
#from tools.mf import off_pad_maker

def plane_table(num_Antennas, trg_xyz, index_of_refr, nadir_min = 0.5, nadir_max = 179.5+0.5, phi_min = 0.5, phi_max = 359.5+0.5, angle_width = 1):

    # nadir dim
    nadir_range = np.radians(np.arange(nadir_min, nadir_max, angle_width))
    nadir_range_len = len(nadir_range)

    # phi dim
    phi_range = np.radians(np.arange(phi_min, phi_max, angle_width))
    phi_range_len = len(phi_range)

    # repeatting to match the dim
    nadir = np.repeat(nadir_range[:,np.newaxis], phi_range_len, axis=1)
    phi = np.repeat(phi_range[np.newaxis, :], nadir_range_len, axis=0)
    #del nadir_range, phi_range

    # xyz for unit vector in center
    sin_nadir = np.sin(nadir)
    cen_unit_x = sin_nadir * np.cos(phi)
    cen_unit_y = sin_nadir * np.sin(phi)
    cen_unit_z = np.cos(nadir)
    del nadir, phi, sin_nadir

    # unit vector in center
    cen_unit = np.array([cen_unit_x, cen_unit_y, cen_unit_z])
    cen_unit = np.repeat(cen_unit[:, np.newaxis, :, :], num_Antennas, axis=1)
    del cen_unit_x, cen_unit_y, cen_unit_z

    # r for unit vector in center
    cen_unit_r = np.sqrt(np.nansum(cen_unit**2,axis=0))

    # center of the station
    trg_cen = np.nanmean(trg_xyz,axis=1)

    # antenna vector from center of the station
    ants_vec = trg_xyz - trg_cen[:,np.newaxis]
    ants_vec = np.repeat(ants_vec[:,:,np.newaxis], nadir_range_len, axis=2)
    ants_vec = np.repeat(ants_vec[:,:,:,np.newaxis], phi_range_len, axis=3)
    del nadir_range_len, phi_range_len, trg_cen
    
    # r for antenna vector from center
    ants_r = np.sqrt(np.nansum(ants_vec**2,axis=0))

    # angle calculation between unit vector and antenna vector
    AB = np.nansum(cen_unit * ants_vec, axis=0)
    ABabs = cen_unit_r * ants_r
    cos_rad=np.arccos(AB/ABabs)
    del cen_unit, ants_vec, cen_unit_r, AB, ABabs

    # path time from the plane
    light_in_ice = 299792458 / index_of_refr
    path_dT = (ants_r * np.cos(cos_rad) / light_in_ice) * 1e9
    path_dT_avg = np.round(path_dT * 2) / 2
    del ants_r, light_in_ice, cos_rad

    # max path_dT
    arr_max_len = np.nanmax([np.abs(np.nanmin(path_dT_avg)),np.nanmax(path_dT_avg)])

    return path_dT, path_dT_avg, arr_max_len, np.degrees(nadir_range), np.degrees(phi_range)

def mov_index_table(path_dT_avg, arr_max_len, Search_Len):

    # interpolation time width
    time_width_ns = interpolation_bin_width()[0]

    # make wf pad
    time_pad_len = time_pad_maker(time_width_ns)[1]

    #off_t = np.arange(0,time_pad_len,1) * time_width_ns
    #off_t_len = len(off_t)
    #off_t_first = off_t[0]
    #off_t_end = off_t[-1]

    # make offset pad
    from tools.mf import off_pad_maker
    off_t, off_t_len, off_t_first, off_t_end = off_pad_maker(time_pad_len, time_width_ns)
    del time_pad_len, off_t_len

    # peak searching length
    ps_len_index = int(Search_Len / time_width_ns) +1

    #moving time
    mov_t = np.arange(off_t_first - arr_max_len - Search_Len, off_t_end + arr_max_len + Search_Len + time_width_ns, time_width_ns)

    #padding time
    #pad_t_first = mov_t[0] - arr_max_len
    #pad_t_end = mov_t[0] + arr_max_len
    pad_t = np.arange(off_t_first - 2*arr_max_len - Search_Len, off_t_end + 2*arr_max_len + Search_Len + time_width_ns, time_width_ns)
    del arr_max_len, off_t_first, off_t_end

    #padding length index
    pad_len_front = int((off_t[0] - pad_t[0]) / time_width_ns)
    pad_len_end  = int((pad_t[-1] - off_t[-1]) / time_width_ns)
    del off_t

    # moving index
    mov_index = np.broadcast_to((mov_t - pad_t[0])[:, np.newaxis, np.newaxis, np.newaxis], (mov_t.shape[0],path_dT_avg.shape[0],path_dT_avg.shape[1],path_dT_avg.shape[2])) + np.broadcast_to(path_dT_avg[np.newaxis, :,:,:], (mov_t.shape[0],path_dT_avg.shape[0],path_dT_avg.shape[1],path_dT_avg.shape[2]))
    mov_index = (mov_index / time_width_ns).astype(int)
    del time_width_ns

    return mov_index, pad_t, pad_len_front, pad_len_end, ps_len_index, mov_t

def table_loader(c_path, Station, grid, peak):

    table_path = c_path + '/table/'
    table_name = 'Plane_Table_A'+str(Station)+'_Y2013_GS'+str(grid)+'_PW'+str(peak)+'_lite.h5'
    table_file = h5py.File(table_path+table_name, 'r')
    
    mov_index = table_file['mov_index'][:] 
    mov_t = table_file['mov_t'][:] 
    pad_t = table_file['pad_t'][:] 
    pad_len_front = table_file['pad_len_front'][0] 
    pad_len_end = table_file['pad_len_end'][0] 
    ps_len_index = table_file['ps_len_index'][0]
    del table_path, table_name, table_file

    print('Table loading is done!')

    return mov_index, len(pad_t), pad_len_front, pad_len_end, ps_len_index, mov_t, pad_t

def nz_ice(z, a = 1.78, b = 1.326, c = 0.0202):

    #if z > 0:
    #    return 0
    #else:
        nz = a - (a - b) * np.exp(c * z)
        return nz




















