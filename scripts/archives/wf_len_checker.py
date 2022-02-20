import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.utility import size_checker
from tools.ara_data_load import analog_buffer_info_loader

def wf_len_loader(Station = None, Year = None):

    print('Station:', Station)
    print('Year:', Year)

    run = 0
    if  Station == 3 and Year == 2019:
        run = 12866
        print(run)
    buffer_info = analog_buffer_info_loader(Station, run, Year, incl_cable_delay = True)
    buffer_info.get_int_time_info()

    time_blk = buffer_info.time_num_arr
    int_time_blk = buffer_info.int_time_num_arr
    int_time_blk_f = buffer_info.int_time_f_num_arr
    print(time_blk.shape)
    print(int_time_blk.shape)
    print(int_time_blk_f.shape)

    time_blk[:,0] += 20.
    int_time_blk[:,0] += 20.
    int_time_blk_f[:,1] += 20.

    time_i = np.nanmin(time_blk, axis = 0)
    time_f = np.nanmax(time_blk, axis = 0)+46*20.
    print(time_i)   
    print(time_f)
    print(np.nanmin(time_i), np.nanmax(time_f), np.nanmax(time_f) - np.nanmin(time_i))

    int_time_i = np.nanmin(int_time_blk, axis = 0)
    int_time_f = np.nanmin(int_time_blk_f, axis = 0)+46*20.
    print(int_time_i)    
    print(int_time_f)
    print(np.nanmin(int_time_i), np.nanmax(int_time_f), np.nanmax(int_time_f) - np.nanmin(int_time_i))

    

    """ 
    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/arr_time_table/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}arr_time_table_A{Station}_Y{Year}.h5'
    hf = h5py.File(h5_file_name, 'w')
    
    #saving result
    hf.create_dataset('ant_pos', data=ara_ray.ant_pos, compression="gzip", compression_opts=9)
    hf.create_dataset('theta_bin', data=ara_ray.theta_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('phi_bin', data=ara_ray.phi_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('radius_bin', data=ara_ray.radius_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('num_ray_sol', data=np.array([ara_ray.num_ray_sol]), compression="gzip", compression_opts=9)
    hf.create_dataset('arr_time_table', data=arr_time_table, compression="gzip", compression_opts=9)
    hf.create_dataset('ice_model', data=ara_ray.ice_model, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name}')

    # quick size check
    size_checker(h5_file_name)  
    """
if __name__ == "__main__":

    if len (sys.argv) < 3:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)3>
    <Year ex)2018>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    year=int(sys.argv[2])

    wf_len_loader(Station = station, Year = year)













    
