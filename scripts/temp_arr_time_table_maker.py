import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker
from tools.ara_sim_load import ara_raytrace_loader

def temp_arr_time_table_loader(Station = None, Year = None):

    print('Collecting arrival time starts!')

    # arasim raytracer
    ara_ray = ara_raytrace_loader(use_bulk_ice = True, verbose = True)

    # get posisiton for vertex and antenna
    theta_bin = np.array([30, 50, 70, 90, 110, 130, 150])
    phi_bin = np.array([-120, -60, 0, 60, 120, 180])
    radius_bin = np.array([300])
    num_ray_sol = np.array([1], dtype = int)
    ara_ray.get_src_trg_position(Station, Year, theta_bin = theta_bin, phi_bin = phi_bin, radius_bin = radius_bin, num_ray_sol = num_ray_sol, debug = True)

    # arrival time table
    path_len, arr_time_table, launch_ang, receipt_ang, reflection_ang, miss, attenuation = ara_ray.get_arrival_time_table(debug = False) 
 
    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/arr_time_table/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}temp_arr_time_table_A{Station}_Y{Year}.h5'
    hf = h5py.File(h5_file_name, 'w')
    
    #saving result
    hf.create_dataset('ant_pos', data=ara_ray.ant_pos, compression="gzip", compression_opts=9)
    hf.create_dataset('theta_bin', data=ara_ray.theta_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('phi_bin', data=ara_ray.phi_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('radius_bin', data=ara_ray.radius_bin, compression="gzip", compression_opts=9)
    hf.create_dataset('num_ray_sol', data=ara_ray.num_ray_sol, compression="gzip", compression_opts=9)
    hf.create_dataset('path_len', data=path_len, compression="gzip", compression_opts=9)
    hf.create_dataset('arr_time_table', data=arr_time_table, compression="gzip", compression_opts=9)
    hf.create_dataset('launch_ang', data=launch_ang, compression="gzip", compression_opts=9)
    hf.create_dataset('receipt_ang', data=receipt_ang, compression="gzip", compression_opts=9)
    hf.create_dataset('reflection_ang', data=reflection_ang, compression="gzip", compression_opts=9)
    hf.create_dataset('miss', data=miss, compression="gzip", compression_opts=9)
    hf.create_dataset('attenuation', data=attenuation, compression="gzip", compression_opts=9)
    hf.create_dataset('ice_model', data=ara_ray.ice_model, compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name}', size_checker(h5_file_name))

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

    temp_arr_time_table_loader(Station = station, Year = year)













    
