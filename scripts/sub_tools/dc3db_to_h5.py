##
# @file dc3db_to_h5.py
#
# @section Created on 08/10/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to collect weatherballoon gps information from METs dc3db files

import os, sys
import numpy as np
import h5py
import csv
from tqdm import tqdm
from glob import glob
from subprocess import call
from datetime import datetime, timezone

def main(dc3db_path, output_path, gt92_path, del_csv = False):
    """! main function for unpack dc3db files and collect gps information and save into h5 files
        since each file takes only few second to process, let just unpack/scrap all together

    @param dc3db_path  string
    @param output_path  string
    @param gt92_path  string
    @param del_csv  boolean 
    """

    ## make a dc3db file list in the dc3db_path
    dc3db_list = glob(f'{dc3db_path}*/*/*/*.dc3db')
    dc3db_len = len(dc3db_list)
    print('# of total dc3db files:', dc3db_len)
 
    ## dc3db to csv and h5
    print('Unpack all dc3db files and save in h5 files!')

    ## keys for metadata
    meta_label = ['Start', 'SOUNDING_DATA_Start', 'RsTimeResetBase', 'RsActualLaunchTime', 'GPSDCC_RESULT_End', 'SOUNDING_DATA_EDT']
    meta_key = [['!00=', ' (00|'], ['SOUNDING_DATA=', ' (00|'], ['RsTimeResetBase=', 'Z'], ['RsActualLaunchTime=', 'Z'], ['GPSDCC_RESULT=', ' (00|'], ['SOUNDING_DATA\EDT=', ' (00|']]
    freq_key = 'RsType\RsFrequency='
    print('We are interesting...')
    print(meta_label)

    ## load dc3db contents that ARA interests
    csv_list = []
    csv_key = []
    curr_path = os.getcwd()
    with open(f'{curr_path}/dc3db_contents.txt', 'r') as csv_contents:
        csv_readline = csv_contents.readlines()
        for r in csv_readline:
            line_split = r.split(';')
            csv_file_name = line_split[0]
            csv_file_tree = line_split[1].split(',')
            csv_file_tree[-1] = csv_file_tree[-1].strip()
            csv_list.append(csv_file_name)
            csv_key.append(csv_file_tree)
            print(f'{csv_file_name}.csv: {csv_file_tree}')

    ## loop over all dc3db files
    os.chdir(gt92_path) # set the current directory to dc3db package...
    for m in tqdm(range(dc3db_len)):
      #if m == 43 or m == 42: # for debug

        ## collect dc3db file name
        pole_key = dc3db_list[m].find('SOUTHPOLE_')
        dc3db_key = dc3db_list[m].find('.dc3db')
        dc3db_name = dc3db_list[m][pole_key:dc3db_key]

        ## collect year and date from path... dc3db might also have a same file name issue... so evil....
        ara_key_name = '_FOR_ARA/'
        ara_key = dc3db_list[m].find(ara_key_name)
        dc3db_dir_key_name = '/DC3DB/'
        dc3db_dir_key =  dc3db_list[m].find(dc3db_dir_key_name)
        year_name = dc3db_list[m][ara_key+len(ara_key_name):dc3db_dir_key]
        date_name = dc3db_list[m][dc3db_dir_key+len(dc3db_dir_key_name):pole_key-1]
        new_dc3db_name = f'{dc3db_name}_{year_name}_{date_name}_c{m}'

        ## create path for csv files
        csv_path = f'{output_path}csv/{new_dc3db_name}/'
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        
        ## unpack dc3db and store csv
        ## it needs java version 8 or 9 !! 
        ## user must put gt92_path on the LDLIBRARY_PATH
        ## contacts to mkim@icecube.wisc.edu, if user need package
        gt92_CMD = f'./gt92 --no-msg-header --no-msg-info -f DC3DB -x {dc3db_list[m]} -o {csv_path}' 
        print('!!!Msg from gt92 Failed silence it...!!!')
        call(gt92_CMD.split(' '))
        print('!!!Msg from gt92 Failed silence it...!!!')

        ## load metadata scrap the information from the files that ARA needs
        ## first metadata.txt
        date_time = []
        unix_time = np.full((len(meta_label)), np.nan, dtype = float)
        meta_indi_path = f'{csv_path}{dc3db_name}_metadata.txt'
        with open(meta_indi_path, 'r') as txt_contents:
            txt_read = txt_contents.read()
            for k in range(len(meta_label)):
                front_idx = txt_read.find(meta_key[k][0])
                end_idx = txt_read.find(meta_key[k][1], front_idx + len(meta_key[k][0]))
                if front_idx == -1 or end_idx == -1: # some files dont have a gps config... so evil...
                    date_str = 'NaN'
                    unix_time[k] = np.nan
                    print(f'No {meta_key[k][0]} key in {meta_indi_path} Move on!')
                    continue
                else:
                    date_str = txt_read[front_idx + len(meta_key[k][0]):end_idx]
                date_time.append(date_str)
              
                ## convert UTC date time to unix time 
                date_format = datetime.strptime(date_str,"%Y-%m-%dT%H:%M:%S.%f")
                date_format = date_format.replace(tzinfo=timezone.utc)
                unix_stamp = date_format.timestamp()
                unix_time[k] = unix_stamp

        ## second frequency
        freq = np.full((1), np.nan, dtype = float)
        with open(meta_indi_path, 'r') as txt_contents:
            txt_readlines = txt_contents.readlines()
            for l in txt_readlines:
                freq_idx = l.find(freq_key)
                if freq_idx != -1:
                    freq[0] = float(l[freq_idx+len(freq_key):])
                    break
   
        ## create h5 path and write h5 file
        h5_path = f'{output_path}h5/'
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)
        hf = h5py.File(f'{h5_path}{new_dc3db_name}.h5', 'w')
        hf_g = hf.create_group('metadata')
        hf_g.create_dataset('TimeLabel', data=meta_label, compression="gzip", compression_opts=9)
        hf_g.create_dataset('DateTime', data=date_time, compression="gzip", compression_opts=9)
        hf_g.create_dataset('UnixTime', data=unix_time, compression="gzip", compression_opts=9)
        hf_g.create_dataset('Frequency', data=freq, compression="gzip", compression_opts=9)

        ## load csv scrap the information from the files that ARA needs
        ## user can change the contents of scrap by changing dc3db_contents.txt files
        for x in range(len(csv_list)):
            csv_indi_path = f'{csv_path}{dc3db_name}_{csv_list[x]}.csv'

            hf_g = hf.create_group(csv_list[x]) # create group for each xml files
            csv_arr = []
            for c in range(len(csv_key[x])): # make list array for all keys
                csv_arr.append([])
            try:
                with open(csv_indi_path, 'r') as csv_contents:        
                    csvreader = csv.DictReader(csv_contents, delimiter=';') # use panda would be much clean
                    for row in csvreader:
                        row_val = row
                        for c in range(len(csv_key[x])):
                            val = row_val[csv_key[x][c]] # scrap row value based on key
                            csv_arr[c].append(val)
            except FileNotFoundError: # some files dont have a gps csv... so evil...
                print(f'There is no {csv_indi_path} Make empty hf group!')
            csv_arr = np.asarray(csv_arr, dtype = float)
            for c in range(len(csv_key[x])): # save data in each key
                hf_g.create_dataset(csv_key[x][c], data=csv_arr[c], compression="gzip", compression_opts=9)
        hf.close() # close h5 file
    
    ## delete if user dont need csv files
    if del_csv: 
        del_CMD = f'rm -rf {output_path}csv'
        call(del_CMD.split(' '))
        print('csv files are deleted!')
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 4 and len (sys.argv) != 5:
        Usage = """

    Usage = python3 %s 
    <dc3db_path ex)/data/user/mkim/OMF_filter/radiosonde_data/WEATHER_DATA_FOR_ARA/> 
    <output_path ex)/data/user/mkim/OMF_filter/radiosonde_data/weather_balloon/> 
    <gt92_path ex)/home/mkim/analysis/AraSoft/GruanToolRs92/>
    <del_csv = 0>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #argv
    DC3DB_path = str(sys.argv[1])
    Output_path = str(sys.argv[2])
    GT92_path = str(sys.argv[3])
    if len (sys.argv) == 5:
        Del_csv = bool(int(sys.argv[4]))
    else:
        Del_csv = False
    print("DC3DB Path: {}, Output Path: {}, gt92 Path: {}, Del CSV: {}".format(DC3DB_path, Output_path, GT92_path, Del_csv))
    
    main(dc3db_path = DC3DB_path, output_path = Output_path, gt92_path = GT92_path, del_csv = Del_csv)















