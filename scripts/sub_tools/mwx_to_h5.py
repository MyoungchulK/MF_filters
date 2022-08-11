##
# @file mwx_to_h5.py
#
# @section Created on 08/06/2022, mkim@icecube.wisc.edu
#
# @brief This is designed to collect weatherballoon gps information from METs mwx files
# Inspired by https://github.com/cozzyd/mwx2root

import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from subprocess import call
from datetime import datetime, timezone
from zipfile import ZipFile, BadZipFile
import xml.etree.ElementTree as ET

def main(mwx_path, output_path, del_xml = False):
    """! main function for unpack mwx files and collect gps information and save into h5 files
        since each file takes only few second to process, let just unpack/scrap all together

    @param mwx_path  string
    @param output_path  string
    @param del_xml  boolean 
    """

    ## make a mwx file list in the mwx_path
    mwx_list = glob(f'{mwx_path}*/*/*/*.mwx')
    mwx_len = len(mwx_list)
    print('# of total mwx files:', mwx_len)
 
    ## mwx to xml and h5
    print('Unpack all mwx files and save in h5 files!')

    ## load mwx contents that ARA interests
    curr_path = os.getcwd()
    xml_contents = open(f'{curr_path}/mwx_contents.txt', 'r')
    xml_readline = xml_contents.readlines()
    xml_list = []
    xml_key = []
    print('We are interesting...')
    for r in range(len(xml_readline)):
        line_split = xml_readline[r].split(' ')
        xml_file_name = line_split[0]
        xml_file_tree = line_split[1].split(',')
        xml_file_tree[-1] = xml_file_tree[-1].strip() 
        print(f'{xml_file_name}.xml: {xml_file_tree}')
        xml_list.append(xml_file_name)
        xml_key.append(xml_file_tree)
    xml_contents.close()

    ## loop over all mwx files
    for m in tqdm(range(mwx_len)):
      #if m > 4053: # for debug

        ## collect mwx file name
        nzsp_key = mwx_list[m].find('NZSP_')
        mwx_key = mwx_list[m].find('.mwx')
        mwx_name = mwx_list[m][nzsp_key:mwx_key]

        ## collect year and date from path... almost 1000 files have a same file name... so evil....
        ara_key_name = '_FOR_ARA/'
        ara_key = mwx_list[m].find(ara_key_name)
        mwx_dir_key_name = '/MWX/'
        mwx_dir_key =  mwx_list[m].find(mwx_dir_key_name)
        if mwx_dir_key == -1:
            mwx_dir_key_name = '/DC3DB/'
            mwx_dir_key =  mwx_list[m].find(mwx_dir_key_name)
        year_name = mwx_list[m][ara_key+len(ara_key_name):mwx_dir_key]
        date_name = mwx_list[m][mwx_dir_key+len(mwx_dir_key_name):nzsp_key-1]
        new_mwx_name = f'{mwx_name}_{year_name}_{date_name}_c{m}'

        ## create path for xml files
        xml_path = f'{output_path}xml/{new_mwx_name}/'
        if not os.path.exists(xml_path):
            os.makedirs(xml_path)
        
        ## unpack mwx and store xml
        with ZipFile(mwx_list[m], 'r') as zipObj:
            xml_sub_list = zipObj.namelist()
            for xl in range(len(xml_sub_list)): # unpack one by one since some of files are showing corrupted
                try:
                    zipObj.extract(xml_sub_list[xl], xml_path)
                except BadZipFile:
                    print(f'{xml_sub_list[xl]} in {mwx_list[m]} are corrupted. Move on!')

        ## deal with corrupted xml files...
        for l in range(len(xml_list)):
            if mwx_name == 'NZSP_20211113_204249' and xml_list[l] == 'GpsResults': # this evil balloon flight was failed to launch
                continue
            with open(f'{xml_path}{xml_list[l]}.xml', "r") as f:
                context = f.read()
                err_idx = context.find('&')
                if err_idx != -1:
                    print(f'value: {context[err_idx]} and index: {err_idx} in {xml_path}{xml_list[l]}.xml are not for xml. Replace to 0!')
                    context = context.replace('&','0')
                    with open(f'{xml_path}{xml_list[l]}.xml', "w") as f:
                        f.write(context)

        ## create h5 path and write h5 file
        h5_path = f'{output_path}h5/'
        if not os.path.exists(h5_path):
            os.makedirs(h5_path) 
        hf = h5py.File(f'{h5_path}{new_mwx_name}.h5', 'w')

        ## scrap the infirmation from the files that ARA needs
        ## user can change the contents of scrap by changing mwx_contents.txt files 
        for x in range(len(xml_list)):      
            hf_g = hf.create_group(xml_list[x]) # create group for each xml files

            if mwx_name == 'NZSP_20211113_204249' and xml_list[x] == 'GpsResults': # this evil balloon flight was failed to launch            
                print(f'{new_mwx_name} doesnt have a GpsResults! Fill the array with NAN!')
                ## ctreate empty dataset
                hf_g.create_dataset('UnixTime', data=temp_unix_time, compression="gzip", compression_opts=9) # fill the SystemEvents unix time to array
                nan_array = np.full((len(temp_unix_time)), np.nan, dtype = float)
                contents = xml_key[x]
                for c in range(len(contents)):
                    hf_g.create_dataset(contents[c], data=nan_array, compression="gzip", compression_opts=9)
                continue

            ## load xml files
            xml_files = f'{xml_path}{xml_list[x]}.xml'
            tree = ET.parse(xml_files)
            root = tree.getroot()
            root_len = len(root)

            ## scrap contents
            contents = xml_key[x]
            for c in range(len(contents)):
                dat_arr = []
                for r in range(root_len): # loop over each row
                    dat_str = root[r].get(contents[c])
                    dat_arr.append(dat_str)            

                ## seperate string array and number array
                if contents[c] == 'DataSrvTimePk' or contents[c] == 'DataSrvTime':
                    unix_time = np.full((len(dat_arr)), np.nan, dtype = float)
                    for t in range(len(dat_arr)):
                        date_format = datetime.strptime(dat_arr[t][:-1],"%Y-%m-%dT%H:%M:%S.%f")
                        date_format = date_format.replace(tzinfo=timezone.utc)
                        unix_stamp= date_format.timestamp()
                        unix_time[t] = unix_stamp
                    hf_g.create_dataset('UnixTime', data=unix_time, compression="gzip", compression_opts=9)
                    if mwx_name == 'NZSP_20211113_204249' and xml_list[x] == 'SystemEvents':
                        temp_unix_time = np.copy(unix_time)
                elif contents[c] == 'EventType' or contents[c] == 'Data':
                    pass
                else:
                    dat_arr = np.asarray(dat_arr, dtype = float)

                ## save into h5 group
                hf_g.create_dataset(contents[c], data=dat_arr, compression="gzip", compression_opts=9)
            
        hf.close() # close h5 file
        
    ## delete if user dont need xml files
    if del_xml: 
        del_CMD = f'rm -rf {output_path}xml'
        call(del_CMD.split(' '))
        print('xml files are deleted!')
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 3 and len (sys.argv) != 4:
        Usage = """

    Usage = python3 %s <mwx_path ex)/data/user/mkim/OMF_filter/radiosonde_data/WEATHER_DATA_FOR_ARA/> <output_path ex)/data/user/mkim/OMF_filter/radiosonde_data/weather_balloon/> <del_xml = 0>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #argv
    MWX_path = str(sys.argv[1])
    Output_path = str(sys.argv[2])
    if len (sys.argv) == 4:
        Del_xml = bool(int(sys.argv[3]))
    else:
        Del_xml = False
    print("MWX Path: {}, Output Path: {}, Del XML: {}".format(MWX_path, Output_path, Del_xml))
    
    main(mwx_path = MWX_path, output_path = Output_path, del_xml = Del_xml)















