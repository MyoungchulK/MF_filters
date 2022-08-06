import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from subprocess import call
from datetime import datetime, timezone
from zipfile import ZipFile, BadZipFile
import xml.etree.ElementTree as ET

def main(del_xml = False, del_indi_h5 = False, use_xml = False):

    # paths
    general_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/radiosonde_data/'
    mwx_path = f'{general_path}MWX/'
    xml_path = f'{general_path}xml/'
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    h5_path = f'{general_path}h5/'
    if not os.path.exists(h5_path):
        os.makedirs(h5_path)

    # mwx list
    mwx_list = glob(f'{mwx_path}*/*/*')
    mwx_len = len(mwx_list)
    print('# of total mwx files:', mwx_len)

    # mwx to xml
    xml_list = []
    mwx_file_name = []
    if use_xml:
        for m in tqdm(range(mwx_len)):
          #if m < 10:
            mwx_indi_path = mwx_list[m]
            nzsp_key = mwx_indi_path.find('NZSP_')
            mwx_key = mwx_indi_path.find('.mwx')
            mwx_name = mwx_indi_path[nzsp_key:mwx_key]
            mwx_file_name.append(mwx_name)
            xml_indi_path = f'{xml_path}{mwx_name}/'
            xml_list.append(xml_indi_path)
    else:
        for m in tqdm(range(mwx_len)):
          #if m < 10:
            mwx_indi_path = mwx_list[m]
            nzsp_key = mwx_indi_path.find('NZSP_')
            mwx_key = mwx_indi_path.find('.mwx')
            mwx_name = mwx_indi_path[nzsp_key:mwx_key]
            mwx_file_name.append(mwx_name)

            xml_indi_path = f'{xml_path}{mwx_name}/'
            if not os.path.exists(xml_indi_path):
                os.makedirs(xml_indi_path)
            xml_list.append(xml_indi_path)

            with ZipFile(mwx_list[m], 'r') as zipObj:
                xml_sub_list = zipObj.namelist()
                for xl in range(len(xml_sub_list)):
                    try:
                        zipObj.extract(xml_sub_list[xl], xml_indi_path)
                    except BadZipFile:
                        print(mwx_list[m])
                    print(xml_sub_list[xl])
    xml_len = len(xml_list)
    print('# of total xml paths:', xml_len)

    # xml to h5. only few infomation that ara needs
    xml_interests = ['SystemEvents', 'RadioDiagnostics', 'GpsResults']
    int_key = [['DataSrvTimePk', 'EventType', 'Data'],
               ['DataSrvTime','RadiosondeCarrier','SignalLevelPeak','NoiseFloor','SignalPower','NoisePower','SNR'],
               ['DataSrvTime','GpsSeconds','Wgs84Latitude','Wgs84Longitude','Wgs84Altitude','Wgs84X','Wgs84Y','Wgs84Z']] 

    for x in tqdm(range(xml_len)):
      #if x > 1766:
        xml_indi_list = xml_list[x]
        h5_indi_path = f'{h5_path}{mwx_file_name[x]}.h5'
        hf = h5py.File(h5_indi_path, 'w')
        #print(mwx_file_name[x])

        for xx in range(len(xml_interests)):
          #if xx == 0:
            #print(xml_interests[xx])
            g = hf.create_group(xml_interests[xx])
            xml_int_path = f'{xml_indi_list}{xml_interests[xx]}.xml'
            tree = ET.parse(xml_int_path)
            root = tree.getroot()
            root_len = len(root)

            dat_key = int_key[xx]
            for d in range(len(dat_key)):
                dat_arr = []
                for r in range(root_len):
                    dat_str = root[r].get(dat_key[d])
                    dat_arr.append(dat_str)
                if dat_key[d] == 'DataSrvTimePk' or dat_key[d] == 'DataSrvTime' or dat_key[d] == 'EventType'  or dat_key[d] == 'Data':
                    pass
                else:
                    dat_arr = np.asarray(dat_arr, dtype = float)
                g.create_dataset(dat_key[d], data=dat_arr, compression="gzip", compression_opts=9)
                if dat_key[d] == 'DataSrvTimePk' or dat_key[d] == 'DataSrvTime':
                    unix_arr = np.full((len(dat_arr)), np.nan, dtype = float)
                    for t in range(len(dat_arr)):
                        date_format = datetime.strptime(dat_arr[t][:-1],"%Y-%m-%dT%H:%M:%S.%f")
                        date_format = date_format.replace(tzinfo=timezone.utc)
                        unix_stamp= date_format.timestamp()
                        unix_arr[t] = unix_stamp
                    g.create_dataset('UnixTime', data=unix_arr, compression="gzip", compression_opts=9)
        hf.close()                
    print('# of total h5 files:', len(glob(f'{h5_path}*')))

    if del_xml: 
        del_CMD = f'rm -rf {xml_path}'
        call(del_CMD.split(' '))
        print('xml files are deleted!')
    if del_indi_h5:
        del_CMD = f'rm -rf {h5_path}'
        call(del_CMD.split(' '))
        print('h5 files are deleted!')
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 4:
        Usage = """

    Usage = python3 %s <del ex)0 ro 1> <del h5 ex)0 or 1> <use xml ex)0 or 1>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    #argv
    Del_xml = bool(int(sys.argv[1]))
    Del_h5 = bool(int(sys.argv[2]))
    USE_xml = bool(int(sys.argv[3]))

    main(Del_xml, Del_h5, USE_xml)















