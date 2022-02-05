import numpy as np
import os, sys
from tqdm import tqdm
from glob import glob
import re

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')

def l1_data_checker(st = None, output = None, analyze_blind_dat = False):

   
    years = np.arange(2013,2020, dtype = int) 
    if st == 3:
        years = years[years != 2017]

    run_num_list = []
    dat_path_list = []
    content_list = []
    problem_list = []

    run_num_key = ['run','/']

    if analyze_blind_dat == True:
        dat_type_2013 = 'full2013Data'
        dat_type = 'blinded'
    else:
        dat_type_2013 = 'burnSample1in10'
        dat_type = 'unblinded'

    for yrs in years:
        yrs = int(yrs)

        if yrs == 2013:
            dat_path = glob(f'/data/exp/ARA/{yrs}/filtered/{dat_type_2013}/ARA0{st}/root/*/*/*')
        else:
            dat_path = glob(f'/data/exp/ARA/{yrs}/{dat_type}/L1/ARA0{st}/*/*/time*')

        for dat in dat_path:
            print(dat)

    print(f'Output path check:{output}')
    if not os.path.exists(output):
        os.makedirs(output)


if __name__ == "__main__":

    if len (sys.argv) < 4:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Output ex)/home/mkim/analysis/MF_filters/scripts/batch_run/wipac/>
    <Analyze_Blind ex)0 or 1>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    output=str(sys.argv[2])
    blind=bool(int(sys.argv[3]))

    l1_data_checker(st = station, output = output, analyze_blind_dat = blind)
 
