import numpy as np

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
        bad_ant = np.array([])
    
    return bad_ant























