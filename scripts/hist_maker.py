import os, sys
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import h5py

Station = str(sys.argv[1])

# tar file list in cobalt
h5_list = sorted(glob(f'/data/user/mkim/OMF_filter/ARA0{Station}/Evt_Wise_SNR_A{Station}_R[0-9][0-9][0-9][0-9].h5'))
print(h5_list)

bin_range_step=np.arange(0,1000,0.05)
#bin_range_step=bin_range_step[:-1]

v_rf = np.zeros((len(bin_range_step)-1))
h_rf = np.copy(v_rf)
v_cal = np.copy(v_rf)
h_cal = np.copy(v_rf)
v_soft = np.copy(v_rf)
h_soft = np.copy(v_rf)

hh = 0
ee = 0
rf_ee = 0
cal_ee = 0
soft_ee = 0
for h in h5_list:
  try:
    h5_file = h5py.File(h, 'r')

    evt_w_snr_v = h5_file['evt_w_snr_v'][:]
    evt_w_snr_h = h5_file['evt_w_snr_h'][:]
    trig = h5_file['trig'][:]
    ee += len(h5_file['evts_num'][:])
    print('stacked evt:',ee)

    rf_loc = np.where(trig == 0)[0]
    cal_loc = np.where(trig == 1)[0]
    soft_loc = np.where(trig == 2)[0]
    rf_ee +=len(rf_loc)
    cal_ee +=len(cal_loc)
    soft_ee +=len(soft_loc)
    #print('stacked rf&cal:',rf_ee, cal_ee)
    print(soft_ee)

    evt_w_snr_v_rf = evt_w_snr_v[rf_loc]
    evt_w_snr_v_cal = evt_w_snr_v[cal_loc]
    evt_w_snr_h_rf = evt_w_snr_h[rf_loc]
    evt_w_snr_h_cal = evt_w_snr_h[cal_loc]
    evt_w_snr_v_soft = evt_w_snr_v[soft_loc]
    evt_w_snr_h_soft = evt_w_snr_h[soft_loc]

    v_rf += np.histogram(evt_w_snr_v_rf, bin_range_step)[0]
    h_rf += np.histogram(evt_w_snr_h_rf, bin_range_step)[0]
    v_cal += np.histogram(evt_w_snr_v_cal, bin_range_step)[0]
    h_cal += np.histogram(evt_w_snr_h_cal, bin_range_step)[0]
    v_soft += np.histogram(evt_w_snr_v_soft, bin_range_step)[0]
    h_soft += np.histogram(evt_w_snr_h_soft, bin_range_step)[0]
    del h5_file, evt_w_snr_v, evt_w_snr_h, trig, rf_loc, cal_loc, soft_loc, evt_w_snr_v_rf, evt_w_snr_v_cal, evt_w_snr_h_rf, evt_w_snr_h_cal, evt_w_snr_v_soft, evt_w_snr_h_soft

    hh += 1
  except OSError:
    pass
print(f'total {hh} runs!')
print(f'total {ee} events!')

path = f'/data/user/mkim/OMF_filter/ARA0{Station}/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
hf = h5py.File(f'Evt_Wise_SNR_A{Station}_RTot.h5', 'w')
hf.create_dataset('bin', data=bin_range_step, compression="gzip", compression_opts=9)
hf.create_dataset('runs', data=np.array([hh]), compression="gzip", compression_opts=9)
hf.create_dataset('evts', data=np.array([ee]), compression="gzip", compression_opts=9)
hf.create_dataset('rfs', data=np.array([rf_ee]), compression="gzip", compression_opts=9)
hf.create_dataset('cals', data=np.array([cal_ee]), compression="gzip", compression_opts=9)
hf.create_dataset('softs', data=np.array([soft_ee]), compression="gzip", compression_opts=9)
hf.create_dataset('v_rf', data=v_rf, compression="gzip", compression_opts=9)
hf.create_dataset('h_rf', data=h_rf, compression="gzip", compression_opts=9)
hf.create_dataset('v_cal', data=v_cal, compression="gzip", compression_opts=9)
hf.create_dataset('h_cal', data=h_cal, compression="gzip", compression_opts=9)
hf.create_dataset('v_soft', data=v_soft, compression="gzip", compression_opts=9)
hf.create_dataset('h_soft', data=h_soft, compression="gzip", compression_opts=9)
hf.close()

fig = plt.figure(figsize=(10, 7))
plt.xlabel(r'Event-wise SNRs [ $V/RMS$ ]', fontsize=25)
plt.ylabel(r'# of events', fontsize=25)
plt.grid(linestyle=':')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.title(f'Evt_Wise_SNR_A{Station}_{hh}runs_{ee}evts', y=1.02,fontsize=15)
#plt.xscale('log')
plt.yscale('log')
#plt.ylim(0,300)
#plt.ylim(0.9,500)
#plt.xlim(3,1e2)
plt.xlim(3,60)
plt.ylim(1e-4,1e8)

plt.plot(bin_range_step[:-1]+(bin_range_step[1]-bin_range_step[0])/2,v_soft, drawstyle='steps',linestyle='-',linewidth=3,color='cyan',alpha=0.5,label=f'Vpol Soft {soft_ee}evts')
plt.plot(bin_range_step[:-1]+(bin_range_step[1]-bin_range_step[0])/2,h_soft, drawstyle='steps',linestyle='-',linewidth=3,color='orange',alpha=0.5,label=f'Hpol Soft {soft_ee}evts')
plt.plot(bin_range_step[:-1]+(bin_range_step[1]-bin_range_step[0])/2,v_cal, drawstyle='steps',linestyle='-',linewidth=3,color='dodgerblue',alpha=0.5,label=f'Vpol Cal {cal_ee}evts')
plt.plot(bin_range_step[:-1]+(bin_range_step[1]-bin_range_step[0])/2,h_cal, drawstyle='steps',linestyle='-',linewidth=3,color='orangered',alpha=0.5,label=f'Hpol Cal {cal_ee}evts')
plt.plot(bin_range_step[:-1]+(bin_range_step[1]-bin_range_step[0])/2,v_rf, drawstyle='steps',linestyle='-',linewidth=3,color='navy',alpha=0.5,label=f'Vpol RF {rf_ee}evts')
plt.plot(bin_range_step[:-1]+(bin_range_step[1]-bin_range_step[0])/2,h_rf, drawstyle='steps',linestyle='-',linewidth=3,color='red',alpha=0.5,label=f'Hpol RF {rf_ee}evts')

plt.legend(loc='lower center',bbox_to_anchor=(1.2,0), numpoints = 1 ,fontsize=15)
fig.savefig(f'Evt_Wise_SNR_A{Station}_{hh}runs_{ee}evts.png',bbox_inches='tight')
#plt.show()
plt.close()

print('Done!')

















