import numpy as np

def freq_pad_maker(t_len, dt):

    freq = np.fft.fftfreq(t_len, dt)

    return freq, np.abs(freq[1]-freq[0])








