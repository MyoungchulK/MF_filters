import numpy as np
from scipy.stats import rayleigh
from scipy.optimize import minimize
#from scipy.stats import chisquare
from tqdm import tqdm

def compressed_array(data, flatten_type = 'F', trans = False):

    com_data = []
    ant_num = data.shape[1]
    for ant in range(ant_num):
        if ant == 15:
            continue
        ant_data = data[:, ant].flatten(flatten_type)
        ant_data = ant_data[~np.isnan(ant_data)]
        com_data.append(ant_data)
    com_data = np.asarray(com_data)
    if trans == True:
        com_data = np.transpose(com_data, (1,0))
    del ant_num

    return com_data

def decompressed_array(data, decom_dim, none_nan_len):

    decom_data = np.full(decom_dim, np.nan, dtype = float)
    front_idx = 0
    for evt in tqdm(range(decom_dim[1])):
        decom_data[:none_nan_len[evt], evt] = data[front_idx:front_idx+none_nan_len[evt]]
        front_idx += none_nan_len[evt]
    del front_idx

    return decom_data

def ratio_cal(data, sig_val = 3):

    data_len = len(data)
    flag_data_len = np.count_nonzero(data < sig_val)

    try:
        ratio = flag_data_len/data_len*100
    except ZeroDivisionError:
        ratio = np.nan

    return ratio

def mu_sigma_unwrapper(argvs, n_dim):

    mu = argvs[:n_dim]
    cov_mtx = argvs[n_dim:].reshape(n_dim,n_dim)

    return mu, cov_mtx

def mahalanobis_distance(mu, cov_mtx, data):

    # coval matrix
    sigma_inv = np.linalg.inv(cov_mtx)

    # mahalanobis distance
    sig = np.sqrt(np.einsum('...k,kl,...l->...', data-mu, sigma_inv, data-mu))
    del sigma_inv

    return sig   

def log_likelihood_multi_dim_gaussian(argvs, data):

    # rescue the dim, mean, coval
    n_len = data.shape[0]
    n_dim = data.shape[1]
    mu, cov_mtx = mu_sigma_unwrapper(argvs, n_dim)

    # coval matrix
    sigma_det = np.linalg.det(cov_mtx)
    sigma_inv = np.linalg.inv(cov_mtx)
    del cov_mtx

    # log gaussian
    const_term = -0.5 * np.log((2*np.pi)**n_dim * sigma_det) * n_len
    exp_term = -0.5 * np.einsum('...k,kl,...l->...', data-mu, sigma_inv, data-mu)
    del sigma_det, sigma_inv, n_dim, mu, n_len

    # negative log likelihood
    nllmdg = -(const_term + np.nansum(exp_term))
    del const_term, exp_term
    
    return nllmdg

def minimize_multi_dim_gaussian(data):

    print('Minimizer starts!')

    # inintial guess
    data_mean = np.nanmean(data, axis=0)
    data_trans = np.transpose(data, (1,0))
    data_cov = np.cov(data_trans).flatten()
    init_guess = np.append(data_mean, data_cov)

    # set boundary
     # set boundary
    bound_arr = np.full((init_guess.shape[0], 2), np.nan, dtype = float)
    bound_range = 1.2
     
    a = data_mean/bound_range
    b = data_mean*bound_range
    d = np.cov(data_trans/bound_range).flatten()
    c = np.cov(data_trans*bound_range).flatten()
    
    bound_arr[:data_mean.shape[0],0] = np.minimum(a,b)
    bound_arr[:data_mean.shape[0],1] = np.maximum(a,b)
    bound_arr[data_mean.shape[0]:,0] = np.minimum(c,d)
    bound_arr[data_mean.shape[0]:,1] = np.maximum(c,d)
    del bound_range, data_mean, data_cov, data_trans, a, b, c, d

    # minimizer
    result = minimize(log_likelihood_multi_dim_gaussian, init_guess, args = (data), bounds = bound_arr)
    del init_guess, bound_arr

    # minimize result
    min_mu, min_cov_mtx = mu_sigma_unwrapper(result.x, data.shape[1])
    success_bool = result.success
    print(f'Success: {success_bool}, Message: {result.message}')
    del result

    print('Minimizer is done!')

    return min_mu, min_cov_mtx, int(success_bool)


def rayleigh_fit(fft, binning = 1000):

    print('Rayleigh fitting starts!')

    # set shape
    freq_len = fft.shape[0]
    ant_num = fft.shape[1]
   
    # output array
    mu = np.full((freq_len, ant_num), np.nan)

    # loop
    for f in tqdm(range(freq_len)):
        for a in range(ant_num):

            # histogram
            bin_range = np.linspace(np.nanmin(fft[f,a]), np.nanmax(fft[f,a]), binning + 1)
            bin_center = (bin_range[1:] + bin_range[:-1])/2
            hist = np.histogram(fft[f,a], bins = bin_range)[0]
            del bin_range

            # mu for data   
            mu_idx = np.where(hist==np.nanmax(hist))[0]
            del hist
            if len(mu_idx) > 0:
                mu_bin = bin_center[mu_idx[0]]
                try:
                    # fitting
                    loc, scale = rayleigh.fit(fft[f,a], loc=np.nanmin(fft[f,a]), scale=mu_bin)
                    mu[f,a] = loc + scale
                    del loc, scale, mu_bin
                except RuntimeError:
                    #print('Runtime Issue!')
                    pass
            else:
                pass
            del mu_idx, bin_center
    del freq_len, ant_num

    print('Rayleigh fitting is done!')

    return mu


def rayleigh_fit_complicate(fft, binning = 1000, 
                    save_mu = False, 
                    save_chi2 = False, 
                    save_pdf = False, 
                    save_hist = False, 
                    save_hist_err = False, 
                    save_bin = False):

    print('Rayleigh fitting starts!')

    # set shape
    freq_len = fft.shape[0]
    ant_num = fft.shape[1]

    # output array
    if save_mu == True:
        mu = np.full((freq_len, ant_num), np.nan)
    else:
        mu = None
    if save_hist == True or save_chi2 == True:
        hist_arr = np.full((freq_len, ant_num, binning), np.nan)
    else:
        hist_arr = None
    if save_hist_err ==True or save_chi2 == True:
        hist_err_arr = np.full((freq_len, ant_num, binning), np.nan)
    else:
        hist_err_arr = None
    if save_pdf == True or save_chi2 == True:
        pdf_arr = np.full((freq_len, ant_num, binning), np.nan)
    else:
        pdf_arr = None
    if save_bin == True: 
        bin_range_arr = np.full((freq_len, ant_num, binning+1), np.nan)
        bin_center_arr = np.full((freq_len, ant_num, binning), np.nan)
    else:
        bin_range_arr = None
        bin_center_arr = None

    # loop
    for f in tqdm(range(freq_len)):
        for a in range(ant_num):
       
            bin_range = np.linspace(np.nanmin(fft[f,a]), np.nanmax(fft[f,a]), binning + 1)
            bin_center = (bin_range[1:] + bin_range[:-1])/2
            if save_bin == True:
                bin_range_arr[f,a] = bin_range
                bin_center_arr[f,a] = bin_center
            else:
                pass

            hist = np.histogram(fft[f,a], bins = bin_range, density = True)[0]
            if save_hist == True or save_chi2 == True:
                hist_arr[f,a] = hist
            else:
                pass
            if save_hist_err == True or save_chi2 == True:
                hist_err_arr[f,a] = np.sqrt(np.histogram(bin_center, bins=bin_range, weights=hist**2)[0])
            else:
                pass

            if save_mu == True or save_pdf == True or save_chi2 == True:
                try:
                    mu_index = bin_center[np.where(hist==np.nanmax(hist))[0][0]]
                    try:
                        loc, scale = rayleigh.fit(fft[f,a], loc=np.nanmin(fft[f,a]), scale=mu_index)
                        del mu_index
                        if save_mu == True:
                            mu[f,a] = loc + scale
                        else:
                            pass
                        if save_pdf == True or save_chi2 == True:
                            pdf_arr[f,a] = rayleigh(scale=scale, loc=loc).pdf(bin_center)
                        else:
                            pass
                        del loc, scale
                    except RuntimeError:
                        print('RuntimeError!')
                        pass
                except IndexError:
                    pass

            del bin_range, bin_center, hist

    if save_chi2 == True:
        ndf = binning - 1 - 2
        chi2 = ((hist_arr - pdf_arr) / hist_err_arr)**2
        chi2 = np.nansum(chi2,axis=2) / ndf
        del ndf
    else:
        chi2 = None
    if save_chi2 == True and save_hist == False:
        hist_arr = None
    else:pass
    if save_chi2 == True and save_hist_err == False:
        hist_err_arr = None
    else:pass
    if save_chi2 == True and save_pdf == False:
        pdf_arr = None
    else:pass

    del freq_len, ant_num

    print('Rayleigh fitting is done!')

    return mu, chi2, hist_arr, hist_err_arr, pdf_arr, bin_range_arr, bin_center_arr


