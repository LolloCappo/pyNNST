__version__ = '0.1'

import numpy as np

def idns(signal, nsec, overlap, f_sample, conf):

    """
    Function to determine the non-stationarity of a time series:
        
    Arguments:
        signal {numpy array} --  column array of the time series
        nsec {float} -- time lenght of moving window
        overlap {float} -- overlap between windows [0 - 1]
        f_sample {int} -- sampling frequency of the time series
        conf {float} -- confidence interval coefficent [1.96 for 95% - 2.58 for 99%]
    
    Returns:
        index {float} -- index of non-stationarity [Non-stationary --> 0 ; Stationary --> 100]
        R {string} -- Definition of stationarity or non-stationarity 
    """

    fftp = int(f_sample * nsec)
    lap = int(fftp * overlap)
    L = signal.shape[0]
    dist = fftp - lap

    stdv = np.std(signal)
    cls = int(np.floor(L/(dist)))
    cmp = np.zeros([dist,cls])
    m_cmp = np.empty(0)
    stdv_cmp = np.empty(0)

    for i in range(0,cls):
        cmp[:,i] = signal[i*dist:i*dist+dist,0]
        m_cmp = np.append(m_cmp, np.mean(cmp[:,i]))
        stdv_cmp = np.append(stdv_cmp, np.std(cmp[:,i]))
    if L % dist != 0:
        res_cmp = signal[cls*dist+1:]
        m_cmp = np.append(m_cmp,np.mean(res_cmp))
        stdv_cmp = np.append(stdv_cmp, np.std(res_cmp))
    else:
        pass

    boundUP = stdv + np.std(stdv_cmp)
    boundDW = stdv - np.std(stdv_cmp)

    run = np.empty(0)
    pos = np.empty(0)
    neg = np.empty(0)

    for i in range(0,stdv_cmp.shape[0]):
        if stdv_cmp[i] > boundUP or stdv_cmp[i] < boundDW:
            run = np.append(run,1)
        else:
            run = np.append(run,0)
    for i in range(0,run.shape[0]):
        if run[i] == 1.:
            pos = np.append(pos, run[i])
        else:
            neg = np.append(neg, run[i])

    N1 = pos.shape[0]
    N0 = neg.shape[0]
    N = N1 + N0
    Nr = 0

    for i in range(1,run.shape[0]):
        if run[i] != run[i-1]:
            Nr += 1
        else:
            continue
        
    mean_val = (2 * N1 * N0)/N+1

    index=100*Nr/mean_val 

    if index > 100:
        index = 100
    else:
        index = np.round(index,2)
        
    var = (2 * N1 * N0 * (2 * N1 * N0-N)) / (N**2 * (N-1))
    INTE1 = mean_val + conf * np.sqrt(var)
    INTE2 = mean_val - conf * np.sqrt(var)
    
    if Nr >= INTE2 and Nr <= INTE1:
        R = 'Stationary signal'
    else:
        R = 'Non-stationary signal'

    
    return index, R