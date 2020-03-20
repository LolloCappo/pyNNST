__version__ = '0.1'

import numpy as np

def idns(signal, nsec, overlap, f_sample, confidence):

    """
    Function to determine the index of non-stationarity of a time series:
        
    Arguments:
        signal {numpy array} --  Numpy array of the time series
        nsec {float} -- Time lenght of moving window
        overlap {float} -- Overlap between windows [0 - 1]
        f_sample {int} -- Sampling frequency of the time series
        confidence {int} -- Confidence: 90% - 95% - 98% - 99% - {Default 95(%)}
    
    Returns:
    Dictionary: 
        index {float} -- Index of non-stationarity [Non-stationary --> 0 ; Stationary --> 100]
        bns {boolean} -- 0 --> Non-Stationary 
                         1 --> Stationary
        ind_up {float} -- Upper limit of stationary [%]
        ind_dw {float} -- Lower limit of stationary [%]
        nsec {float} -- Time lenght of moving window
        overlap {float} -- Overlap between windows [0 - 1]
        confidence {int} -- Confidence: 90% - 95% - 98% - 99% - {Default 95(%)}
    """

    
    if confidence == 90:
        conf = 1.645
    elif confidence == 95:
        conf = 1.96
    elif confidence == 98:
        conf = 2.326
    elif confidence == 99:
        conf = 2.576
    else:
        print('Please enter a confidence interval between: 90% - 95% - 98% - 99%')
        print('Default value of 95% is going to be used')
        conf = 1.96
        
        
    ## Definition of time-window variables 
    fftp = int(f_sample * nsec)
    lap = int(fftp * overlap)
    L = signal.shape[0]
    dist = fftp - lap    
    cls = int(np.floor(L/(dist)))
    cmp = np.zeros([dist,cls])
    m_cmp = np.empty(0)
    stdv_cmp = np.empty(0)
    
    ## Definition of statistical properties
    stdv = np.std(signal)

    ## Division of signal in compartments (+residues)
    for i in range(0,cls):
        cmp[:,i] = signal[i*dist:i*dist+dist]
        m_cmp = np.append(m_cmp, np.mean(cmp[:,i]))
        stdv_cmp = np.append(stdv_cmp, np.std(cmp[:,i]))
        
    if L % dist != 0:
        res_cmp = signal[cls*dist+1:]
        m_cmp = np.append(m_cmp,np.mean(res_cmp))
        stdv_cmp = np.append(stdv_cmp, np.std(res_cmp))
    else:
        pass

    ## Definition of boundaries
    boundUP = stdv + np.std(stdv_cmp)
    boundDW = stdv - np.std(stdv_cmp)

    
    ## Run-computation using boundaures
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
    Nr = 0
    N = N1 + N0
    

    for i in range(1,run.shape[0]):
        if run[i] != run[i-1]:
            Nr += 1
        else:
            continue
    
    ## Definition of expected runs and of the variance of their distribution 
    mean_val = (2 * N1 * N0) / N + 1
    var = (2 * N1 * N0 * (2 * N1 * N0 - N)) / (N**2 * (N - 1))
    
    ## Definition of limits
    lim_up = mean_val + conf * np.sqrt(var)
    lim_dw = mean_val - conf * np.sqrt(var)
    
    ind_up = np.round(100 * lim_up / mean_val, 2)
    ind_dw = np.round(100 * lim_dw / mean_val, 2)
    
    
    ## True == Stationary & False == Non-stationary
    if Nr >= lim_dw and Nr <= lim_up:
        bns = True
    else:
        bns = False

    ## Index of non-stationary
    index = 100 * Nr / mean_val 

    if index > 100:
        index = 100
    else:
        index = np.round(index,2)
        
    nnst = {'index': index, 'bns':bns, 'ind_up':ind_up, 'ind_dw':ind_dw, 'nsec':nsec, 'overlap':overlap, 'confidence':confidence}
    
    return nnst