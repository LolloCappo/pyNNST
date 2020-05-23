Index of non-stationarity
---------------------------------------------

Obtaining non-stationary index for time-series..

.. code-block:: console


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    signal = np.random.rand(10000) #Create random time-series
    nsec = 0.05                    #Window length [s]
    sampling_freq = 400		   #Sampling frequency [Hz]
    overlap = 0                    #Overlap [0-1]
    confidence = 95                #Confidence [%]

    A = idns(signal, nsec, sampling_freq, overlap, confidence) #Initialize the class
    A.calc() #Do the calculation
    A.get_run() #Get the informations about the run
    A.get_limits() #Get the limits of stationarity [%]
    A.get_bns() #Get the outcome of the test (str)
    A.get_index() #Get the index of non-stationarity [%]
    A.get_plot() #Get the plot of the signals and all the parameters
    

Reference:

Non-stationarity index in vibration fatigue: Theoretical and experimental research; L. Capponi, M. Česnik, J. Slavič, F. Cianetti, M. Boltežar; International Journal of Fatigue 104, 221-230
https://www.sciencedirect.com/science/article/abs/pii/S014211231730316X