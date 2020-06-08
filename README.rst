Index of non-stationarity
---------------------------------------------

Obtaining non-stationary index for time-series..

.. code-block:: console


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

    # Import packages 
    from pyNNST import *
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.lib.stride_tricks import as_strided

    # Define a sample signal x
    T = 20                                # Time length of x
    fs = 400                              # Sampling frequency of x
    dt = 1 / fs                           # Time between discreete signal values
    x = np.random.rand(T * fs)            # Signal
    time = np.linspace(0, T - dt, T * fs) # Time vector
    std = np.std(x, ddof = 1)             # Standard deviation of x
    mean = np.mean(x)                     # Mean value of x

    # Class initialization
    nperseg = 100
    noverlap = 0
    confidence = 95
    example = Idns(x, nperseg = nerperseg, noverlap = noverlap, confidence = confidence)
    
    # Compute the run test for non-stationarity
    example.nnst() 
    outcome = example.get_outcome()  # Get the results of the test as a string
    index = example.get_index()      # Get the index of non-stationarity
    limits = example.get_limits()    # Get the limits outside of which the signal is non-stationary

    segments_std, bound_dw, bound_up = example.get_segments() # Standard deviation of the segments
    time_segments = np.linspace(0, T - dt, len(segments_std))  # Time vector of the segments

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(time, x, color = 'darkgray', zorder = 0, label = 'Signal')
    ax.plot(time_segments, mean + segments_std, color = 'C0', zorder = 1, label = 'Segments std')
    ax.hlines(mean + std, 0, T, colors='C1', linestyles='solid', zorder = 2, label = 'Signal std')
    ax.hlines(mean + bound_dw, 0, T, colors='C3', linestyles='dashed', zorder = 3, label = 'Boundaries')
    ax.hlines(mean + bound_up, 0, T, colors='C3', linestyles='dashed', zorder = 4)
    ax.grid()
    ax.legend(loc = 4)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [\]')
    ax.set_title('Index: '+ str(index) + '%\n' + outcome)
    

Reference:

Non-stationarity index in vibration fatigue: Theoretical and experimental research; L. Capponi, M. Česnik, J. Slavič, F. Cianetti, M. Boltežar; International Journal of Fatigue 104, 221-230
https://www.sciencedirect.com/science/article/abs/pii/S014211231730316X
