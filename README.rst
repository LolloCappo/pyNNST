Index of non-stationarity
---------------------------------------------

Obtaining non-stationary index for time-series.

Installing this package
-----------------------

Use `pip` to install it by:

.. code-block:: console

    $ pip install pyNNST


Simple examples
---------------

Here is a simple example on how to use the code:

.. code-block:: python

    # Import packages 
    import pyNNST
    import numpy as np

    # Define a sample signal x
    T = 20                                # Time length of x
    fs = 400                              # Sampling frequency of x
    dt = 1 / fs                           # Time between discreete signal values
    x = np.random.rand(T * fs)            # Signal
    time = np.linspace(0, T - dt, T * fs) # Time vector
    std = np.std(x, ddof = 1)             # Standard deviation of x
    mean = np.mean(x)                     # Mean value of x

    # Class initialization
    example = pyNNST.nnst(x, nperseg = 100, noverlap = 0, confidence = 95)
    
    # Compute the run test for non-stationarity
    example.idns() 
    outcome = example.get_outcome()  # Get the results of the test as a string
    index = example.get_index()      # Get the index of non-stationarity
    limits = example.get_limits()    # Get the limits outside of which the signal is non-stationary


Reference:

Non-stationarity index in vibration fatigue: Theoretical and experimental research; L. Capponi, M. Cesnik, J. Slavic, F. Cianetti, M. Boltezar; International Journal of Fatigue 104, 221-230
https://www.sciencedirect.com/science/article/abs/pii/S014211231730316X
