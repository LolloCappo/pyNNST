import numpy as np
import matplotlib.pyplot as plt
from pyNNST import *


T = 20
fs = 400
dt = 1 / fs
x = np.random.rand(T * fs)
time = np.linspace(0, T - dt, T * fs)
std = np.std(x, ddof = 1) 
mean = np.mean(x) 

example = nnst(x, nperseg = 2, noverlap = 0, confidence = 95)
example.idns()

segments_std, bound_dw, bound_up = example.get_segments()
time_segments = np.linspace(0, T - dt, len(segments_std))

limits = example.get_limits()
outcome = example.get_outcome()
index = example.get_index()

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
plt.show()
