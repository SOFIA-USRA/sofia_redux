import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.convolve.kernel import SavgolConvolve

x = np.linspace(0, 4 * np.pi, 512)
y = np.sin(x)
error = 0.2
rand = np.random.RandomState(41)
noise = rand.normal(loc=0.0, scale=error, size=x.size)
y += noise

# Add NaN Values
y[300:350] = np.nan

# Mask certain values to be excluded from fit, but interpolated over
mask = np.full(x.size, True)
mask[75:175] = False

width = 31
s = SavgolConvolve(x, y, width, error=error, mask=mask, order=2)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
black_plot = s.result.copy()
black_plot[~mask] = np.nan
black_plot[np.isnan(y)] = np.nan
ax[0].set_title("NaN and Masked Value Handling")
ax[0].plot(x, y, '.', markersize=3)
ax[0].plot(x, black_plot, color='black',
           label='Standard')
ax[0].plot(x[300:350], s.result[300:350], '--', color='red',
           label='NaN handling')
ax[0].plot(x[75:175], s.result[75:175], '--', color='magenta',
           label='Mask handling')
ax[0].legend(loc='lower left')
ax[0].set_ylim(-2, 1.5)

ax[1].set_title("Error Propagation")
ax[1].plot(x, s.error)
ax[1].set_xlabel("X")
ax[1].set_ylabel("Error in Convolved values")
ax[0].set_ylabel("Y")