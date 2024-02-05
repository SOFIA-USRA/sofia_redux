import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.resampling.resample_kernel import ResampleKernel

# Create an irregular kernel
rand = np.random.RandomState(2)
width = 4
x_range = 30
n_impulse = 10
n_kernel = 1000
n_samples = 1000

x = width * (rand.random(n_kernel) - 0.5)
kernel = np.sinc(x * 4) * np.exp(-(x ** 2))

# Add random impulses
impulse_locations = rand.random(n_samples) * x_range
impulses = np.zeros(n_samples)
impulses[:n_impulse] = 1 - 0.5 * rand.random(n_impulse)

resampler = ResampleKernel(impulse_locations, impulses, kernel,
                           kernel_offsets=x[None], smoothing=1e-6)

x_out = np.linspace(0, x_range, 500)
fit = resampler(x_out, normalize=False)

plt.plot(x_out, fit, label='fit')
plt.vlines(impulse_locations[:n_impulse], 0, impulses[:n_impulse],
           linestyles='dashed', colors='r', linewidth=1)
plt.plot(impulse_locations[:n_impulse], impulses[:n_impulse], 'x',
         color='r', label='impulses')
plt.legend()
plt.title('A set of impulse signals convolved with an irregular kernel.')