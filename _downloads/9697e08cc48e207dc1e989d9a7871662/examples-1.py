import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.resampling import Resample

rand = np.random.RandomState(100)
noise = rand.rand(100) - 0.5
x = np.linspace(-np.pi, np.pi, 100)
ytrue = np.sin(x)
y = ytrue + noise
xout = np.linspace(-np.pi, np.pi, 1000)

narrow_resampler = Resample(x, y, order=2)
yfit = narrow_resampler(xout)
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x, y, '.', label='Samples')
plt.plot(xout, yfit, label='Fit')
plt.plot(x, ytrue, '--', label='Truth')
plt.legend()
plt.title("Fit using minimal window")

wide_resampler = Resample(x, y, window=np.pi / 2, order=2)
yfit2 = wide_resampler(xout, smoothing=0.4, relative_smooth=True)
plt.subplot(122)
plt.plot(x, y, '.', label='Samples')
plt.plot(xout, yfit2, label='Fit')
plt.plot(x, ytrue, '--', label='Truth')
plt.legend()
plt.title("Fit using wide window with distance weighting")