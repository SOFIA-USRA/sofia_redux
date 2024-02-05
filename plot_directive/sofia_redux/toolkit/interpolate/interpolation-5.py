from sofia_redux.toolkit.interpolate import spline
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(7).astype(float)
y = (-1) ** x

tensions = [1, 10, 100]
xout = np.linspace(x.min(), x.max(), np.ptp(x.astype(int) * 20))
fits = np.zeros((len(tensions), xout.size))
for i, sigma in enumerate(tensions):
    fits[i] = spline(x, y, xout, sigma=sigma)

plt.figure(figsize=(5, 5))
plt.plot(x, y, 'x', color='k')
for i in range(len(tensions)):
    plt.plot(xout, fits[i])
plt.title("Tensioned Splines")
legend = ['Samples']
legend += ['sigma = %s' % tensions[i] for i in range(len(tensions))]
plt.legend(legend, loc='upper right')