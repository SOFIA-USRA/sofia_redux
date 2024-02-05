import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D
from sofia_redux.toolkit.fitting.fitpeaks1d import medabs_baseline, guess_xy_mad

x = np.linspace(0, 1, 512)
y = (x / 4) + 2
y += Gaussian1D(stddev=0.05, mean=0.2)(x)
y -= Gaussian1D(stddev=0.1, mean=0.7)(x)
y_prime, baseline = medabs_baseline(x, y)
x_peak, y_peak = guess_xy_mad(x, y_prime)

plt.figure(figsize=(5, 5))
plt.plot(x, y_prime)
plt.plot(x_peak, y_peak, 'x', color='red', markersize=10,
         label="$(x_{peak}, y_{peak})$")
plt.title("Most prominent peak")
plt.xlabel("x")
plt.ylabel("$y^{\prime}$")