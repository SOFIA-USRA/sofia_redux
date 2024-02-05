import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from sofia_redux.toolkit.fitting.fitpeaks1d import (medabs_baseline, get_search_model,
                                          guess_xy_mad, dofit, get_fitter)

x = np.linspace(0, 10, 1001)
model1 = models.Gaussian1D(amplitude=3, mean=5, stddev=0.5)
model2 = models.Gaussian1D(amplitude=4, mean=4, stddev=0.2)
y = model1(x) + model2(x) + 0.05 * x + 100
rand = np.random.RandomState(42)
y += rand.normal(0, 0.01, y.size)

peak_model = models.Gaussian1D
fitter = get_fitter(fitting.LevMarLSQFitter)
search_model = get_search_model(
    peak_model, models.Box1D, box_width=('stddev', 3),
    stddev_0=0.1)

y_prime, baseline = medabs_baseline(x, y)
x_peak, y_peak = guess_xy_mad(x, y_prime)
tmp = search_model.parameters
tmp[0:2] = y_peak, x_peak
search_model.parameters = tmp

x_peak, y_peak = guess_xy_mad(x, y_prime)
fit = dofit(fitter, search_model, x, y_prime)
y_prime -= fit(x)
x_peak, y_peak = guess_xy_mad(x, y_prime)

tmp = search_model.parameters
tmp[0:2] = y_peak, x_peak
search_model.parameters = tmp
fit = dofit(fitter, search_model, x, y_prime)

plt.figure(figsize=(7, 5))
plt.plot(x, y_prime, color='blue',
         label='$y^{\prime}$ with 1st peak removed')
plt.plot(x, fit(x), color='limegreen',
         label='Fit to 2nd peak')
plt.plot(x_peak, y_peak, 'x', color='red', markersize=10,
         label='2nd peak guess position')
plt.xlabel('x')
plt.ylabel('$y^{\prime}$')
plt.title('Second peak initial parameterization')
plt.legend()