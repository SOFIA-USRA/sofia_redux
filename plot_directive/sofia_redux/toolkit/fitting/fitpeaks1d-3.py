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
model_box = get_search_model(
    peak_model, models.Box1D, box_width=('stddev', 3))
model_nobox = get_search_model(peak_model, None)

y_prime, baseline = medabs_baseline(x, y)
x_peak, y_peak = guess_xy_mad(x, y_prime)
tmp = model_box.parameters
tmp[0:3] = y_peak, x_peak, 0.1
model_box.parameters = tmp
tmp = model_nobox.parameters
tmp[0:3] = y_peak, x_peak, 0.1
model_nobox.parameters = tmp

x_peak, y_peak = guess_xy_mad(x, y_prime)
fit_box = dofit(fitter, model_box, x, y_prime)
fit_nobox = dofit(fitter, model_nobox, x, y_prime)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].plot(x, y_prime)
ax[0].plot(x, fit_box(x))
ax[0].set_title("Fit with box function")
ax[0].set_xlabel("x")
ax[0].set_ylabel("$y^{\prime}$")
ax[0].set_xlim(3, 6)

ax[1].plot(x, y_prime)
ax[1].plot(x, fit_nobox(x))
ax[1].set_title("Fit without box function")
ax[1].set_xlabel("x")
ax[1].set_ylabel("$y^{\prime}$")
ax[1].set_xlim(3, 6)