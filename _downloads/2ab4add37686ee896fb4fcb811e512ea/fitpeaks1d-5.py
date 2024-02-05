import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from sofia_redux.toolkit.fitting.fitpeaks1d import (get_search_model, initial_search,
                                     dofit, get_fitter, get_background_fit)

x = np.linspace(0, 10, 1001)
model1 = models.Gaussian1D(amplitude=3, mean=5, stddev=0.5)
model2 = models.Gaussian1D(amplitude=4, mean=4, stddev=0.2)
y = model1(x) + model2(x) + 0.05 * x + 100
rand = np.random.RandomState(42)
y += rand.normal(0, 0.01, y.size)

peak_model = models.Gaussian1D()
fitter = get_fitter(fitting.LevMarLSQFitter)
search_model = get_search_model(
    peak_model, models.Box1D, box_width=('stddev', 3), stddev_0=0.1)
pinit = initial_search(fitter, search_model, x, y, npeaks=2)
binit = get_background_fit(fitter, search_model[0],
                           models.Linear1D, x, y, pinit)

residual = y.copy()
first_fit = np.zeros_like(residual)
for params in pinit:
    peak_model.parameters = params
    peak_fit = peak_model(x)
    residual -= peak_fit
    first_fit += peak_fit

baseline_model = models.Linear1D()
baseline_model.parameters = binit
first_fit += baseline_model(x)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax[0].plot(x, residual, label='Residual = data - peaks')
ax[0].plot(x, baseline_model(x), label='Initial baseline fit')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title("Initial Baseline Fit")
ax[1].plot(x, y, label='Data')
ax[1].plot(x, first_fit, label='Initial Fit')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title("Full Initial Fit")
ax[1].legend()
ax[2].plot(x, y - first_fit)
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].set_title("Residual of the Initial Fit")