import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
from sofia_redux.toolkit.fitting.fitpeaks1d import fitpeaks1d
from astropy.modeling.polynomial import Polynomial1D

x = np.linspace(0, 10, 1000)
baseline = 10 - (x - 5) ** 2
positive_source = models.Voigt1D(x_0=4, amplitude_L=5, fwhm_G=0.1)
negative_source = models.Voigt1D(x_0=7, amplitude_L=-5, fwhm_G=0.2)
rand = np.random.RandomState(41)
noise = rand.normal(loc=0, scale=1.5, size=x.size)
y = baseline + positive_source(x) + negative_source(x) + noise

def baseline_func(x, y):
    baseline = np.poly1d(np.polyfit(x, y, 2))(x)
    return np.abs(y - baseline), baseline

model = fitpeaks1d(x, y, npeaks=2,
                   peak_class=models.Voigt1D,
                   box_width=(None, 3),
                   background_class=Polynomial1D(2),
                   baseline_func=baseline_func)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].plot(x, y, '.', label='Data', color='blue', markersize=2)
ax[0].plot(x, model(x), label='Fit', color='red')
ax[0].plot(x, model[2](x), '--', label='Baseline', color='green',
           linewidth=3)
ax[0].legend(loc='lower center')
ax[0].set_title("Fit to data with large baseline structure")
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].plot(x, (model[0] + model[1])(x), label='Fitted peaks')
ax[1].plot(x, positive_source(x) + negative_source(x), label='True peaks')
ax[1].legend(loc='lower left')
ax[1].set_title("Identified peaks")
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
print(model)