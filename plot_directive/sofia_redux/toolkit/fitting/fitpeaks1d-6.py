import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from sofia_redux.toolkit.fitting.fitpeaks1d import fitpeaks1d

# Create some fake data with noise
x = np.linspace(0, 10, 1001)
model1 = models.Gaussian1D(amplitude=3, mean=5, stddev=0.5)
model2 = models.Gaussian1D(amplitude=4, mean=4, stddev=0.2)
y = model1(x) + model2(x) + 0.05 * x + 100
rand = np.random.RandomState(42)
y += rand.normal(0, 0.01, y.size)

model = fitpeaks1d(x, y, npeaks=2, background_class=models.Linear1D,
                   box_width=('stddev', 3),
                   search_kwargs={'stddev_0': 0.1})

fit = model(x)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].plot(x, y, label='Data', color='blue')
ax[0].plot(x, model(x), label='Model', color='red')
ax[0].legend()
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title("Data and Fit")
ax[1].plot(x, y - fit)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title("Residual")