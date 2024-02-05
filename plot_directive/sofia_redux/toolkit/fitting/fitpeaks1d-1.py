import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D
from sofia_redux.toolkit.fitting.fitpeaks1d import medabs_baseline

x = np.linspace(0, 1, 512)
y = (x / 4) + 2
y += Gaussian1D(stddev=0.05, mean=0.2)(x)
y -= Gaussian1D(stddev=0.1, mean=0.7)(x)
y_prime, baseline = medabs_baseline(x, y)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].plot(x, y, label="Data")
ax[0].plot(x, baseline, label="Baseline")
ax[0].set_title("Unprocessed Data")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].legend()
ax[1].plot(x, y_prime)
ax[1].set_title("Processed Data")
ax[1].set_xlabel("X")
ax[1].set_ylabel("$Y^{\prime}$")