import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.fitting.polynomial import Polyfit

# Create function
x = np.linspace(0, 2, 256)
y = -0.5 + (x - 2) ** 2

# add noise with a normal distribution
rand = np.random.RandomState(42)
noise = rand.normal(loc=0, scale=0.5, size=x.size)
y += noise

# Fit a 2nd order polynomial to the noisy data
# Since we know the scale of the error, it may be included
pfit = Polyfit(x, y, 2, error=0.5)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.tight_layout()
ax[0].plot(x, y, '.', label="Samples")
ax[0].plot(x, pfit.stats.fit, '-', label="Fit")
ax[0].legend()
ax[0].set_title("Polynomial fit to noisy data")
ax[1].plot(x, pfit.stats.residuals, '.')
ax[1].set_title("Residuals to the fit")

print(pfit)