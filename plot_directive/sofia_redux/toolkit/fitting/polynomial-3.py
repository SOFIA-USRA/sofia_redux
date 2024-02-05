import numpy as np
from sofia_redux.toolkit.fitting.polynomial import Polyfit

# Create a function
x = np.linspace(0, 5, 256)
y = 1 + x
rand = np.random.RandomState(42)

# Add noise
noise = rand.normal(loc=0, scale=0.5, size=y.shape)
y += noise

# Add some obvious outliers, throwing off the fit
inds = rand.randint(0, 255, 25)
y[inds] += 5

# Standard fit
pfit = Polyfit(x, y, 1)

# Robust fit with 3 sigma outlier rejection
rfit = Polyfit(x, y, 1, robust=3)
outliers = np.argwhere(~rfit.mask)[:, 0]

plt.figure(figsize=(5, 5))
plt.plot(x, y, '.', label='Samples')
plt.plot(x, pfit.stats.fit, label='Standard Fit')
plt.plot(x, rfit.stats.fit, label='Robust Fit')
plt.plot(x[outliers], y[outliers], 'x', color='red', label='Outliers')
plt.title("Robust Outlier Rejection")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()

# Display robust statistics
print(rfit)