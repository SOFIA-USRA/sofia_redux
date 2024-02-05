import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.fitting.polynomial import poly1d, polyfitnd

x = np.linspace(0, 4 * np.pi, 1000)
y = np.sin(x)
error = np.random.RandomState(41).normal(loc=0, scale=0.1, size=x.size)
y += error

# use polyfitnd to fit a polynomial and get the covariance on the fit
# coefficients
coeffs, cvar = polyfitnd(x, y, 7, covar=True)
yfit, yvar = poly1d(x, coeffs, covar=cvar)
error = np.sqrt(yvar)

plt.figure(figsize=(7, 5))
plt.plot(x, y, '.', markersize=3, label='data', color='blue')
plt.fill_between(x, yfit - error, yfit + error, color='red',
                 label='$1\sigma$ fit error')
plt.plot(x, yfit, label='fit', color='lime')
plt.legend(loc='lower left')
plt.title("7th order polynomial fit and fit error")
plt.xlabel('x')
plt.ylabel('y')