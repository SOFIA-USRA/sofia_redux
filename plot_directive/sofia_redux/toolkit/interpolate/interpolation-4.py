from sofia_redux.toolkit.interpolate import sincinterp
from scipy import signal
import numpy as np

x = np.linspace(0, 10, 20, endpoint=False)
y = np.cos(-x ** 2 / 6.0)
xout = np.linspace(0, 10, 100, endpoint=False)
yout = sincinterp(x, y, xout)
truth = np.cos(-xout ** 2 / 6)
scipy_try = signal.resample(y, 100)
plt.figure(figsize=(5, 5))
plt.plot(x, y, 'x', xout, yout, '-',
         xout, truth, ':', xout, scipy_try,'--')
plt.legend(['data', 'sincinterp', 'truth', 'scipy'],
           loc='lower left')
plt.title("Sinc Upsampling")