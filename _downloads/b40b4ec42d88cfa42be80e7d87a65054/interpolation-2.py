import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.interpolate import Interpolate

y = np.arange(10).astype(float)
x = np.arange(10).astype(float)
x1 = np.linspace(-1, 2, 51)

interpolator = Interpolate(x, y, method='cubic', mode='nearest')
y_nearest = interpolator(x1)

interpolator = Interpolate(x, y, method='cubic', cval=1)
y_constant = interpolator(x1)

interpolator = Interpolate(x, y, method='cubic', mode='wrap')
y_wrap = interpolator(x1)

plt.figure(figsize=(5, 5))
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Edge Modes")
plt.plot(x, y, 's', color='k', linewidth=10, label='samples')
plt.plot(x1, y_nearest, label='nearest')
plt.plot(x1, y_constant, label='constant=1')
plt.plot(x1, y_wrap, label='wrap')
plt.legend()