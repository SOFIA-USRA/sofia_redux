import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sofia_redux.toolkit.interpolate.interpolate import Interpolate

x = np.linspace(-2, 2, 10)
y = x.copy()
xx, yy = np.meshgrid(x, y)
zz = np.exp(-((xx ** 2) + (yy ** 2)))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5),
                       subplot_kw={'projection': '3d'})
fig.tight_layout()
xout = np.linspace(-2, 2, 50)
yout = xout.copy()
xxout, yyout = np.meshgrid(xout, yout)
cmap = 'nipy_spectral'

interpolator = Interpolate(x, y, zz, method='nearest')
z_nearest = interpolator(xxout, yyout, mode='nearest')
ax[0].set_title("Nearest")
ax[0].plot_surface(xxout, yyout, z_nearest, cmap=cmap,
                   rstride=1, cstride=1, linewidth=0)

interpolator = Interpolate(x, y, zz, method='linear')
z_linear = interpolator(xxout, yyout, mode='nearest')
ax[1].set_title("Linear")
ax[1].plot_surface(xxout, yyout, z_linear, cmap=cmap,
                   rstride=1, cstride=1, linewidth=0)

interpolator = Interpolate(x, y, zz, method='cubic')
z_cubic = interpolator(xxout, yyout, mode='nearest')
ax[2].set_title("Cubic")
ax[2].plot_surface(xxout, yyout, z_cubic, cmap=cmap,
                   rstride=1, cstride=1, linewidth=0)