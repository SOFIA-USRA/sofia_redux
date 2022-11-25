from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

hdul = fits.open('UNKNOWN_FC_IMA_AOR_NONE_RIM_UNKNOWN.fits')
smap = hdul[5].data[200:600, :]
sprof = hdul[6].data[200:600]
x, y = np.meshgrid(np.arange(smap.shape[1]), np.arange(smap.shape[0]))
y += 200

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(x, y, smap, cmap='viridis',
                       antialiased=False, linewidth=0)
ax.set_xlabel('Wavelength\n(column pixels)', fontsize='small')
ax.set_ylabel('Slit position\n(row pixels)', fontsize='small')
ax.set_zlabel('Relative flux', fontsize='small')
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_xaxis()

ax = fig.add_subplot(1, 2, 2)
surf = ax.plot(y, sprof, color='#298289')
ax.set_xlabel('Slit position (row pixels)')
ax.set_ylabel('Relative flux')

fig.suptitle('Spatial Map and Median Profile')

fig.tight_layout()
plt.show()
