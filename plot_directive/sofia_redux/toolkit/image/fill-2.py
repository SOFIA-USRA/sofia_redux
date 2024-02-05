import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

y, x = np.mgrid[:15, :15]
ap = 5
badpix = 7, 7
r = np.hypot(x - badpix[1], y - badpix[0])

data = np.zeros(x.shape)
data[r < ap] = 1
data[badpix[0]:badpix[0] + 5, badpix[1] - 3:badpix[1] + 5] = 2
data[badpix[0], badpix[1]] = 0
data[r >= ap] = 0

goodmask = data == 1
comx = np.mean(x[goodmask])
comy = np.mean(y[goodmask])

radius = plt.Circle(badpix, ap, color='blue', fill=False, linewidth=3,
                    linestyle='--')
radius.set_label("Aperture")

cmap = colors.ListedColormap(['lime', 'white', 'cyan', 'red'])
bounds = [-1, 0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(data, cmap=cmap, norm=norm)

ax.add_artist(radius)
ax.set_xticks(np.arange(x.shape[1]) - 0.5)
ax.set_yticks(np.arange(y.shape[0]) - 0.5)
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

plt.plot([badpix[1], comx], [badpix[0], comy], '-', color='k',
         markersize=8)
plt.plot(badpix[1], badpix[0], 'x', color='k', markersize=10,
         label='Bad pixel')
plt.plot(comx, comy, 'o', color='k', markersize=10,
         label='Center-of-mass')
plt.legend(loc='upper right', framealpha=1)
plt.title("Pixels within aperture radius (masked = red)")