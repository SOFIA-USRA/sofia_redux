import matplotlib.pyplot as plt
import numpy as np
from sofia_redux.toolkit.image.fill import polyfillaa

# Define a polygon
px = [5, 3, 9]
py = [2, 10, 5]

pixels, areas = polyfillaa(px, py, area=True)
grid = np.full((11, 11), np.nan)
grid[pixels[:, 0], pixels[:, 1]] = areas

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xticks(np.arange(-0.5, 10, 1))
ax.set_yticks(np.arange(-0.5, 10, 1))
ax.set_xticklabels(np.arange(11))
ax.set_yticklabels(np.arange(11))

ax.grid(which='major', axis='both', linestyle='--',
        color='k', linewidth=1)
img = ax.imshow(grid, cmap='cividis', origin='lower')

ax.plot(np.array(px + [px[0]]) - 0.5, np.array(py + [py[0]]) - 0.5,
        '-o', color='red', linewidth=3, markersize=10)
fig.colorbar(img, ax=ax)
ax.set_title("Pixels within polygon and fractional area")