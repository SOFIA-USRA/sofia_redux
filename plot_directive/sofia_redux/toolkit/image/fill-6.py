from sofia_redux.toolkit.image.fill import polyfillaa
import numpy as np
import matplotlib.pyplot as plt

px = [[0, 0, 2, 2], [4, 2, 8], [10, 10, 12, 12]]
py = [[0, 2, 2, 0], [2, 9, 5], [4,  5,  5,  4]]

# Add a pentagon
def mypoly(x, y, r, n):
    ang = (np.arange(n) + 1) * 2 * np.pi / n
    return list(r * np.cos(ang) + x), list(r * np.sin(ang) + y)

hx, hy = mypoly(10.5, 10.5, 3, 5)
px.append(hx)
py.append(hy)

result, areas = polyfillaa(px, py, area=True)

grid = np.full((15, 15), np.nan)
for i in range(len(result.keys())):
    grid[result[i][:,0], result[i][:,1]] = areas[i]

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xticks(np.arange(-0.5, 15, 1))
ax.set_yticks(np.arange(-0.5, 15, 1))
ax.set_xticklabels(np.arange(16))
ax.set_yticklabels(np.arange(16))

ax.grid(which='major', axis='both', linestyle='--',
        color='k', linewidth=1)
img = ax.imshow(grid, cmap='cividis', origin='lower')

for i in range(len(px)):
    x = px[i] + [px[i][0]]
    y = py[i] + [py[i][0]]
    ax.plot(np.array(x) - 0.5, np.array(y) - 0.5,
            '-o', color='red', linewidth=3, markersize=7)

fig.colorbar(img, ax=ax)
ax.set_title("Multiple polygons and fractional area")