from sofia_redux.toolkit.interpolate import interp_error
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
error = np.ones(10)
points_out = np.linspace(0, 3, 301)
i_error_1d = interp_error(x, error, points_out)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.tight_layout()

ax[0].plot(points_out, i_error_1d)
ax[0].set_title("1-D Linear Interpolation Error")

y, x = np.mgrid[:4, :4]
error = np.full(x.size, 1.0)
points = np.stack((x.ravel(), y.ravel())).T
xg = np.linspace(2.1, 2.9, 101)
points_out = np.array([x.ravel() for x in np.meshgrid(xg, xg)]).T
i_error = interp_error(points, error, points_out)
ax[1].imshow(i_error.reshape((101, 101)))
ax[1].set_title("Delaunay triangulation error inside single grid cell")