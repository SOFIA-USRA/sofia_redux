import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.image.smooth import bicubic_evaluate

z_corners = np.array([0.0, 1.0, 2.0, 1.0])  # values at corners
dx = np.full(4, 1.0)  # x-gradients at corners
dy = dx.copy()  # y-gradients at corners
dxy = np.zeros(4)  # not present here
xrange = [0, 1]
yrange = [0, 1]

x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
z_new = bicubic_evaluate(z_corners, dx, dy, dxy, xrange, yrange, x, y)
plt.imshow(z_new, origin='lower', cmap='gray', extent=[0, 1, 0, 1])
plt.colorbar()
plt.title("Bicubic Interpolation")
plt.xlabel("x")
plt.ylabel("y")