import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.splines.spline import Spline

image = imageio.imread('imageio:coins.png')
noisy_image = image[165:225,70:130].astype(float)
noisy_image -= noisy_image.min()
noisy_image /= noisy_image.max()

# Create spline representations of the image at different smoothing levels
splines = []
for smoothing in [100, 25, 10]:
    splines.append(Spline(noisy_image, degrees=3, smoothing=smoothing))

# Create a finer grid
ny, nx = noisy_image.shape
x = np.linspace(0, nx - 1, nx * 3)
y = np.linspace(0, ny - 1, ny * 3)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(noisy_image, cmap='gray')
ax1.set_title('Original image')

for i, axis in enumerate([ax2, ax3, ax4]):
    axis.imshow(splines[i](x, y), cmap='gray')
    axis.set_title(f'smoothing={splines[i].smoothing}, '
                   f'ssr=%.5f' % splines[i].sum_square_residual)
    kx = np.unique(splines[i].knots[0][:splines[i].n_knots[0]]) * 3
    ky = np.unique(splines[i].knots[1][:splines[i].n_knots[1]]) * 3
    kg = np.meshgrid(kx, ky)
    ky, kx = kg[0].ravel(), kg[1].ravel()
    axis.plot(kx, ky, '.', color='r', markersize=4 / (i + 1))