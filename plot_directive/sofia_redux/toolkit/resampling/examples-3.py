from sofia_redux.toolkit.resampling import Resample
import imageio
import numpy as np

image = imageio.imread('imageio:coins.png')
image = image / image.max()
image = image[12:76, 303:367]  # 64 x 64 pixel image

y, x = np.mgrid[:image.shape[0], :image.shape[1]]
coordinates = np.vstack([c.ravel() for c in [x, y]])

# default resampler
resampler = Resample(coordinates, image.ravel(), order=3)

# blow up by a factor of 5
xout = np.linspace(0, 63, 64 * 5)
yout = np.linspace(0, 63, 64 * 5)

# Default uses "bounded" order algorithm
bounded_mode = resampler(xout, yout, smoothing=0.05,
                         relative_smooth=True, order_algorithm='bounded',
                         jobs=-1)
extrap_mode = resampler(xout, yout, smoothing=0.05,
                        relative_smooth=True,
                        order_algorithm='extrapolate', jobs=-1)
com_edges = resampler(xout, yout, smoothing=0.05,
                      order_algorithm='extrapolate',
                      edge_threshold=0.8, relative_smooth=True, jobs=-1)

plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title("Original image")
plt.imshow(image, cmap='copper')
plt.subplot(222)
plt.imshow(bounded_mode, cmap='copper')
plt.title("'edges' order mode")
plt.subplot(223)
plt.imshow(extrap_mode, cmap='copper')
plt.title("'extrapolate' order mode")
plt.subplot(224)
plt.imshow(com_edges, cmap='copper')
plt.title("'com_distance' edge mode")