import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.resampling.resample_kernel import ResampleKernel

def mexican_hat(x, y, period=1):
    r = np.sqrt(x ** 2 + y ** 2 + np.finfo(float).eps)
    rs = r * 2 * np.pi / period
    result = np.sin(rs) / rs
    return result

# Create a set of random kernel coordinates and values
rand = np.random.RandomState(0)
width = 6
w2 = width / 2

kx = width * (rand.random(1000) - 0.5)
ky = width * (rand.random(1000) - 0.5)
kernel = mexican_hat(kx, ky, period=w2)
kernel_offsets = np.stack([kx, ky])

# First create a representation of the kernel on a grid by convolving
# with a delta function.
xx, yy = np.meshgrid(np.linspace(-w2, w2, 101), np.linspace(-w2, w2, 101))
cc = np.stack([xx.ravel(), yy.ravel()])
delta = np.zeros_like(xx)
delta[50, 50] = 1

resampler = ResampleKernel(cc, delta.ravel(), kernel, degrees=3,
                           smoothing=1e-5, kernel_offsets=kernel_offsets)

regular_kernel = resampler(cc, jobs=-1, normalize=False).reshape(
    delta.shape)

# Now show an example of edge detection using the irregular kernel
image = imageio.imread('imageio:camera.png').astype(float)
image -= image.min()
image /= image.max()

ny, nx = image.shape
iy, ix = np.mgrid[:ny, :nx]
coordinates = np.stack([ix.ravel(), iy.ravel()])
data = image.ravel()

resampler = ResampleKernel(coordinates, data, kernel, degrees=3,
                           smoothing=1e-3, kernel_offsets=kernel_offsets)
edges = abs(resampler(coordinates, jobs=-1, normalize=False)).reshape(
    image.shape)

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original image')
ax2.imshow(regular_kernel, interpolation='none', extent=[-w2, w2, -w2, w2])
ax2.set_title('Interpolated regular kernel')
ax3.imshow(edges, cmap='gray')
ax3.set_title('Irregular kernel convolved with image')