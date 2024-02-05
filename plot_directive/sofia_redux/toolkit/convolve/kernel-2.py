import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.convolve.kernel import KernelConvolve

image = imageio.imread('imageio:coffee.png').sum(axis=2)  # Gray scale
image = (image - image.min()) / (np.ptp(image))  # normalize for plotting

# Create a Sobel filter
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1.]])

# Edges in 1 direction
sx = KernelConvolve(image, sobel, normalize=False).result

# Edges in the other
sy = KernelConvolve(image, sobel.T, normalize=False).result

# Edge amplitude
sxy = np.hypot(sx, sy)

plt.figure(figsize=(5, 5))
plt.imshow(sxy, cmap='gray_r')
plt.title("Applying User Defined Kernel")