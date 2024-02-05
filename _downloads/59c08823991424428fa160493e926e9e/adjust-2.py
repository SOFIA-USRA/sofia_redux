import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.image.adjust import shift, rotate

image = imageio.imread('imageio:camera.png').astype(float)
image /= image.max()

# pad so we don't cut off information during rotation
image = np.pad(image, 100, mode='constant', constant_values=np.nan)

# insert some NaN values (off with his head)
image[170:260, 300:370] = np.nan

# rotate then shift the image
pivot = [600, 100]  # rotate around this pixel

# rotate by 10 degrees around `pivot`
# set `missing_limit` to zero to skip NaN replacement by interpolation,
# and simply disallow any pixels containing part of an original NaN
# pixel.
image_rotated = rotate(image, 10.0, missing_limit=0, pivot=pivot)

# shift by 100 pixels in x and 145.5 pixels in y
image_shifted = shift(image_rotated, [145.5, 100], missing_limit=0)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title("Original Padded Image")

plt.subplot(132)
plt.imshow(image_rotated, cmap='gray')
plt.title("Rotated Image")

plt.subplot(133)
plt.imshow(image_shifted, cmap='gray')
plt.title("Rotated and Shifted Image")