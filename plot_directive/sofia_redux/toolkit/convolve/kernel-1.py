import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.convolve.kernel import BoxConvolve

image = imageio.imread('imageio:coffee.png').sum(axis=2)  # Gray scale
image = (image - image.min()) / (np.ptp(image))  # normalize for plotting
mean_smooth = BoxConvolve(image, 11)  # an 11 x 11 box filter

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.tight_layout()
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(mean_smooth.result, cmap='gray')
ax[1].set_title("Smoothed Image")