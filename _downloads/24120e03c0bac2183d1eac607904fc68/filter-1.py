import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.convolve.filter import sobel

image = imageio.imread('imageio:page.png')
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].set_title("Image")
ax[0, 1].imshow(sobel(image), cmap='gray_r')
ax[0, 1].set_title("Sobel-Feldman filter p-norm=1")
ax[1, 0].imshow(sobel(image, pnorm=2), cmap='gray_r')
ax[1, 0].set_title("Sobel-Feldman filter p-norm=2")
ax[1, 1].imshow(sobel(image, pnorm=2, kperp=(3, 10, 3)), cmap='gray_r')
ax[1, 1].set_title("Scharr filter p-norm=2")
fig.tight_layout()