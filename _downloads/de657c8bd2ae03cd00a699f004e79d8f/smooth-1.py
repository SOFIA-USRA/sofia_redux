from sofia_redux.toolkit.image.smooth import fiterpolate
import matplotlib.pyplot as plt
import imageio

image = imageio.imread('imageio:camera.png').astype(float)
image -= image.min()
image /= image.max()

smoothed = fiterpolate(image, 32, 32)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original image")
ax[1].imshow(smoothed, cmap='gray')
ax[1].set_title("Image smoothed with fiterpolate (32 x 32) grid")