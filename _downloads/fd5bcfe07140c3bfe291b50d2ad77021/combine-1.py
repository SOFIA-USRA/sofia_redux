import matplotlib.pyplot as plt
import imageio
import numpy as np
from sofia_redux.toolkit.image.combine import combine_images
from mpl_toolkits.axes_grid1 import make_axes_locatable

images = imageio.imread('imageio:stent.npz').astype(float)
sum_image, variance = combine_images(images, method='sum')
error = np.sqrt(variance)
mean_image = combine_images(images, method='mean', returned=False)
med_image = combine_images(images, method='median', returned=False)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
titles = ['Sum of Images', 'Mean of Images', 'Median of Images', 'Error']
imgs = [sum_image, mean_image, med_image, error]
for i, (ax, img, title) in enumerate(zip(axs.flatten(), imgs, titles)):
    img2 = ax.imshow(img, cmap='gray')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img2, cax=cax)

plt.tight_layout()