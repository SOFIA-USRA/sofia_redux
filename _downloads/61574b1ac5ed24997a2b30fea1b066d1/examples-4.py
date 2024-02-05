from sofia_redux.toolkit.resampling import Resample
from astropy.stats import gaussian_fwhm_to_sigma
import imageio
import matplotlib.pyplot as plt
import numpy as np

# Cut out a section of the image for analysis
image = imageio.imread('imageio:hubble_deep_field.png')
image = image[325:475, 45:195].sum(axis=-1).astype(float)
image -= image.min()
image /= image.max()
y, x = np.mgrid[:image.shape[0], :image.shape[1]]
coordinates = np.vstack([c.ravel() for c in [x, y]])

# blow up center of the image leaving a border to avoid edge effects
xout = np.linspace(25, 125, 300)
yout = np.linspace(25, 125, 300)

resampler = Resample(coordinates, image.ravel(), order=2, window=9,
                     error=1e-3)

sigma = gaussian_fwhm_to_sigma * 3

low, low_weights = resampler(xout, yout,
                             smoothing=3 * sigma,
                             get_distance_weights=True,
                             order_algorithm='extrapolate', jobs=-1)

high, high_weights = resampler(xout, yout, smoothing=sigma / 3,
                               get_distance_weights=True,
                               order_algorithm='extrapolate', jobs=-1)

adaptive, adaptive_weights = resampler(xout, yout,
                                       smoothing=sigma,
                                       adaptive_threshold=1,
                                       get_distance_weights=True,
                                       order_algorithm='extrapolate',
                                       jobs=-1)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for ax in axs.ravel():
    ax.axis('off')
fig.subplots_adjust(left=0.05, right=0.95,
                    wspace=0.25, hspace=0.01,
                    top=0.95, bottom=0.05)
color = 'twilight_shifted'

low_img = axs[0, 0].imshow(low, cmap=color)
axs[0, 0].title.set_text("Standard Fitting")
fig.colorbar(low_img, ax=axs[0, 0], fraction=0.046, pad=0.04)

high_img = axs[0, 1].imshow(high, cmap=color)
axs[0, 1].title.set_text("Over-fitting")
fig.colorbar(high_img, ax=axs[0, 1], fraction=0.046, pad=0.04)

adapt_img = axs[0, 2].imshow(adaptive, cmap=color)
axs[0, 2].title.set_text("Adaptive Fitting")
fig.colorbar(adapt_img, ax=axs[0, 2], fraction=0.046, pad=0.04)

wlow_img = axs[1, 0].imshow(low_weights, cmap=color)
axs[1, 0].title.set_text("Standard Fit Weights")
fig.colorbar(wlow_img, ax=axs[1, 0], fraction=0.046, pad=0.04,
             format='%.3f')

whigh_img = axs[1, 1].imshow(high_weights, cmap=color)
axs[1, 1].title.set_text("Over-fitting Weights")
fig.colorbar(whigh_img, ax=axs[1, 1], fraction=0.046, pad=0.04,
             format='%.3f')

wadapt_img = axs[1, 2].imshow(adaptive_weights, cmap=color)
axs[1, 2].title.set_text("Adaptive Fitting Weights")
fig.colorbar(wadapt_img, ax=axs[1, 2], fraction=0.046, pad=0.04,
             format='%.3f')