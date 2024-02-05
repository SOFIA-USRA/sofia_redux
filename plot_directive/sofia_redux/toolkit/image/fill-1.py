import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.image.fill import image_naninterp
from sofia_redux.toolkit.utilities.func import julia_fractal
from astropy.modeling.models import Gaussian2D

image = julia_fractal(300, 300)
g = Gaussian2D(x_mean=250, y_mean=250, x_stddev=10, y_stddev=10)
image += g(*np.meshgrid(np.arange(300), np.arange(300)))
original = image.copy()

# add a few single pixel holes
rand = np.random.RandomState(41)
mask = rand.rand(*image.shape[:2]) < 0.1
image[mask] = np.nan

# add some larger holes
image[210:235, 0:25] = np.nan  # edge example
image[100:140, 160:200] = np.nan  # large structure example
image[230:255, 230:255] = np.nan  # smooth example
bad = image.copy()
image = np.clip(image_naninterp(image), 0, 1)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
c = 'cubehelix' #'jet'
ax[0].imshow(original, cmap=c)
ax[0].set_title("Original Image")
ax[1].imshow(bad, cmap=c)
ax[1].set_title("Holey Image")
ax[2].imshow(image, cmap=c)
ax[2].set_title("Corrected Image (image_naninterp)")