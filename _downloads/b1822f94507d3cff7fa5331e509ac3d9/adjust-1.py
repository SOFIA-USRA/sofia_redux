import matplotlib.pyplot as plt
import numpy as np
from sofia_redux.toolkit.image.adjust import shift

image = np.repeat(np.arange(5, dtype=float)[None], 3, axis=0)
image[1, 2] = np.nan

# shift the image by 0.75 pixels in the x-direction
offset = [0, 0.75]

default = shift(image, offset, nan_interpolation=None)
no_fractional_nans = shift(image, offset, missing_limit=0,
                           nan_interpolation=None)
replace_nans = shift(image, offset, missing=None, mode='nearest',
                     nan_interpolation=None)

# plot results
plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.05, right=0.97, bottom=0.03, top=0.97,
                    wspace=0.15)

plt.subplot(221)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(222)
plt.imshow(default)
plt.title("Shifted using default settings")

plt.subplot(223)
plt.imshow(no_fractional_nans)
plt.title("Shifted disallowing partial NaNs")

plt.subplot(224)
plt.imshow(replace_nans)
plt.title("Shifted, replacing NaNs")