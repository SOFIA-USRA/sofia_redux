import matplotlib.pyplot as plt
import numpy as np
import imageio
from sofia_redux.toolkit.image.fill import maskinterp

image = imageio.imread('imageio:camera.png').astype(float)
image /= image.max()
original = image.copy()
rand = np.random.RandomState(41)
badpix = rand.rand(100, 100) > 0.5
cut = image[75:175, 180:280]
cut[badpix] = np.nan

result = maskinterp(image, kx=2, ky=2, minpoints=9)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
c = 'copper'
ax[0].imshow(original, cmap=c)
ax[0].set_title("Original Image")
ax[1].imshow(image, cmap=c)
ax[1].set_title("Holey Image")
ax[2].imshow(result, cmap=c)
ax[2].set_title("Corrected Image (maskinterp)")
for a in ax:
    a.set_xlim(165, 295)
    a.set_ylim(190, 60)