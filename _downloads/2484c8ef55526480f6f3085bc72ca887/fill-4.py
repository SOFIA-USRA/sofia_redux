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

median_filled = maskinterp(image, func=np.median, statistical=True)
plt.figure(figsize=(5, 5))
plt.imshow(median_filled[60:190, 165:295], cmap='copper')
plt.title("Maskinterp with user defined median function")