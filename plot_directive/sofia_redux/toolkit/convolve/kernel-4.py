import imageio
import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.convolve.kernel import SavgolConvolve

# Normalize for plotting
image = imageio.imread('imageio:immunohistochemistry.png').astype(float)
image = image.sum(axis=-1)
image -= image.min()
image /= image.max()

# Add some bad values
rand = np.random.RandomState(41)
inds = rand.choice(image.shape[0], 100), rand.choice(image.shape[1], 100)
image[inds] += 2

mask = np.full(image.shape, True)
mask[inds] = False

s = SavgolConvolve(image, 7, order=3)
s_robust = SavgolConvolve(image, 7, robust=5, order=3)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax[0, 0].imshow(image)
ax[0, 0].set_title("Corrupted original")
ax[0, 1].imshow(s.result)
ax[0, 1].set_title("Standard convolution")
ax[1, 0].imshow(s_robust.result)
ax[1, 0].set_title("Robust convolution")
ax[1, 1].imshow(s_robust.error)
ax[1, 1].set_title("Robust errors")

# Display statistics
print(s_robust)