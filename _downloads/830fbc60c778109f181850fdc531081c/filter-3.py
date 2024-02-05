import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.convolve.filter import savgol

x, y, z = np.mgrid[-70:70, -70:70, -70:70]
d = np.cos((x ** 2 + x ** 2 + z ** 2) / 200)
rand = np.random.RandomState(41)
d += rand.normal(size=x.shape)

result = savgol(d, 7, order=2, mode='constant', cval=np.nan)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
c = 'gist_stern'
ax[0, 0].set_title("x-section of original data along axis 0")
ax[0, 0].imshow(d[40], cmap=c)
ax[0, 1].set_title("x-section of original data along axis 1")
ax[0, 1].imshow(d[:, 40], cmap=c)
ax[1, 0].set_title("x-section of filtered data along axis 0")
ax[1, 0].imshow(result[40], cmap=c)
ax[1, 1].set_title("x-section of filtered data along axis 1")
ax[1, 1].imshow(result[:, 40], cmap=c)
fig.tight_layout()