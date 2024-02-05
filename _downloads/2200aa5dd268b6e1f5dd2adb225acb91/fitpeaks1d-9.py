import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.fitting.fitpeaks1d import fitpeaks1d

image = imageio.imread('imageio:hubble_deep_field.png')
y = image[400].sum(axis=1).astype(float)
x = np.arange(y.size)
model = fitpeaks1d(x, y, npeaks=10)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax[0].plot(x, y)
background = model[-1].amplitude

for i in range(10):
    px, py = model[i].mean.value, model[i].amplitude.value + background
    ax[0].plot(px, py, 'x',
             markersize=10, color='red')
    ax[0].annotate(str(i + 1), (px - 40, py))

ax[0].legend(['Data', 'Fitted peak'], loc='upper right')
ax[0].set_title("Default Settings and Identification Order")
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].plot(x, y, label='Data', color='blue')
ax[1].plot(x, model[0](x) + background, '-.', label='Gaussian Fit',
           color='green')
ax[1].plot(x, model(x), '--', label='Composite Fit',
           color='red')
ax[1].set_xlim(90, 160)
ax[1].legend(loc='upper right')
ax[1].set_title("Peak 1: Simple and Composite Fit")
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')