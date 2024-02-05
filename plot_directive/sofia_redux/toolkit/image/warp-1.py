import numpy as np
import matplotlib.pyplot as plt
import imageio
from sofia_redux.toolkit.image.warp import warp_image

image = imageio.imread('imageio:checkerboard.png')
sy, sx = image.shape

# Define some original grid positions
xi, yi = np.meshgrid(np.linspace(0, sx, sx // 20),
                     np.linspace(0, sy, sy // 20))

# Define some rotated grid positions
cenx, ceny = sx / 2, sy / 2
xo = xi - cenx
yo = yi - ceny
r = np.sqrt((xo ** 2) * (yo ** 2))
r /= r.max()
a = np.radians(-20) * r
xo = xo * np.cos(a) - yo * np.sin(a)
yo = xo * np.sin(a) + yo * np.cos(a)
xo += cenx
yo += ceny

# create a new image on the rotated coordinates
rotated = warp_image(image, xi, yi, xo, yo, mode='edge')

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(122)
plt.imshow(rotated, cmap='gray')
plt.title("Warped Image")