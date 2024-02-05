import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.image.adjust import frebin

image = np.zeros((5, 5))
image[2, 2] = 1

shape = [x * 2 for x in image.shape]

resized0 = frebin(image, shape, order=0)
resized1 = frebin(image, shape, order=1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title("Original Image %s" % repr(image.shape))

plt.subplot(132)
plt.imshow(resized0, cmap='gray')
plt.title("Nearest-Neighbor %s" % repr(resized0.shape))

plt.subplot(133)
plt.imshow(resized1, cmap='gray')
plt.title("Linear %s" % repr(resized1.shape))