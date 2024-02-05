from sofia_redux.toolkit.image.warp import polywarp_image
import matplotlib.pyplot as plt
import imageio

image = imageio.imread('imageio:camera.png')
# Define warp based on corners of image for this example
x0 = [0, 0, 511, 511]
y0 = [511, 0, 0, 511]
x1 = [200, 100, 300, 400]
y1 = [400, 200, 200, 400]
warped = polywarp_image(image, x0, y0, x1, y1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].set_xlim(-10, 521)
ax[0].set_ylim(521, -10)
ax[0].plot(x0 + [x0[0]], y0 + [y0[0]], '-o', color='red', markersize=6,
           label="$(x_0, y_0)$")
ax[0].plot(x1 + [x1[0]], y1 + [y1[0]], '-o', color='lime', markersize=6,
           label="$(x_1, y_1)$")
for i in range(4):
    ax[0].plot([x0[i], x1[i]], [y0[i], y1[i]], '--', color='cyan')

ax[0].legend(loc=(0.12, 0.82))
ax[1].imshow(warped, cmap='gray')
ax[1].set_title("Warped Image")