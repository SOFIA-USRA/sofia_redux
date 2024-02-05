from sofia_redux.toolkit.resampling import Resample
import imageio
import numpy as np

image = imageio.imread('imageio:chelsea.png').astype(float)
s = image.shape
rand = np.random.RandomState(42)
bad_pix = rand.rand(*s) < 0.7  # 70 percent corruption
bad_image = image.copy()
bad_image[bad_pix] = np.nan

y, x = np.mgrid[:s[0], :s[1]]
coordinates = np.vstack([c.ravel() for c in [x, y]])

yout = np.arange(s[0])
xout = np.arange(s[1])

# supply data in the form (nsets, ndata)
data = np.empty((s[2], s[0] * s[1]), dtype=float)
for frame in range(s[2]):
    data[frame] = bad_image[:, :, frame].ravel()

resampler = Resample(coordinates, data, window=10, order=2)
good = resampler(xout, yout, smoothing=0.1, relative_smooth=True,
                 order_algorithm='extrapolate', jobs=-1)

# get it b1ack into the correct shape and RGB format for plotting
good = np.clip(np.moveaxis(good, 0, -1).astype(int), 0, 255)

# Use the original good pixel coordinates where available
good[~bad_pix] = bad_image[~bad_pix]

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(bad_image / 255)
plt.title("Corrupted image (70% NaN)")

plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(good)
plt.title("Reconstructed image")