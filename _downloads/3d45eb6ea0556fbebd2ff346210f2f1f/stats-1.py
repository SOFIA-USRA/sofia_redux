import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.stats import find_outliers

rand = np.random.RandomState(42)
x = rand.rand(16, 16) - 0.5
outliers = find_outliers(x, threshold=5)
assert outliers.all()  # Verify no outliers identified at this stage

# Insert a single bad row
x[4] += 100

# Insert a single bad column
x[:, 4] += 100

# Find outliers from the entire distribution, and along each dimension
full_outliers = find_outliers(x, threshold=5)
row_outliers = find_outliers(x, axis=0, threshold=5)
col_outliers = find_outliers(x, axis=1, threshold=5)

plt.figure(figsize=(10,5))
plt.subplot(131)
plt.imshow(full_outliers, cmap='gray')
plt.title("Ouliers over full distribution")
plt.subplot(132)
plt.imshow(row_outliers, cmap='gray')
plt.title("Outliers along dimension 0")
plt.subplot(133)
plt.imshow(col_outliers, cmap='gray')
plt.title("Outliers along dimension 1")