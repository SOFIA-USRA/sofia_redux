import matplotlib.pyplot as plt
import numpy as np
from sofia_redux.toolkit.fitting.polynomial import Polyfit

y, x = np.mgrid[:5, :5]
z = 1 + (2 * x * y) + (0.1 * x ** 2 * y ** 2)
redundant_pfit = Polyfit(x, y, z, 2)

plt.figure(figsize=(5, 5))
plt.plot(z, label='Original')
plt.plot(redundant_pfit(x, y), '--', label="Redundant fit")
plt.xlabel('x')
plt.ylabel('f(x, y)')
plt.title("Redundant term set fit")
plt.legend()