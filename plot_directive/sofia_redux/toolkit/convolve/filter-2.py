import numpy as np
import matplotlib.pyplot as plt
from sofia_redux.toolkit.convolve.filter import savgol

x = np.arange(-20, 20)
y = np.zeros(x.size)
y[x.size // 2] = 1

for i in range(0, 8, 2):
    plt.plot(x, savgol(y, 11, order=i), '-o', label="order %i" % i)
plt.legend()
plt.xlim(-7, 7)
plt.title("Savitzky-Golay Coefficients")
plt.xlabel("Kernel index")
plt.ylabel("Coefficient value")