import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

x, y, pot = np.loadtxt('potential.dat', unpack=True)
ngrid = 2000
fig, ax1 = plt.subplots(nrows=1)

# Contour plot of irregularly spaced data coordinates via interpolation on a grid.

# Create grid values first.
xi = np.linspace(x.min(), x.max(), ngrid)
yi = np.linspace(y.min(), y.max(), ngrid)
zi = griddata((x, y), pot, (xi[None,:], yi[:,None]), method='linear')

plt.contourf(xi, yi, zi, 20, cmap = plt.cm.RdBu)
plt.colorbar() # draw colorbar
plt.plot(x, y, 'ko', ms=1)
plt.savefig('potential.png')