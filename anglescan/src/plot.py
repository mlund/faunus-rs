import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from math import ceil

angle_res = 0.05 # radians
x, y, pot = np.loadtxt('potential.dat', unpack=True)
print("x_min = ", x.min(), " x_max = ", x.max())
print("y_min = ", y.min(), " y_max = ", y.max())

# Contour plot of irregularly spaced data coordinates via grid interpolation
ngridx = ceil((x.max() - x.min()) / angle_res)
ngridy = ceil((y.max() - y.min()) / angle_res)
print("ngridx = ", ngridx)
print("ngridy = ", ngridy)
xi = np.linspace(x.min(), x.max(), ngridx)
yi = np.linspace(y.min(), y.max(), ngridy)
zi = griddata((x, y), pot, (xi[None,:], yi[:,None]), method='linear')

fig, ax1 = plt.subplots(nrows=1)
plt.contourf(xi, yi, zi, 20, cmap = plt.cm.RdBu)
plt.colorbar() # draw colorbar
plt.plot(x, y, 'ko', ms=1)

# Load tabulated data
x, y, pot = np.loadtxt('potential.tab', unpack=True)
mask = (pot > 0.001) | (pot < -0.001)
print("non-zero potential points = ", len(pot[mask]))
#plt.scatter(x[mask], y[mask], c=pot[mask], cmap=plt.cm.RdBu, s=5, marker='o')
plt.savefig('potential.png')
