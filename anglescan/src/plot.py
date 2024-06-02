import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from math import ceil

angle_res = 0.05 # radians
x, y, pot = np.loadtxt('potential.dat', unpack=True)

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
plt.savefig('potential.png')