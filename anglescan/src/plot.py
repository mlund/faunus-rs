import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from math import ceil


exact = True
angle_res = 0.005 # radians

x, y, pot_interpolated, pot_exact, err = np.loadtxt('pot_at_angles.dat', unpack=True)
if exact == True:
    pot = pot_exact
else:
    pot = pot_interpolated

fig, ax1 = plt.subplots(nrows=1)
maxpot = max(abs(pot.min()), abs(pot.max()))

ngridx = ceil((x.max() - x.min()) / angle_res)
ngridy = ceil((y.max() - y.min()) / angle_res)
print("ngridx = ", ngridx)
print("ngridy = ", ngridy)
xi = np.linspace(x.min(), x.max(), ngridx)
yi = np.linspace(y.min(), y.max(), ngridy)
zi = griddata((x, y), pot, (xi[None,:], yi[:,None]), method='linear')
plt.contourf(xi, yi, zi, 20, cmap = plt.cm.RdBu)

# Overlay exact potential from each vertic (colored circles)
x, y, pot = np.loadtxt('pot_at_vertices.dat', unpack=True)
plt.scatter(x, y, c=pot, cmap=plt.cm.RdBu, s=15, marker='o', edgecolor='k', linewidths=0.1)
plt.colorbar() # draw colorbar
plt.clim(-maxpot, maxpot)
plt.ylim(0.0, 2.0*np.pi)
plt.xlim(0.0, np.pi)
plt.savefig('potential.pdf')

