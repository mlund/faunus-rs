#!/usr/bin/env python3
"""Compare PMFs from Duello, twobody (explicit), and twobody_6dtable."""
import numpy as np
import matplotlib.pyplot as plt

def com_distance_to_pmf(file, bins=40):
    """Extract PMF from COM distance histogram: w(r) = -ln(p(r)/p_max)."""
    data = np.loadtxt(file, usecols=1)  # column 1 = instantaneous value
    p, edges = np.histogram(data, bins=bins, range=(38, 100))
    r = 0.5 * (edges[:-1] + edges[1:])
    mask = p > 0
    pmf = np.full_like(r, np.nan)
    pmf[mask] = -np.log(p[mask].astype(float) / p[mask].max())
    return r, pmf

# Duello reference
r_d, pmf_d, u_d = np.loadtxt(
    "/Users/mikael/github/duello/examples/cppm/pmf.dat",
    usecols=[0, 1, 2], unpack=True
)

# Faunus explicit MC
r_ex, pmf_ex = com_distance_to_pmf("twobody/com_distance.dat.gz")

# Faunus 6D table MC
r_tb, pmf_tb = com_distance_to_pmf("twobody_6dtable/com_distance.dat.gz")

# Shift MC PMFs to match Duello at large r
for r_mc, pmf_mc in [(r_ex, pmf_ex), (r_tb, pmf_tb)]:
    mask = (r_mc > 70) & np.isfinite(pmf_mc)
    if mask.any():
        duello_interp = np.interp(r_mc[mask], r_d, pmf_d)
        shift = np.nanmean(pmf_mc[mask] - duello_interp)
        pmf_mc -= shift

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_d, pmf_d, 'k-', lw=2, label='Duello (Boltzmann inversion)')
ax.plot(r_ex, pmf_ex, 'C0s-', ms=5, alpha=0.7, label='Explicit MC')
ax.plot(r_tb, pmf_tb, 'C1^-', ms=5, alpha=0.7, label='6D table MC')
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.set_xlabel(r'Mass center separation ($\AA$)')
ax.set_ylabel(r'Free energy ($k_BT$)')
ax.set_xlim(38, 100)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('pmf_comparison.pdf', dpi=150)
fig.savefig('pmf_comparison.png', dpi=150)
print("Saved pmf_comparison.pdf and pmf_comparison.png")
plt.show()
