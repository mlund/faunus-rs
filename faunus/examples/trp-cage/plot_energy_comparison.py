"""Compare 6D table vs explicit nonbonded energy per frame."""
import numpy as np
import matplotlib.pyplot as plt

tab = np.loadtxt("energy_6dtable.dat", skiprows=1)
exp = np.loadtxt("energy_explicit_rerun.dat", skiprows=1)

# Skip frame 0 (pre-state-load placement) in rerun
exp = exp[1:]

n = min(len(tab), len(exp))
e_6d = tab[:n, 3]       # tabulated6d column
e_ex = exp[:n, 2]       # nonbonded column

# Filter: keep frames where explicit energy is finite and mild
mask = (e_ex > -100) & (e_ex < 100)
print(f"Good frames: {mask.sum()} / {n} ({100*mask.mean():.1f}%)")

e_6d_f = e_6d[mask]
e_ex_f = e_ex[mask]
offset = np.mean(e_6d_f - e_ex_f)
e_6d_shifted = e_6d_f - offset
print(f"Offset: {offset:.2f} kJ/mol")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Correlation plot
axes[0].plot(e_ex_f, e_6d_shifted, '.', ms=1, alpha=0.2)
lims = [min(e_ex_f.min(), e_6d_shifted.min()), max(e_ex_f.max(), e_6d_shifted.max())]
axes[0].plot(lims, lims, 'k--', lw=0.8, label='y = x')
axes[0].set_xlabel("Explicit (kJ/mol)")
axes[0].set_ylabel("6D table - offset (kJ/mol)")
axes[0].set_aspect('equal')
axes[0].legend()
r = np.corrcoef(e_ex_f, e_6d_shifted)[0, 1]
axes[0].set_title(f"Correlation (r = {r:.3f})")

# Overlaid histograms
bins = np.linspace(lims[0], lims[1], 80)
axes[1].hist(e_ex_f, bins=bins, alpha=0.5, density=True, label='Explicit')
axes[1].hist(e_6d_shifted, bins=bins, alpha=0.5, density=True, label=f'6D table (offset {offset:.1f} kJ/mol removed)')
axes[1].set_xlabel("Energy (kJ/mol)")
axes[1].set_ylabel("Probability density")
axes[1].legend()
axes[1].set_title("Energy distributions")

fig.suptitle("Trp-cage: 6D table vs explicit nonbonded")
fig.tight_layout()
fig.savefig("energy_comparison.pdf")
print("Saved energy_comparison.pdf")
