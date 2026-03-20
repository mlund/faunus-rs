#!/usr/bin/env python3
"""Plot 2D free energy surface from Wang-Landau output."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "free_energy.csv"
df = pd.read_csv(path)
piv = df.pivot_table(index="cv1", columns="cv2", values="free_energy_kT")

fig, ax = plt.subplots()
im = ax.pcolormesh(piv.columns, piv.index, piv.values, shading="auto", cmap="viridis")
ax.set_xlabel("y")
ax.set_ylabel("x")
fig.colorbar(im, ax=ax, label="F / kT")
ax.set_title("Free energy surface")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("free_energy.png", dpi=150)
print(f"Saved free_energy.png (Δg = {piv.values.max() - piv.values.min():.2f} kT)")
