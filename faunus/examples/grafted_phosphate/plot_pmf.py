#!/usr/bin/env python3
"""Plot phytate PMF from umbrella sampling."""

import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("pmf.csv", delimiter=",", names=True)
z = data["cv"]
pmf = data["pmf_kT"]
err = data["stderr_kT"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(z, pmf - err, pmf + err, alpha=0.3, color="C0")
ax.plot(z, pmf, "C0-", lw=1.5)
ax.set_xlabel("z (Å)")
ax.set_ylabel("PMF (kT)")
ax.set_title("Phytate PMF near oleosin brush")
ax.axhline(0, color="gray", ls="--", lw=0.5)
ax.set_xlim(z.min(), z.max())
fig.tight_layout()
fig.savefig("pmf.png", dpi=150)
print("Saved pmf.png")
plt.show()
