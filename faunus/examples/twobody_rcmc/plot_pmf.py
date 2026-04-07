#!/usr/bin/env python3
"""Plot PMF from umbrella sampling output."""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("pmf.csv", delimiter=",", skiprows=1)
r, pmf, stderr = data[:, 0], data[:, 1], data[:, 2]

plt.figure(figsize=(8, 5))
plt.fill_between(r, pmf - stderr, pmf + stderr, alpha=0.3)
plt.plot(r, pmf, "b-")
plt.xlabel("z separation (Å)")
plt.ylabel("PMF (kJ/mol)")
plt.title("Umbrella Sampling PMF")
plt.tight_layout()
plt.savefig("pmf.png", dpi=150)
plt.show()
