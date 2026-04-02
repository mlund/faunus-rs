#!/usr/bin/env python3
"""Plot PMF from umbrella sampling and Wang-Landau."""

import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(8, 4))

# Umbrella sampling
if os.path.exists("pmf.csv"):
    data = np.genfromtxt("pmf.csv", delimiter=",", names=True)
    z = data["cv"]
    pmf = data["pmf_kT"]
    err = data["stderr_kT"]
    pmf -= pmf[-1]
    ax.fill_between(z, pmf - err, pmf + err, alpha=0.3, color="C0")
    ax.plot(z, pmf, "C0-", lw=1.5, label="Umbrella")

# Wang-Landau
if False and os.path.exists("pmf_wl.csv"):
    wl = np.genfromtxt("pmf_wl.csv", delimiter=",", names=True)
    fe = -wl["free_energy_kT"]
    fe -= fe[-1]
    ax.plot(wl["cv"], fe, "C1-", lw=1.5, label="Wang-Landau")

# RDF → PMF
if os.path.exists("rdf.csv"):
    rdf = np.genfromtxt("rdf.csv", delimiter=",", names=True)
    mask = rdf["gr"] > 0
    pmf_rdf = -np.log(rdf["gr"][mask])
    pmf_rdf -= pmf_rdf[-1]
    ax.plot(rdf["r"][mask], pmf_rdf, "C2-", lw=1.5, label="RDF")

ax.set_xlabel("COM-COM z distance (Å)")
ax.set_ylabel("PMF (kT)")
ax.axhline(0, color="gray", ls="--", lw=0.5)
ax.legend()
fig.tight_layout()
fig.savefig("pmf.png", dpi=150)
print("Saved pmf.png")
plt.show()
