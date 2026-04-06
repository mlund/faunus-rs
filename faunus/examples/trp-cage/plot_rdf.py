#!/usr/bin/env python3
"""Compare RDFs at 30% volume fraction with Duello pair PMF."""
import numpy as np
import matplotlib.pyplot as plt

explicit = np.loadtxt("rdf_explicit_30pct.dat")
table6d = np.loadtxt("rdf_6dtable_30pct.dat")

pmf = np.loadtxt("/Users/mikael/github/duello/examples/trp-cage/pmf.dat")
g_pmf = np.exp(-pmf[:, 1])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(explicit[:, 0], explicit[:, 1], label="Explicit (N=10)", lw=1.5)
ax.plot(table6d[:, 0], table6d[:, 1], label="6D table (N=10)", lw=1.5, ls="--")
ax.plot(pmf[:, 0], g_pmf, label="Duello PMF (pair)", lw=1.5, ls=":")
ax.set_xlabel("r (Å)")
ax.set_ylabel("g(r)")
ax.set_title("Trp-cage COM–COM RDF (φ≈30%, 50 mM NaCl)")
ax.legend()
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig("rdf_comparison_30pct.pdf")
print("Saved rdf_comparison_30pct.pdf")
