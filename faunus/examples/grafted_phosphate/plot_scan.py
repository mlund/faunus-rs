#!/usr/bin/env python3
"""Plot PMF scan over pH and ionic strength."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

runs = Path("runs")

def load_pmf(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    z, pmf, err = data["cv"], data["pmf_kT"], data["stderr_kT"]
    # Shift so the midplane region (z ≈ 0, away from both walls and brush) averages to zero
    mid = np.abs(z) < 20
    if mid.any():
        pmf -= np.nanmean(pmf[mid])
    return z, pmf, err

def parse_dir(name):
    """Extract pH and I from directory name like 'pH7_I0.15'."""
    parts = name.split("_")
    pH = float(parts[0].replace("pH", ""))
    I = float(parts[1].replace("I", ""))
    return pH, I

# Collect all available results
results = {}
for d in sorted(runs.iterdir()):
    pmf_file = d / "pmf.csv"
    if pmf_file.exists():
        pH, I = parse_dir(d.name)
        results[(pH, I)] = load_pmf(pmf_file)

if not results:
    print("No pmf.csv files found in runs/")
    raise SystemExit(1)

pHs = sorted({k[0] for k in results})
Is = sorted({k[1] for k in results})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

# Left: PMF vs pH (one curve per pH, at each I use the middle I)
I_ref = Is[len(Is) // 2]
for pH in pHs:
    key = (pH, I_ref)
    if key not in results:
        continue
    z, pmf, err = results[key]
    ax1.fill_between(z, pmf - err, pmf + err, alpha=0.15)
    ax1.plot(z, pmf, lw=1.5, label=f"pH {pH:.0f}")

ax1.set_xlabel("z (Å)")
ax1.set_ylabel("PMF (kT)")
ax1.set_title(f"Effect of pH (I = {I_ref} M)")
ax1.legend()
ax1.axhline(0, color="gray", ls="--", lw=0.5)
ax1.set_xlim(None, 0)

# Right: PMF vs ionic strength (one curve per I, at the middle pH)
pH_ref = pHs[len(pHs) // 2]
for I in Is:
    key = (pH_ref, I)
    if key not in results:
        continue
    z, pmf, err = results[key]
    ax2.fill_between(z, pmf - err, pmf + err, alpha=0.15)
    ax2.plot(z, pmf, lw=1.5, label=f"I = {I} M")

ax2.set_xlabel("z (Å)")
ax2.set_title(f"Effect of ionic strength (pH {pH_ref:.0f})")
ax2.legend()
ax2.axhline(0, color="gray", ls="--", lw=0.5)
ax2.set_xlim(None, 0)

fig.tight_layout()
fig.savefig("pmf_scan.png", dpi=150)
print("Saved pmf_scan.png")
plt.show()
