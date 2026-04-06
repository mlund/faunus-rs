#!/usr/bin/env python3
"""Plot phytate orientation (Szz/Rg²) vs z-position from umbrella window outputs."""

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

runs = Path("runs")


def _make_loader():
    """YAML loader that handles Faunus custom tags (!Stochastic, !pK, etc.)."""
    loader = type("Loader", (yaml.SafeLoader,), {})

    def _any_constructor(l, _suffix, node):
        if isinstance(node, yaml.MappingNode):
            return l.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return l.construct_sequence(node)
        return l.construct_scalar(node)

    loader.add_multi_constructor("!", _any_constructor)
    return loader


def parse_dir(name):
    parts = name.split("_")
    pH = float(parts[0].replace("pH", ""))
    I = float(parts[1].replace("I", ""))
    return pH, I


def extract_orientation(run_dir):
    """Extract (window_center, Szz/Rg²) from each window's output.yaml."""
    state_dir = run_dir / "umbrella_states"
    if not state_dir.exists():
        return None, None

    centers = []
    orientations = []

    for f in sorted(state_dir.glob("window*_output.yaml")):
        with open(f) as fh:
            data = yaml.load(fh, Loader=_make_loader())

        for entry in data.get("analysis", []):
            if not isinstance(entry, dict):
                continue
            if "polymershape" in entry:
                shape = entry["polymershape"]
                szz = shape.get("Szz")
                rg = shape.get("Rg")
                if szz is not None and rg is not None and rg > 0:
                    orientations.append(szz / rg**2)
            if "collectivevariable" in entry:
                cv = entry["collectivevariable"]
                if cv.get("property") == "AtomPosition":
                    mean = cv.get("mean")
                    if mean is not None and not np.isnan(mean):
                        centers.append(mean)

    if len(centers) != len(orientations):
        n = min(len(centers), len(orientations))
        centers = centers[:n]
        orientations = orientations[:n]

    return np.array(centers), np.array(orientations)


# Collect results
results = {}
for d in sorted(runs.iterdir()):
    if (d / "umbrella_states").exists():
        pH, I = parse_dir(d.name)
        z, orient = extract_orientation(d)
        if z is not None and len(z) > 0:
            results[(pH, I)] = (z, orient)

if not results:
    print("No orientation data found")
    raise SystemExit(1)

pHs = sorted({k[0] for k in results})
Is = sorted({k[1] for k in results})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

# Left: orientation vs pH
I_ref = Is[len(Is) // 2]
for pH in pHs:
    key = (pH, I_ref)
    if key not in results:
        continue
    z, orient = results[key]
    idx = np.argsort(z)
    ax1.plot(z[idx], orient[idx], "o-", ms=3, lw=1.5, label=f"pH {pH:.0f}")

ax1.set_xlabel("z (Å)")
ax1.set_ylabel("$S_{zz} / R_g^2$")
ax1.set_title(f"Phytate orientation (I = {I_ref} M)")
ax1.legend()
ax1.axhline(1 / 3, color="gray", ls="--", lw=0.5, label="isotropic")
ax1.set_ylim(0, 0.7)

# Right: orientation vs ionic strength
pH_ref = pHs[len(pHs) // 2]
for I in Is:
    key = (pH_ref, I)
    if key not in results:
        continue
    z, orient = results[key]
    idx = np.argsort(z)
    ax2.plot(z[idx], orient[idx], "o-", ms=3, lw=1.5, label=f"I = {I} M")

ax2.set_xlabel("z (Å)")
ax2.set_title(f"Phytate orientation (pH {pH_ref:.0f})")
ax2.legend()
ax2.axhline(1 / 3, color="gray", ls="--", lw=0.5)
ax2.set_ylim(0, 0.7)

fig.tight_layout()
fig.savefig("orientation_scan.png", dpi=150)
print("Saved orientation_scan.png")
plt.show()
