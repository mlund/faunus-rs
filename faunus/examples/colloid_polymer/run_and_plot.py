#!/usr/bin/env python3
"""Colloid-polymer depletion: many-body Hamiltonian (Metropolis MC).

Reproduces g(r) from Fig. 8a of Forsman & Woodward, Soft Matter, 2012, 8, 2121.
https://doi.org/10.1039/c2sm06737d

Hard-sphere colloids interact via a many-body depletion potential (eq 17)
arising from an implicit ideal polymer fluid with an exponential (Schulz-Flory)
molecular weight distribution. The polymers are not simulated explicitly;
instead their effect is captured by a density-dependent Hamiltonian that
includes higher-order (beyond pairwise) correlations. The colloid-colloid
hard-sphere exclusion uses sigma = 2*R_c = 40 Å as the contact distance.

Usage:
    python run_and_plot.py              # equilibrate + production + plot
    python run_and_plot.py --restart    # skip equilibration, reuse state.yaml
    python run_and_plot.py --plot-only  # only plot existing rdf.dat
"""

import argparse
import math
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FAUNUS = Path(__file__).resolve().parents[2] / "target" / "release" / "faunus"
WORKDIR = Path(__file__).resolve().parent
STATE = WORKDIR / "state.yaml"
# Paper parameters (Forsman & Woodward, 2012):
#   q = R_g/R_c = 1, with R_c = R_g = 20σ (σ = bond length)
#   κ = 1 → polydisperse (equilibrium/living) polymers (eq 1 with n=0)
#
# Reduced polymer reservoir density (eq 14):
#   polymer_density = ρ'_P × R_g³ (dimensionless)
#   where ρ'_P is the polymer chain number density in the bulk reservoir.
#   Fig 8a caption: ρ'_P is ~25% above the critical value.
#   From Fig 9a (many-body phase diagram), 4πρ'_P R_g³/3 ≈ 0.85 at critical,
#   giving ρ'_P R_g³ ≈ 0.20, so Fig 8a uses ρ'_P R_g³ ≈ 0.25.
#
# Colloid volume fraction (estimated from coexisting liquid branch, Fig 9a):
#   η_c ≈ 0.25–0.30 → L = (N × (4π/3) R_c³ / η_c)^(1/3) ≈ 230 Å
N = 100
BOX = 230.0
RG = 20.0

# Template shared by equilibration and production; filled via str.format()
INPUT_TEMPLATE = """\
# Colloid-polymer depletion: implicit ideal polymers
# Forsman & Woodward, Soft Matter, 2012, 8, 2121
# https://doi.org/10.1039/c2sm06737d
#
# Many-body depletion Hamiltonian (eqn 17), q = R_g/R_c = 1

atoms:
  - {{name: C, mass: 1.0, charge: 0.0, sigma: 40.0}}

molecules:
  - name: Colloid
    atoms: [C]

system:
  cell: !Cuboid [{box}, {box}, {box}]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  blocks:
    - molecule: Colloid
      N: {n}
      insert: !RandomAtomPos {{}}
  energy:
    nonbonded:
      default:
        - !HardSphere {{mixing: LB}}
    polymer_depletion:
      polymer_rg: 20.0
      polymer_density: 0.25
      kappa: 1.0
      colloid_radius: 20.0
      molecules: [Colloid]

propagate:
  seed: Hardware
  criterion: Metropolis
  repeat: {repeat}
  collections:
    - !Deterministic
      repeat: {n}
      moves:
        - !TranslateMolecule {{molecule: Colloid, dp: {dp}}}

analysis:
{analysis}
"""

PRODUCTION_ANALYSIS = """\
  - !RadialDistribution
    selections: ["molecule Colloid", "molecule Colloid"]
    file: rdf.dat
    dr: 1.0
    frequency: !Every 10
  - !Trajectory
    file: confout.xyz
    frequency: End"""

# Reference data digitized from Fig. 8a (liquid phase, polydisperse, q=1)
REF_MANYBODY = np.array(
    [
        [2.05, 1.5],
        [2.10, 2.4],
        [2.15, 2.5],
        [2.20, 2.3],
        [2.30, 1.9],
        [2.40, 1.6],
        [2.50, 1.35],
        [2.70, 1.1],
        [3.00, 0.85],
        [3.30, 0.80],
        [3.50, 0.82],
        [3.80, 0.88],
        [4.00, 0.92],
        [4.20, 0.95],
        [4.50, 0.97],
        [5.00, 1.0],
    ]
)

REF_PAIR = np.array(
    [
        [2.05, 2.5],
        [2.10, 3.7],
        [2.15, 3.8],
        [2.20, 3.3],
        [2.30, 2.5],
        [2.50, 1.5],
        [2.70, 1.05],
        [3.00, 0.70],
        [3.30, 0.65],
        [3.50, 0.72],
        [3.80, 0.82],
        [4.00, 0.88],
        [4.20, 0.93],
        [4.50, 0.97],
        [5.00, 1.0],
    ]
)


def write_yaml(filename, repeat, dp, analysis):
    Path(filename).write_text(
        INPUT_TEMPLATE.format(box=BOX, n=N, repeat=repeat, dp=dp, analysis=analysis)
    )


def write_lattice_state():
    """Write state.yaml with particles on a simple cubic lattice."""
    ncube = math.ceil(N ** (1.0 / 3.0))
    spacing = BOX / ncube
    lines = [
        f"cell: !Cuboid [{BOX}, {BOX}, {BOX}]",
        "step: 0",
        "groups:",
    ]
    for _ in range(N):
        lines.append("  - {molecule: 0, capacity: 1, size: Full}")
    lines.append("particles:")
    idx = 0
    for ix in range(ncube):
        for iy in range(ncube):
            for iz in range(ncube):
                if idx >= N:
                    break
                x = (ix + 0.5) * spacing - BOX / 2
                y = (iy + 0.5) * spacing - BOX / 2
                z = (iz + 0.5) * spacing - BOX / 2
                lines.append(
                    f"  - {{atom_id: 0, index: {idx}, "
                    f"pos: [{x:.4f}, {y:.4f}, {z:.4f}]}}"
                )
                idx += 1
    STATE.write_text("\n".join(lines) + "\n")
    print(f"Lattice state: {N} particles, spacing {spacing:.1f} Å")


def faunus(*args):
    cmd = [str(FAUNUS)] + list(args)
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=WORKDIR)
    if result.returncode != 0:
        sys.exit(f"Faunus exited with code {result.returncode}")


def equilibrate():
    write_lattice_state()
    write_yaml("equilibrate.yaml", repeat=100, dp=50.0, analysis="  []")
    print("=== Equilibration ===")
    faunus("run", "-i", "equilibrate.yaml", "-s", "state.yaml")
    print()


def production(repeat):
    write_yaml("input.yaml", repeat=repeat, dp=5.0, analysis=PRODUCTION_ANALYSIS)
    print(f"=== Production ({repeat} sweeps) ===")
    faunus("run", "-i", "input.yaml", "-s", "state.yaml")
    print()


def plot_rdf():
    rdf_file = WORKDIR / "rdf.dat"
    if not rdf_file.exists():
        sys.exit(f"{rdf_file} not found — run simulation first")

    data = np.loadtxt(rdf_file, comments="#")
    r, gr = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        *REF_MANYBODY.T, "s", ms=5, mfc="none", color="C0", label="many-body (Fig. 8a)"
    )
    ax.plot(
        *REF_PAIR.T, "o", ms=5, mfc="none", color="C1", label="pair approx. (Fig. 8a)"
    )
    ax.plot(r / RG, gr, "k-", lw=1.5, label="Faunus (many-body)")
    ax.axhline(1, color="gray", ls="--", lw=0.5)
    ax.set(xlabel=r"$r\, /\, R_g$", ylabel=r"$g(r)$", xlim=(2, 7), ylim=(0, 5))
    ax.set_title(r"Liquid phase, polydisperse polymers, $q = 1$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(WORKDIR / "rdf.pdf")
    fig.savefig(WORKDIR / "rdf.png", dpi=150)
    print("Saved rdf.pdf and rdf.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "repeat",
        nargs="?",
        type=int,
        default=1000,
        help="number of production sweeps (default: 1000)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="skip equilibration, reuse existing state.yaml",
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="only plot existing rdf.dat"
    )
    args = parser.parse_args()

    if args.plot_only:
        plot_rdf()
    else:
        if not args.restart:
            equilibrate()
        elif not STATE.exists():
            sys.exit(f"--restart requires {STATE}")
        production(args.repeat)
        plot_rdf()
