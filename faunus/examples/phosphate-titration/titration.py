#!/usr/bin/env python3
"""
Phosphate titration curve via Monte Carlo reaction ensemble.

For each pH value, Faunus is run with a single phosphate molecule in a spherical
cell.  The protonation state fluctuates between four species through molecular
swap moves (reaction ensemble, Smith & Triska 1994, doi:10.1063/1.466443):

    H₃PO₄  ⇌  H₂PO₄⁻ + H⁺   pKa 2.15
    H₂PO₄⁻ ⇌  HPO₄²⁻ + H⁺   pKa 7.20
    HPO₄²⁻ ⇌  PO₄³⁻  + H⁺   pKa 12.35

Each swap preserves the molecular geometry (COM + random orientation) and the
H⁺ chemical potential is set by its activity (= 10^{-pH}).  No nonbonded
interactions are included, so the simulation reproduces the ideal
Henderson-Hasselbalch (HH) result exactly in the large-sample limit.

Workflow
--------
1. `set_ph`      – patches the H⁺ activity in input.yaml for each pH point.
2. Faunus run    – MC simulation; CollectiveVariable:Count tracks each species.
3. `parse_counts`– extracts mean atom counts from the output YAML.
4. `net_charge`  – converts atom counts to average charge per molecule.
5. Plot          – MC points vs. analytic HH curve with pKa markers.

Output
------
titration.png   average charge <z> vs pH
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FAUNUS = Path(__file__).resolve().parents[3] / "target" / "release" / "faunus"
WORKDIR = Path(__file__).resolve().parent
TEMPLATE = (WORKDIR / "input.yaml").read_text()

# Net charge per molecule for each protonation state
SPECIES = {"H₃PO₄": 0, "H₂PO₄⁻": -1, "HPO₄²⁻": -2, "PO₄³⁻": -3}
PKA = [2.15, 7.20, 12.35]


def set_ph(text: str, ph: float, repeat: int = 5000) -> str:
    """Return input YAML with H⁺ activity and sweep count patched for *ph*."""
    activity = 10.0 ** (-ph)
    # Replace activity value on the H+ atom line (single-line YAML dict)
    text = re.sub(
        r"(name:\s*H\+.*?activity:\s*)[\d.eE+-]+",
        rf"\g<1>{activity:.6e}",
        text,
    )
    # Replace the outermost repeat (propagate.repeat = number of MC sweeps)
    return re.sub(r"(repeat:\s*)\d+", rf"\g<1>{repeat}", text, count=1)


def parse_counts(path: Path) -> dict[str, float]:
    """Extract mean atom counts from CollectiveVariable:Count entries in output YAML.

    Faunus writes counts as *atom* counts (5 atoms per phosphate molecule), so
    the species fractions are obtained by dividing by the total atom count.
    The description field written by Faunus has the form:
        description: 'selection: molecule ''H₂PO₄⁻'''
    """
    text = path.read_text()
    counts = {}
    for m in re.finditer(
        r"description:\s*'selection:\s*molecule\s*''(.+?)'''\s*\n"
        r"\s*num_samples:.*\n"
        r"\s*mean:\s*([\d.eE+-]+)",
        text,
    ):
        counts[m.group(1)] = float(m.group(2))
    return counts


def net_charge(counts: dict[str, float]) -> float:
    """Average net charge per molecule from atom-count dict.

    Because every species has the same number of atoms, the atom counts cancel
    in the weighted average:  <z> = Σ z_i · n_i / Σ n_i.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(SPECIES[sp] * counts.get(sp, 0) for sp in SPECIES) / total


def hh_charge(ph: float) -> float:
    """Analytic Henderson-Hasselbalch average charge for triprotic phosphoric acid.

    Derived from the distribution of microstates weighted by the three
    stepwise acid dissociation constants Ka1, Ka2, Ka3.
    """
    h = 10.0 ** (-ph)
    ka1, ka2, ka3 = (10.0 ** (-pk) for pk in PKA)
    denom = h**3 + ka1 * h**2 + ka1 * ka2 * h + ka1 * ka2 * ka3
    return -(ka1 * h**2 + 2 * ka1 * ka2 * h + 3 * ka1 * ka2 * ka3) / denom


def main():
    ph_values = np.arange(1.0, 14.5, 0.5)
    charges = []

    for ph in ph_values:
        print(f"pH {ph:.1f} ...", end=" ", flush=True)
        run_input = WORKDIR / f"_run_ph{ph:.1f}.yaml"
        run_output = WORKDIR / f"_run_ph{ph:.1f}_out.yaml"
        run_input.write_text(set_ph(TEMPLATE, ph))

        subprocess.run(
            [str(FAUNUS), "-o", str(run_output), "run", "-i", str(run_input)],
            cwd=str(WORKDIR),
            check=True,
            capture_output=True,
            env={**os.environ, "RUST_LOG": "warn"},
        )

        counts = parse_counts(run_output)
        q = net_charge(counts)
        charges.append(q)

        text = run_output.read_text()
        acc = re.search(r"acceptance_ratio:\s*([\d.eE+-]+)", text)
        print(f"<z> = {q:.3f}  acc = {acc.group(1) if acc else '?'}")

        run_input.unlink()
        run_output.unlink()

    ph_fine = np.linspace(ph_values[0], ph_values[-1], 500)
    hh = [hh_charge(p) for p in ph_fine]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ph_values, charges, "o", label="Faunus MC", zorder=3)
    ax.plot(ph_fine, hh, "-", color="gray", label="Henderson-Hasselbalch")
    for pk in PKA:
        ax.axvline(pk, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("pH")
    ax.set_ylabel("Average charge")
    ax.set_title("Phosphate titration")
    ax.legend()
    fig.tight_layout()
    out = WORKDIR / "titration.png"
    fig.savefig(str(out), dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
