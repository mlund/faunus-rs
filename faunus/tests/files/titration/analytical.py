#!/usr/bin/env python3
"""
Analytical Henderson-Hasselbalch predictions for the ideal titration test.

Computes expected protonation states and ion concentrations for a system
of titratable residues at a given pH, coupled to a grand canonical 1:1
salt bath. No interactions are assumed (ideal limit).

Theory
------
Each residue titrates independently via Henderson-Hasselbalch:

    HA <-> A- + H+,   K_a = 10^(-pK)

    alpha_deprot = 1 / (1 + 10^(pK - pH))

For acids (GLU, CYS), the deprotonated form carries charge -1.
For bases (HIS), the protonated form (HHIS+) carries charge +1.

The grand canonical salt reaction (empty <-> Na+ + Cl-) with molecular
activities a_Na = a_Cl = a gives the equilibrium constraint:

    N_Na * N_Cl = (V * z)^2

where z = a * N_A / 1e27 is the fugacity in inverse angstrom cubed,
V is the cell volume in angstrom cubed, and N_A is Avogadro's number.

Combined with charge conservation (total system charge = 0):

    N_Na - N_Cl = -Q_protein

these two equations determine the equilibrium ion counts.
"""

import math

# ----- Parameters (must match input.yaml) -----
pH = 7.0
residues = [
    # (name, pK, charge_deprot, charge_prot, count)
    ("GLU", 4.24, -1.0, 0.0, 1),
    ("HIS", 6.54, 0.0, +1.0, 1),
    ("CYS", 8.55, -1.0, 0.0, 1),
]
salt_activity = 0.030  # molar
sphere_radius = 50.0  # angstrom
# Total system charge is fixed by initial conditions and conserved by
# all reactions. The simulation must start electroneutral for these
# predictions to hold (e.g. fully protonated protein + compensating ions).
total_charge = 0.0

# ----- Constants -----
AVOGADRO = 6.02214076e23
# 1 mol/L = N_A / (1e-3 m^3) = N_A / (1e27 A^3)
MOLAR_TO_INV_A3 = AVOGADRO * 1e-27  # 6.022e-4

# ----- Derived quantities -----
V = (4.0 / 3.0) * math.pi * sphere_radius**3  # cell volume in A^3
z = salt_activity * MOLAR_TO_INV_A3  # fugacity in A^-3

print(f"Cell volume:  {V:.1f} A^3")
print(f"Salt fugacity: {z:.6e} A^-3")
print(f"V * z = {V * z:.4f}")
print()

# ----- Henderson-Hasselbalch for each residue -----
print("Henderson-Hasselbalch predictions (ideal, no interactions):")
print("-" * 60)

Q_protein = 0.0
for name, pK, q_deprot, q_prot, n in residues:
    alpha_deprot = 1.0 / (1.0 + 10.0 ** (pK - pH))
    alpha_prot = 1.0 - alpha_deprot
    avg_charge = alpha_deprot * q_deprot + alpha_prot * q_prot

    Q_protein += n * avg_charge

    print(f"  {name} (pK {pK}):")
    print(f"    deprot fraction = {alpha_deprot:.6f}")
    print(f"    prot fraction   = {alpha_prot:.6f}")
    print(f"    <charge>        = {avg_charge:+.6f}")
    if n > 1:
        print(f"    count (x{n})     = {n * alpha_deprot:.6f} (deprot)")

print()
print(f"  Total protein charge: {Q_protein:+.6f}")

# ----- Ion concentrations from electroneutrality + GC equilibrium -----
# Two constraints determine <N_Na> and <N_Cl>:
#   1. Charge conservation: N_Na - N_Cl = total_charge - Q_protein
#   2. GC equilibrium for "= Na+ + Cl-" with K=1: at equilibrium
#      mu_Na + mu_Cl = 0, giving N_Na * N_Cl = (V * z)^2
# Substituting (1) into (2) yields a quadratic in N_Cl.
delta = total_charge - Q_protein  # = N_Na - N_Cl
product = (V * z) ** 2  # = N_Na * N_Cl
discriminant = delta**2 + 4.0 * product
N_Cl = (-delta + math.sqrt(discriminant)) / 2.0
N_Na = N_Cl + delta

print()
print("Grand canonical salt equilibrium:")
print("-" * 60)
print(f"  N_Na - N_Cl = {delta:.4f}")
print(f"  N_Na * N_Cl = {product:.4f}")
print(f"  <N_Na> = {N_Na:.4f}")
print(f"  <N_Cl> = {N_Cl:.4f}")

# ----- Summary for comparison with simulation -----
print()
print("Summary (compare with simulation output):")
print("-" * 60)
for name, pK, q_deprot, q_prot, n in residues:
    alpha_deprot = 1.0 / (1.0 + 10.0 ** (pK - pH))
    if q_deprot != 0:  # acid: charged when deprotonated (GLU-, CYS-)
        print(f"  count({name:4s}) = {n * alpha_deprot:.6f}")
    else:  # base: charged when protonated (HHIS+)
        print(f"  count({'H' + name:4s}) = {n * (1.0 - alpha_deprot):.6f}")
print(f"  count(Na)   = {N_Na:.4f}")
print(f"  count(Cl)   = {N_Cl:.4f}")
print(f"  charge(all) = {total_charge:.1f}")
