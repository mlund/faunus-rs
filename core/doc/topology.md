# Topology

## Chemical reactions

This module contains support for chemical reactions, including
parsing and handling of reaction strings.
This is used for speciation moves in the grand canonical ensemble.
A participant is either an atom, a molecule or an implicit participant.
When parsing a reaction, atoms are prefixed with a dot or an atom sign, e.g. _.Na_ or _⚛Na_.
Implicit participants are prefixed with a tilde or a ghost, e.g. _~H_ or _👻H_.
Molecules are not prefixed, e.g. _Cl_.

Participant | Example                |  Notes
------------|----------------------- | ------------------------------------
Molecular   | `A + A ⇌ D`            | Possible arrows: `=`, `⇌`, `⇄`, `→`
Implicit    | `RCOO- + 👻H+ ⇌ RCOOH` | Mark with `👻` or `~`
Atomic      | `⚛Pb ⇄ ⚛Au`            | Mark with `⚛` or `.`

