[![docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mlund.github.io/faunus-rs/)

# Faunus

Experimental Rust implementation of [Faunus](https://github.com/mlund/faunus) —
a molecular simulation framework for Monte Carlo and Langevin dynamics with
support for arbitrary potentials, Ewald summation, coarse-grained models,
and GPU-accelerated dynamics.

## Install

Requires the [Rust toolchain](https://www.rust-lang.org/tools/install).

~~~ bash
cargo install --git https://github.com/mlund/faunus-rs faunus --features gpu
~~~

## Usage

~~~ bash
faunus run --input input.yaml
~~~

See `examples/` for simulation setups and the [documentation](https://mlund.github.io/faunus-rs/) for details.

## Claude Code

Skills `faunus-input` and `faunus-run` help create YAML input files and
run simulations, respectively. They trigger automatically, e.g.:

~~~
Set up an NPT simulation of 256 SPC/E water molecules with Ewald summation
~~~

## Examples

Command               | Description
--------------------- | -----------------------------------------------
`scripts/twobody.sh`  | Free energy between two rigid multipolar bodies
