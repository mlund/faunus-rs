# faunus-rs

Cargo workspace for co-developing the following Rust crates:

| Crate | Description |
|-------|-------------|
| [faunus](faunus/) | Molecular simulation framework |
| [coulomb](https://github.com/mlund/coulomb) | Electrolytes and electrostatic interactions |
| [interatomic](https://github.com/mlund/interatomic) | Inter-particle interactions |
| [icotable](https://github.com/mlund/icotable) | Icosphere-based 6D angular lookup tables |
| [duello](https://github.com/mlund/duello) | Osmotic second virial coefficients and dissociation constants |
| [cgkitten](https://github.com/mlund/cgkitten) | Coarse-grained protein structures with Monte Carlo titration |

> **Looking for `faunus`?** The `faunus` crate lives in the [`faunus/`](faunus/) subdirectory.

## Getting started

Clone with submodules:

```sh
git clone --recurse-submodules https://github.com/mlund/faunus-rs.git
cd faunus-rs
```

If you already cloned without `--recurse-submodules`:

```sh
git submodule update --init
```

Then build the entire workspace:

```sh
cargo build
```

## Pushing over SSH

Submodule URLs use HTTPS so that cloning works without authentication.
To push over SSH, add this to your `~/.gitconfig`:

```ini
[url "git@github.com:"]
    pushInsteadOf = https://github.com/
```

## Why a workspace?

The crates in this workspace depend on each other.
The workspace `Cargo.toml` uses `[patch]` sections to redirect git and crates.io dependencies
to local paths, so changes in one crate are immediately visible to its dependents without
publishing or pushing first.

Each crate is published to [crates.io](https://crates.io) individually and can be used
independently — the workspace is a development convenience, not a requirement for end users.

## License

See individual crate directories for license information.
