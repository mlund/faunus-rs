# Code

- prefer deep module design; minimum public API (see e.g. https://deepwiki.com/mattpocock/skills/4.1.4-designing-deep-modules)
- use idiomatic rust and descriptive names
- make misuse hard w. e.g newtypes, enums
- avoid panic under normal use, prefer error
- unknown user input (serde YAML etc) is an error
- when executing plans, finish with a "/code-review"
- I/O should support mac/linux/windows (OS independent)

## Code comments

  - prefer *why* over *what* comments
  - use unicode for math

## Faunus-rs workspace

  - has git submodules (duello, interatomic, ...)

## Regression tests

  - never override fixtures without checking physics drift
  - prefer manual, targeted fixture updates

# End user documentation (docs/.md)

  - write and review with the /scientific-writing skill
  - don't leak internals
  - target audience: physicist/biophysics/chemist
  - don't emphasize w. bold unless it's a warning
  - use markdown compatible LaTeX
  - verify with `scripts/docs-check/`
  - prefer .csv over .dat in examples

# Before committing:
  - cargo clippy / fmt
  - *ask* to run `scripts/regression_tests.sh`
  - don't add commit co-authorships (commits, PRs, issues). Be brief.
  - verify compilation with and without `gpu` feature
  - verify with `cargo check --workspace`
  - verify DOIs correctness with `doi2bib` CLI command

