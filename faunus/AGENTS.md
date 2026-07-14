# Code

- prefer deep module design; minimum public API (see skill https://deepwiki.com/mattpocock/skills/4.1.4-designing-deep-modules)
- use idiomatic rust and descriptive names
- make misuse hard w. e.g newtypes, enums
- avoid panic under normal use, prefer error
- unknown user input (serde YAML etc) is an error
- when executing plans, finish with a "/code-review", then "/simplify"
- I/O should support mac/linux/windows (OS independent)
- YAML output: prefer {mean, error} key mapping, not "x ± y" str
- remember to update docs/.md for changes in end-user behavior / new features (see below)
- internal state shouldn't be handled by caller, but automatic, hidden by deep design interfaces
- test driven development, TDD (skill https://deepwiki.com/mattpocock/skills/4.1-test-driven-development)

## Code comments

  - prefer *why* over *what* comments
  - use unicode for math

## Faunus-rs workspace

  - has git submodules (duello, interatomic, ...)

## Regression tests

  - never override fixtures without checking physics drift
  - prefer manual, targeted fixture updates

# End user documentation (docs/.md)

  - write and review with the /scientific-writing skill (https://raw.githubusercontent.com/mlund/claude-skills/refs/heads/main/plugins/scientific-writing/skills/scientific-writing/SKILL.md)
  - don't leak internals
  - target audience: physicist/biophysics/chemist
  - don't emphasize w. bold unless it's a warning
  - use markdown compatible LaTeX
  - verify with `scripts/docs-check/`
  - prefer .csv over .dat in examples
  - Prefer verified, hyperlinked DOI references

# Before committing:
  - cargo clippy --tests --no-deps
  - cargo fmt
  - *ask* to run `scripts/regression_tests.sh`. Useful to use `--exact` when manually running ignored regression tests to avoid overwriting fixtures.
    Never commit fixture updates without checking that physics is conserved.
  - commits, PRs, issue messages: Be brief, never append co-authorships
  - verify compilation with and without `gpu` feature
  - verify with `cargo check --workspace`
  - verify DOIs correctness with `doi2bib` CLI command

