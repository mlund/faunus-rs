# Agent instructions for faunus-rs

Applies to all coding agents (Claude Code, Codex, ...). `.claude/CLAUDE.md` is a symlink to this file.

The workspace uses git submodules (`duello`, `interatomic`, ...).

## Design

- Prefer deep modules with a minimal public API ([skill](https://deepwiki.com/mattpocock/skills/4.1.4-designing-deep-modules)).
- Make misuse hard: newtypes, enums, type-state over documented conventions.
- Internal state is the module's job, not the caller's; hide it behind the interface.
- Unknown user input (serde YAML, ...) is an error. Avoid panics under normal use; return errors.
- I/O must work on macOS, Linux and Windows.

## Rust

- Idiomatic Rust, descriptive names.
- Use the LSP tool / rust-analyzer where possible.
- Comments explain *why*, not *what*; unicode for math.

## Testing

- Test-driven development ([skill](https://deepwiki.com/mattpocock/skills/4.1-test-driven-development)).
- Cover physics correctness in unit tests, comparing against analytical theory in some limit where one exists.
- Regression tests use `macro_rules! regression_test` and are ignored by default, so they may run longer.
- Never overwrite fixtures without checking for physics drift; prefer manual, targeted updates.

## YAML

- Output: prefer a `{mean, error}` mapping over an `"x ± y"` string.
- Output: unicode math where it beats ascii for readability (⟨q²⟩-⟨q⟩²); keep ascii where it already reads fine (Rg).
- Input keys: add unicode only as an alias — it can be hard to type.

## End-user documentation (`docs/*.md`)

- Write it only after implementation, tests, code review and simplification are done.
- Draft and review with the [/scientific-writing skill](https://raw.githubusercontent.com/mlund/claude-skills/refs/heads/main/plugins/scientific-writing/skills/scientific-writing/SKILL.md).
- Audience: physicists, biophysicists, chemists. Don't leak internals.
- Markdown-compatible LaTeX; `.csv` over `.dat` in examples; verified, hyperlinked DOI references.
- Bold is for warnings only.

## Workflow

- Update `docs/*.md` whenever end-user behaviour changes or a feature is added.
- Finish a plan with `/code-review`, then `/simplify`.
- Keep commit, PR and issue messages brief. Never append co-authorship lines.

## Before committing

- `cargo fmt`
- `cargo clippy --tests --no-deps`
- `cargo check --workspace`, and verify compilation both with and without the `gpu` feature.
- If needed, *Ask* before running `scripts/regression_tests.sh`. When running ignored tests by hand, pass `--exact` to avoid overwriting fixtures. Never commit fixture updates without confirming the physics is conserved.
- `scripts/docs-check/`, and confirm that the formulas and math in `docs/` agree with the code.
- Verify DOIs with the `doi2bib` CLI.
