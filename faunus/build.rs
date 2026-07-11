//! Embed the git revision at build time so simulations can record it in
//! `output.yaml` for reproducibility. `git describe` yields the nearest tag,
//! commit count, short SHA, and a `-dirty` suffix for an unclean working tree.
//!
//! When built outside a git checkout — a source archive, a published crate, or
//! a machine without `git` — the emit fails; we then set the sentinel that
//! `git_revision()` in `simulation.rs` maps to "unknown", so the build still
//! succeeds instead of aborting.

use vergen_gitcl::{Emitter, GitclBuilder};

fn main() {
    if let Err(e) = emit_git_revision() {
        // Surface *why* provenance was lost: a source archive has no `.git`, but so
        // does a misconfigured CI checkout (missing tags, or git refusing a
        // dubious-ownership repo in a container) — those should be noticed, not masked.
        println!(
            "cargo:warning=git revision unavailable ({e}); output.yaml will report \"unknown\""
        );
        println!("cargo:rustc-env=VERGEN_GIT_DESCRIBE=VERGEN_IDEMPOTENT_OUTPUT");
    }
}

fn emit_git_revision() -> Result<(), Box<dyn std::error::Error>> {
    let gitcl = GitclBuilder::default().describe(true, true, None).build()?;
    Emitter::default().add_instructions(&gitcl)?.emit()?;
    Ok(())
}
