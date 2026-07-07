//! Embed the git revision at build time so simulations can record it in
//! `output.yaml` for reproducibility. `git describe` yields the nearest tag,
//! commit count, short SHA, and a `-dirty` suffix for an unclean working tree.
//!
//! When built outside a git checkout (e.g. from a source archive or a published
//! crate) vergen falls back to a placeholder; see `git_revision()` in `cli.rs`.

use vergen_gitcl::{Emitter, GitclBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gitcl = GitclBuilder::default().describe(true, true, None).build()?;
    Emitter::default().add_instructions(&gitcl)?.emit()?;
    Ok(())
}
