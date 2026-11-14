//! The `include:` file tree, shared by every reader so topology and energy
//! see the same files in the same order.

use std::path::{Path, PathBuf};

/// Files included from `root`, transitively, in the order their definitions
/// take precedence: depth-first, each file before the files it includes.
/// A file reached twice is read once; a cycle is an error.
pub(crate) fn included_files(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    visit(root, &mut vec![canonical(root)], &mut files)?;
    Ok(files.into_iter().map(|(_, spelled)| spelled).collect())
}

/// `stack` and the first of each `files` pair are canonical, so differently
/// spelled paths to one file compare equal.
fn visit(
    file: &Path,
    stack: &mut Vec<PathBuf>,
    files: &mut Vec<(PathBuf, PathBuf)>,
) -> anyhow::Result<()> {
    for child in direct_includes(file)? {
        let id = canonical(&child);
        if stack.contains(&id) {
            let chain: Vec<_> = stack
                .iter()
                .chain([&id])
                .map(|f| f.display().to_string())
                .collect();
            anyhow::bail!("include cycle: {}", chain.join(" → "));
        }
        if files.iter().any(|(seen, _)| *seen == id) {
            continue;
        }
        files.push((id.clone(), child.clone()));
        stack.push(id);
        visit(&child, stack, files)?;
        stack.pop();
    }
    Ok(())
}

/// The `include:` list of `file`, resolved against its directory.
fn direct_includes(file: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let yaml = crate::auxiliary::read_yaml(file)
        .map_err(|err| anyhow::anyhow!("Error reading file {}: {err}", file.display()))?;
    let root: yaml_serde::Value = yaml_serde::from_str(&yaml)?;
    let Some(include) = root.get("include") else {
        return Ok(vec![]);
    };
    let names: Vec<super::InputPath> = yaml_serde::from_value(include.clone()).map_err(|err| {
        anyhow::anyhow!(
            "in {}: `include` must be a list of file names: {err}",
            file.display()
        )
    })?;
    Ok(names
        .into_iter()
        .filter_map(|mut name| {
            name.finalize(file);
            name.path().cloned()
        })
        .collect())
}

/// A missing file keeps its spelling; reading it reports the error.
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Depth-first, each file before its own includes; a file reached twice is listed once.
    #[test]
    fn files_are_listed_in_precedence_order_once() {
        let dir = tempfile::tempdir().unwrap();
        let write =
            |name: &str, content: &str| std::fs::write(dir.path().join(name), content).unwrap();
        write("root.yaml", "include: [a.yaml, sub/c.yaml]\n");
        write("a.yaml", "include: [b.yaml]\n");
        write("b.yaml", "atoms: []\n");
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        // Relative to its own directory, and b.yaml again.
        write("sub/c.yaml", "include: [../b.yaml]\n");

        let names: Vec<_> = included_files(&dir.path().join("root.yaml"))
            .unwrap()
            .iter()
            .map(|f| f.strip_prefix(dir.path()).unwrap().to_path_buf())
            .collect();
        let expected: Vec<PathBuf> = ["a.yaml", "b.yaml", "sub/c.yaml"].map(PathBuf::from).into();
        assert_eq!(names, expected);
    }
}
