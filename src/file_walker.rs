use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_IGNORED_PATTERNS: &[&str] = &[
    ".git/",
    ".hg/",
    ".svn/",
    "__pycache__/",
    "node_modules/",
    ".venv/",
    "venv/",
    ".tox/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".cache/",
    ".semble/",
    ".next/",
    "dist/",
    "build/",
    ".eggs/",
    "target/",
    "target-*/",
    "*-target/",
    ".lightweight-test/",
];

#[derive(Clone, Debug)]
struct IgnorePattern {
    negated: bool,
    pattern: String,
}

/// Yield files under `root` matching `extensions`, skipping common generated
/// directories and a practical subset of `.gitignore` / `.sembleignore` rules.
pub fn walk_files(
    root: &Path,
    extensions: &[String],
    ignore: Option<&[String]>,
) -> std::io::Result<Vec<PathBuf>> {
    let ext_set: BTreeSet<String> = extensions.iter().map(|e| e.to_ascii_lowercase()).collect();
    let mut patterns = Vec::new();
    for pattern in DEFAULT_IGNORED_PATTERNS {
        patterns.push(IgnorePattern {
            negated: false,
            pattern: (*pattern).to_string(),
        });
    }
    if let Some(extra) = ignore {
        for pat in extra {
            patterns.push(parse_pattern(pat));
        }
    }
    let mut out = Vec::new();
    walk_inner(root, root, &ext_set, &patterns, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk_inner(
    root: &Path,
    dir: &Path,
    extensions: &BTreeSet<String>,
    inherited: &[IgnorePattern],
    out: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    let mut patterns = inherited.to_vec();
    patterns.extend(load_ignore_file(&dir.join(".gitignore"))?);
    patterns.extend(load_ignore_file(&dir.join(".sembleignore"))?);

    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)? {
        entries.push(entry?);
    }
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let meta = match fs::symlink_metadata(&path) {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        let is_dir = meta.is_dir();
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let (ignored, found_by_negated_extension) = is_ignored(rel, is_dir, &patterns);
        if ignored {
            continue;
        }
        if is_dir {
            walk_inner(root, &path, extensions, &patterns, out)?;
        } else if meta.is_file() {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| format!(".{}", s.to_ascii_lowercase()));
            if found_by_negated_extension || ext.as_ref().is_some_and(|e| extensions.contains(e)) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn load_ignore_file(path: &Path) -> std::io::Result<Vec<IgnorePattern>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(parse_pattern)
        .collect())
}

fn parse_pattern(raw: &str) -> IgnorePattern {
    let (negated, pattern) = raw
        .strip_prefix('!')
        .map_or((false, raw), |rest| (true, rest));
    IgnorePattern {
        negated,
        pattern: pattern.trim().to_string(),
    }
}

fn is_ignored(rel: &Path, is_dir: bool, patterns: &[IgnorePattern]) -> (bool, bool) {
    let mut ignored = false;
    let mut found_by_negated_extension = false;
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    for pat in patterns {
        if pattern_matches(&pat.pattern, &rel_str, is_dir) {
            ignored = !pat.negated;
            found_by_negated_extension = pat.negated && has_extension_suffix(&pat.pattern);
        }
    }
    (ignored, found_by_negated_extension)
}

fn has_extension_suffix(pattern: &str) -> bool {
    let trimmed = pattern.trim_end_matches('/');
    Path::new(trimmed).extension().is_some()
}

fn pattern_matches(pattern: &str, rel: &str, is_dir: bool) -> bool {
    let pat = pattern.trim();
    if pat.is_empty() {
        return false;
    }
    let pat = pat.trim_start_matches('/');
    if let Some(dir_pat) = pat.strip_suffix('/') {
        if !is_dir {
            return false;
        }
        if dir_pat.contains('*') {
            return rel.split('/').any(|part| wildcard_match(dir_pat, part));
        }
        return rel == dir_pat
            || rel.starts_with(&format!("{dir_pat}/"))
            || rel.split('/').any(|part| part == dir_pat);
    }
    if let Some(suffix) = pat.strip_prefix("*.") {
        return rel.ends_with(&format!(".{suffix}"));
    }
    if pat.contains('*') {
        return wildcard_match(pat, rel);
    }
    rel == pat || rel.ends_with(&format!("/{pat}")) || rel.split('/').any(|part| part == pat)
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut rest = text;
    let starts_with_anchor = !pattern.starts_with('*');
    let ends_with_anchor = !pattern.ends_with('*');
    for (idx, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if idx == 0 && starts_with_anchor {
            if !rest.starts_with(part) {
                return false;
            }
            rest = &rest[part.len()..];
            continue;
        }
        if let Some(pos) = rest.find(part) {
            rest = &rest[pos + part.len()..];
        } else {
            return false;
        }
    }
    if ends_with_anchor {
        if let Some(last) = parts.last() {
            return text.ends_with(last);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn walks_supported_files() {
        let root = std::env::temp_dir().join(format!(
            "semble-rs-walk-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::write(root.join("a.py"), "x").unwrap();
        fs::write(root.join("node_modules/b.py"), "x").unwrap();
        let files = walk_files(&root, &[".py".to_string()], None).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name().unwrap(), "a.py");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ignores_generated_target_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "semble-rs-ignore-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target/debug/deps")).unwrap();
        fs::create_dir_all(root.join("target-codex-servo/debug/deps")).unwrap();
        fs::create_dir_all(root.join("real-site-harness-target/debug/deps")).unwrap();
        fs::create_dir_all(root.join(".lightweight-test/real-site-harness-target/debug/deps"))
            .unwrap();
        fs::write(root.join("src/app.py"), "print('source')").unwrap();
        fs::write(root.join("target/debug/deps/generated.d"), "ignored").unwrap();
        fs::write(
            root.join("target-codex-servo/debug/deps/generated.d"),
            "ignored",
        )
        .unwrap();
        fs::write(
            root.join("real-site-harness-target/debug/deps/generated.d"),
            "ignored",
        )
        .unwrap();
        fs::write(
            root.join(".lightweight-test/real-site-harness-target/debug/deps/generated.d"),
            "ignored",
        )
        .unwrap();

        let files = walk_files(&root, &[".py".to_string(), ".d".to_string()], None).unwrap();
        let rel_paths: Vec<String> = files
            .iter()
            .map(|path| {
                path.strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();

        assert_eq!(rel_paths, vec!["src/app.py"]);

        let _ = fs::remove_dir_all(root);
    }
}
