use crate::types::{Chunk, SearchResult};

const GIT_URL_SCHEMES: &[&str] = &[
    "https://",
    "http://",
    "ssh://",
    "git://",
    "git+ssh://",
    "file://",
];

pub fn is_git_url(path: &str) -> bool {
    GIT_URL_SCHEMES
        .iter()
        .any(|scheme| path.starts_with(scheme))
        || looks_like_scp_git_url(path)
}

fn looks_like_scp_git_url(path: &str) -> bool {
    let Some((user_host, rest)) = path.split_once(':') else {
        return false;
    };
    !rest.starts_with('/')
        && user_host.contains('@')
        && user_host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '@' | '.' | '-' | '_'))
}

pub fn resolve_chunk<'a>(chunks: &'a [Chunk], file_path: &str, line: usize) -> Option<&'a Chunk> {
    chunks.iter().find(|chunk| {
        chunk.file_path == file_path && chunk.start_line <= line && line <= chunk.end_line
    })
}

pub fn format_results(header: &str, results: &[SearchResult]) -> String {
    let mut lines = vec![header.to_string(), String::new()];
    for (i, result) in results.iter().enumerate() {
        lines.push(format!(
            "## {}. {}  [score={:.3}]",
            i + 1,
            result.chunk.location(),
            result.score
        ));
        lines.push("```".to_string());
        lines.push(result.chunk.content.trim().to_string());
        lines.push("```".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_git_urls() {
        assert!(is_git_url("https://github.com/org/repo"));
        assert!(is_git_url("git@github.com:org/repo.git"));
        assert!(!is_git_url("./local:path"));
    }
}
