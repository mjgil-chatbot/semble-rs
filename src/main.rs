use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use semble_rs::agent::SEMBLE_SEARCH_AGENT;
use semble_rs::stats::format_savings_report;
use semble_rs::types::{Result, SearchMode, SembleError};
use semble_rs::utils::{format_results, is_git_url, resolve_chunk};
use semble_rs::SembleIndex;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let output = run_with_args(&args)?;
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}

fn run_with_args(args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Ok(help_text());
    }
    match args[0].as_str() {
        "search" => cmd_search(&args[1..]),
        "find-related" => cmd_find_related(&args[1..]),
        "init" => cmd_init(&args[1..]),
        "savings" => cmd_savings(&args[1..]),
        "-h" | "--help" | "help" => Ok(help_text()),
        other => Err(SembleError::Message(format!("unknown command: {other}"))),
    }
}

fn cmd_search(args: &[String]) -> Result<String> {
    if args.is_empty() || has_help(args) {
        return Ok(search_help_text());
    }
    let query = args[0].clone();
    let mut path = ".".to_string();
    let mut top_k = 5usize;
    let mut mode = SearchMode::Hybrid;
    let mut include_text_files = false;
    let mut i = 1;
    if i < args.len() && !args[i].starts_with('-') {
        path = args[i].clone();
        i += 1;
    }
    while i < args.len() {
        match args[i].as_str() {
            "-k" | "--top-k" => {
                i += 1;
                top_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(5);
            }
            "-m" | "--mode" => {
                i += 1;
                let value = args.get(i).cloned().unwrap_or_else(|| "hybrid".to_string());
                mode = SearchMode::from_str(&value).map_err(SembleError::InvalidMode)?;
            }
            "--include-text-files" => include_text_files = true,
            other => {
                return Err(SembleError::Message(format!(
                    "unknown search option: {other}"
                )));
            }
        }
        i += 1;
    }

    let index = load_index(&path, include_text_files)?;
    let results = index.search(&query, top_k, mode)?;
    if results.is_empty() {
        Ok("No results found.".to_string())
    } else {
        Ok(format_results(
            &format!("Search results for: {} (mode={mode})", python_repr(&query)),
            &results,
        ))
    }
}

fn cmd_find_related(args: &[String]) -> Result<String> {
    if args.len() < 2 || has_help(args) {
        return Ok(find_related_help_text());
    }
    let file_path = args[0].clone();
    let line: usize = args[1]
        .parse()
        .map_err(|_| SembleError::Message(format!("invalid line number: {}", args[1])))?;
    let mut path = ".".to_string();
    let mut top_k = 5usize;
    let mut include_text_files = false;
    let mut i = 2;
    if i < args.len() && !args[i].starts_with('-') {
        path = args[i].clone();
        i += 1;
    }
    while i < args.len() {
        match args[i].as_str() {
            "-k" | "--top-k" => {
                i += 1;
                top_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(5);
            }
            "--include-text-files" => include_text_files = true,
            other => {
                return Err(SembleError::Message(format!(
                    "unknown find-related option: {other}"
                )));
            }
        }
        i += 1;
    }

    let index = load_index(&path, include_text_files)?;
    let Some(chunk) = resolve_chunk(&index.chunks, &file_path, line) else {
        return Err(SembleError::Message(format!(
            "No chunk found at {file_path}:{line}."
        )));
    };
    let results = index.find_related(chunk, top_k);
    if results.is_empty() {
        Ok(format!("No related chunks found for {file_path}:{line}."))
    } else {
        Ok(format_results(
            &format!("Chunks related to {file_path}:{line}"),
            &results,
        ))
    }
}

fn cmd_init(args: &[String]) -> Result<String> {
    let force = args.iter().any(|a| a == "--force" || a == "-f");
    let dest = Path::new(".claude").join("agents").join("semble-search.md");
    if dest.exists() && !force {
        return Err(SembleError::Message(format!(
            "{} already exists. Run with --force to overwrite.",
            dest.display()
        )));
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&dest, SEMBLE_SEARCH_AGENT)?;
    Ok(format!("Created {}", dest.display()))
}

fn cmd_savings(args: &[String]) -> Result<String> {
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    Ok(format_savings_report(None, verbose))
}

fn load_index(path: &str, include_text_files: bool) -> Result<SembleIndex> {
    if is_git_url(path) {
        if !(path.starts_with("https://")
            || path.starts_with("http://")
            || path.starts_with("file://"))
        {
            return Err(SembleError::Git(format!(
                "for safety, this CLI accepts only https://, http://, file://, or local paths for git cloning; got {path:?}"
            )));
        }
        SembleIndex::from_git(path, None, include_text_files)
    } else {
        SembleIndex::from_path_with_options(PathBuf::from(path), include_text_files, None)
    }
}

fn has_help(args: &[String]) -> bool {
    args.iter().any(|a| a == "-h" || a == "--help")
}

fn python_repr(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('\'', "\\'");
    format!("'{escaped}'")
}

fn help_text() -> String {
    "semble — local hybrid code search for agents\n\n\
Usage:\n\
  semble search <query> [path] [--top-k N] [--mode hybrid|semantic|bm25] [--include-text-files]\n\
  semble find-related <file_path> <line> [path] [--top-k N] [--include-text-files]\n\
  semble init [--force]\n\
  semble savings [--verbose]\n\n\
When path is omitted, the current directory is indexed. Local paths and https/http/file git URLs are accepted."
        .to_string()
}

fn search_help_text() -> String {
    "Usage: semble search <query> [path] [--top-k N] [--mode hybrid|semantic|bm25] [--include-text-files]"
        .to_string()
}

fn find_related_help_text() -> String {
    "Usage: semble find-related <file_path> <line> [path] [--top-k N] [--include-text-files]"
        .to_string()
}

#[cfg(test)]
mod tests {
    use semble_rs::types::{Chunk, SearchMode, SearchResult};
    use semble_rs::utils::format_results;

    use super::{find_related_help_text, help_text, python_repr, run_with_args, search_help_text};

    fn chunk(path: &str, content: &str) -> Chunk {
        Chunk {
            content: content.to_string(),
            file_path: path.to_string(),
            start_line: 1,
            end_line: 2,
            language: Some("python".to_string()),
        }
    }

    #[test]
    fn search_output_matches_python_header_shape() {
        let result = SearchResult {
            chunk: chunk("src/foo.py", "def foo():\n    return 1"),
            score: 0.9,
            source: SearchMode::Hybrid,
        };
        let rendered = format_results("Search results for: 'query text' (mode=hybrid)", &[result]);
        assert!(rendered.starts_with(
            "Search results for: 'query text' (mode=hybrid)\n\n## 1. src/foo.py:1-2  [score=0.900]"
        ));
    }

    #[test]
    fn related_output_matches_python_header_shape() {
        let result = SearchResult {
            chunk: chunk("src/bar.py", "class Bar:\n    pass"),
            score: 0.8,
            source: SearchMode::Semantic,
        };
        let rendered = format_results("Chunks related to src/bar.py:1", &[result]);
        assert!(rendered
            .starts_with("Chunks related to src/bar.py:1\n\n## 1. src/bar.py:1-2  [score=0.800]"));
    }

    #[test]
    fn help_dispatch_returns_usage_text() {
        assert!(help_text().contains("semble search <query>"));
        assert!(search_help_text().contains("semble search <query>"));
        assert!(find_related_help_text().contains("semble find-related <file_path> <line>"));
    }

    #[test]
    fn no_args_returns_help() {
        assert!(run_with_args(&[])
            .unwrap()
            .contains("local hybrid code search"));
    }

    #[test]
    fn python_repr_matches_single_quoted_cli_headers() {
        assert_eq!(python_repr("query text"), "'query text'");
        assert_eq!(python_repr("it's"), "'it\\'s'");
    }
}
