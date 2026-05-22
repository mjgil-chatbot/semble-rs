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
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return Ok(());
    }
    match args[0].as_str() {
        "search" => cmd_search(&args[1..]),
        "find-related" => cmd_find_related(&args[1..]),
        "init" => cmd_init(&args[1..]),
        "savings" => cmd_savings(&args[1..]),
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        other => Err(SembleError::InvalidMode(format!(
            "unknown command: {other}"
        ))),
    }
}

fn cmd_search(args: &[String]) -> Result<()> {
    if args.is_empty() || has_help(args) {
        print_search_help();
        return Ok(());
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
                return Err(SembleError::InvalidMode(format!(
                    "unknown search option: {other}"
                )));
            }
        }
        i += 1;
    }

    let index = load_index(&path, include_text_files)?;
    let results = index.search(&query, top_k, mode)?;
    if results.is_empty() {
        println!("No results found.");
    } else {
        println!(
            "{}",
            format_results(&format!("Search results for {query:?}"), &results)
        );
    }
    Ok(())
}

fn cmd_find_related(args: &[String]) -> Result<()> {
    if args.len() < 2 || has_help(args) {
        print_find_related_help();
        return Ok(());
    }
    let file_path = args[0].clone();
    let line: usize = args[1]
        .parse()
        .map_err(|_| SembleError::InvalidMode(format!("invalid line number: {}", args[1])))?;
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
                return Err(SembleError::InvalidMode(format!(
                    "unknown find-related option: {other}"
                )));
            }
        }
        i += 1;
    }

    let index = load_index(&path, include_text_files)?;
    let Some(chunk) = resolve_chunk(&index.chunks, &file_path, line) else {
        return Err(SembleError::InvalidMode(format!(
            "No chunk found at {file_path}:{line}"
        )));
    };
    let results = index.find_related_chunk(chunk, top_k);
    if results.is_empty() {
        println!("No related chunks found.");
    } else {
        println!(
            "{}",
            format_results(&format!("Related chunks for {file_path}:{line}"), &results)
        );
    }
    Ok(())
}

fn cmd_init(args: &[String]) -> Result<()> {
    let force = args.iter().any(|a| a == "--force" || a == "-f");
    let dest = Path::new(".claude").join("agents").join("semble-search.md");
    if dest.exists() && !force {
        return Err(SembleError::InvalidMode(format!(
            "{} already exists. Run with --force to overwrite.",
            dest.display()
        )));
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&dest, SEMBLE_SEARCH_AGENT)?;
    println!("Created {}", dest.display());
    Ok(())
}

fn cmd_savings(args: &[String]) -> Result<()> {
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    println!("{}", format_savings_report(None, verbose));
    Ok(())
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

fn print_help() {
    println!(
        "semble — local hybrid code search for agents\n\n\
Usage:\n\
  semble search <query> [path] [--top-k N] [--mode hybrid|semantic|bm25] [--include-text-files]\n\
  semble find-related <file_path> <line> [path] [--top-k N] [--include-text-files]\n\
  semble init [--force]\n\
  semble savings [--verbose]\n\n\
When path is omitted, the current directory is indexed. Local paths and https/http/file git URLs are accepted."
    );
}

fn print_search_help() {
    println!(
        "Usage: semble search <query> [path] [--top-k N] [--mode hybrid|semantic|bm25] [--include-text-files]"
    );
}

fn print_find_related_help() {
    println!(
        "Usage: semble find-related <file_path> <line> [path] [--top-k N] [--include-text-files]"
    );
}
