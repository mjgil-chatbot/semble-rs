use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::{CallType, SearchResult};

#[derive(Clone, Debug, Default)]
struct BucketStats {
    calls: usize,
    snippet_chars: usize,
    file_chars: usize,
    saved_chars: usize,
}

impl BucketStats {
    fn add(&mut self, snippet_chars: usize, file_chars: usize) {
        self.calls += 1;
        self.snippet_chars += snippet_chars;
        self.file_chars += file_chars;
        self.saved_chars += file_chars.saturating_sub(snippet_chars);
    }
}

pub fn stats_file() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".semble").join("savings.jsonl")
}

pub fn save_search_stats(
    results: &[SearchResult],
    call_type: CallType,
    file_sizes: &BTreeMap<String, usize>,
) {
    let snippet_chars: usize = results
        .iter()
        .map(|result| result.chunk.content.len())
        .sum();
    let mut seen = std::collections::BTreeSet::new();
    let file_chars: usize = results
        .iter()
        .filter_map(|result| {
            if seen.insert(result.chunk.file_path.clone()) {
                file_sizes.get(&result.chunk.file_path).copied()
            } else {
                None
            }
        })
        .sum();
    let ts = now_secs();
    let record = format!(
        "{{\"ts\":{ts},\"call\":\"{}\",\"results\":{},\"snippet_chars\":{},\"file_chars\":{}}}\n",
        call_type.as_str(),
        results.len(),
        snippet_chars,
        file_chars
    );

    let path = stats_file();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(record.as_bytes());
    }
}

pub fn format_savings_report(path: Option<&Path>, verbose: bool) -> String {
    let path_buf;
    let path = if let Some(path) = path {
        path
    } else {
        path_buf = stats_file();
        path_buf.as_path()
    };
    if !path.exists() {
        return "No stats yet. Run a search first.".to_string();
    }

    let now = now_secs();
    let day = 86_400_u64;
    let seven_days = day * 7;
    let mut buckets: BTreeMap<&str, BucketStats> = BTreeMap::from([
        ("Today", BucketStats::default()),
        ("Last 7 days", BucketStats::default()),
        ("All time", BucketStats::default()),
    ]);
    let mut call_counts: BTreeMap<String, usize> = BTreeMap::new();

    let Ok(file) = fs::File::open(path) else {
        return "No stats yet. Run a search first.".to_string();
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let ts = extract_u64(&line, "ts").unwrap_or(0);
        let snippet_chars = extract_u64(&line, "snippet_chars").unwrap_or(0) as usize;
        let file_chars = extract_u64(&line, "file_chars").unwrap_or(0) as usize;
        let call = extract_string(&line, "call").unwrap_or_else(|| "unknown".to_string());
        *call_counts.entry(call).or_default() += 1;
        buckets
            .get_mut("All time")
            .unwrap()
            .add(snippet_chars, file_chars);
        if now.saturating_sub(ts) <= seven_days {
            buckets
                .get_mut("Last 7 days")
                .unwrap()
                .add(snippet_chars, file_chars);
        }
        if now.saturating_sub(ts) <= day {
            buckets
                .get_mut("Today")
                .unwrap()
                .add(snippet_chars, file_chars);
        }
    }

    let heavy_line = "  ════════════════════════════════════════════════════════════════";
    let light_line = "  ────────────────────────────────────────────────────────────────";
    let mut lines = vec![
        "  Semble Token Savings".to_string(),
        heavy_line.to_string(),
        format!("  {:<12}  {:<6}  Savings", "Period", "Calls"),
        light_line.to_string(),
    ];

    for label in ["Today", "Last 7 days", "All time"] {
        let b = buckets.get(label).unwrap();
        let saved_tokens = b.saved_chars / 4;
        let pct = if b.file_chars > 0 {
            (b.saved_chars as f32 / b.file_chars as f32 * 100.0).round() as usize
        } else {
            0
        };
        let filled = ((pct as f32 / 100.0) * 16.0).round() as usize;
        let bar = format!(
            "{}{}",
            "█".repeat(filled.min(16)),
            "░".repeat(16 - filled.min(16))
        );
        lines.push(format!(
            "  {:<12}  {:<6}  [{}]  ~{} tokens ({}%)",
            label,
            format_count(b.calls),
            bar,
            format_tokens(saved_tokens),
            pct
        ));
    }

    if verbose && !call_counts.is_empty() {
        lines.push(String::new());
        lines.push("  Usage Breakdown".to_string());
        lines.push(light_line.to_string());
        lines.push(format!("  {:<16}  Calls", "Call type"));
        for (call, count) in call_counts {
            lines.push(format!("  {:<16}  {}", call, format_count(count)));
        }
        lines.push(heavy_line.to_string());
    }
    lines.push(String::new());
    lines.join("\n")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn format_count(n: usize) -> String {
    if n >= 1_000 {
        format!("{:.1}k", n as f32 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_tokens(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f32 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn extract_u64(line: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_string(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_missing_stats() {
        let tmp = std::env::temp_dir().join("semble-rs-missing-stats.jsonl");
        let _ = fs::remove_file(&tmp);
        assert!(format_savings_report(Some(&tmp), false).contains("No stats"));
    }
}
