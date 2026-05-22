use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use regex::{Regex, RegexBuilder};

use crate::tokens::split_identifier;
use crate::types::Chunk;

const ALPHA_SYMBOL: f32 = 0.3;
const ALPHA_NL: f32 = 0.5;
const DEFINITION_BOOST_MULTIPLIER: f32 = 3.0;
const STEM_BOOST_MULTIPLIER: f32 = 1.0;
const FILE_COHERENCE_BOOST_FRAC: f32 = 0.2;
const EMBEDDED_STEM_MIN_LEN: usize = 4;
const EMBEDDED_SYMBOL_BOOST_SCALE: f32 = 0.5;

const STRONG_PENALTY: f32 = 0.3;
const MODERATE_PENALTY: f32 = 0.5;
const MILD_PENALTY: f32 = 0.7;
const FILE_SATURATION_THRESHOLD: usize = 1;
const FILE_SATURATION_DECAY: f32 = 0.5;

static SYMBOL_QUERY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+|_[A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*|[A-Z][A-Za-z0-9]*)$",
    )
    .expect("valid symbol regex")
});

static EMBEDDED_SYMBOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b(?:[A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*|[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]+)\b",
    )
    .expect("valid embedded-symbol regex")
});

static KEYWORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z_][a-zA-Z0-9_]*").expect("valid keyword regex"));

static TEST_FILE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?:^|/)(?:test_[^/]*\.py|[^/]*_test\.py|[^/]*_test\.go|[^/]*Tests?\.java|[^/]*Test\.php|[^/]*_spec\.rb|[^/]*_test\.rb|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?|[^/]*Tests?\.kt|[^/]*Spec\.kt|[^/]*Tests?\.swift|[^/]*Spec\.swift|[^/]*Tests?\.cs|test_[^/]*\.cpp|[^/]*_test\.cpp|test_[^/]*\.c|[^/]*_test\.c|[^/]*Spec\.scala|[^/]*Suite\.scala|[^/]*Test\.scala|[^/]*_test\.dart|test_[^/]*\.dart|[^/]*_spec\.lua|[^/]*_test\.lua|test_[^/]*\.lua|test_helpers?[^/]*\.\w+)$",
    )
    .expect("valid test-file regex")
});
static TEST_DIR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)").unwrap());
static COMPAT_DIR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)").unwrap());
static EXAMPLES_DIR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)").unwrap());
static TYPE_DEFS_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\.d\.ts$").unwrap());

const DEFINITION_KEYWORDS: &[&str] = &[
    "class",
    "module",
    "defmodule",
    "def",
    "interface",
    "struct",
    "enum",
    "trait",
    "type",
    "func",
    "function",
    "object",
    "abstract class",
    "data class",
    "fn",
    "fun",
    "package",
    "namespace",
    "protocol",
    "record",
    "typedef",
];

const SQL_DEFINITION_KEYWORDS: &[&str] = &[
    "CREATE TABLE",
    "CREATE VIEW",
    "CREATE PROCEDURE",
    "CREATE FUNCTION",
];

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "has", "have",
    "how", "if", "in", "is", "it", "not", "of", "on", "or", "the", "to", "was", "what", "when",
    "where", "which", "who", "why", "with",
];

pub fn resolve_alpha(query: &str, explicit: Option<f32>) -> f32 {
    explicit.unwrap_or_else(|| {
        if is_symbol_query(query) {
            ALPHA_SYMBOL
        } else {
            ALPHA_NL
        }
    })
}

pub fn is_symbol_query(query: &str) -> bool {
    SYMBOL_QUERY_RE.is_match(query.trim())
}

pub fn apply_query_boost(
    combined_scores: HashMap<usize, f32>,
    query: &str,
    all_chunks: &[Chunk],
) -> HashMap<usize, f32> {
    if combined_scores.is_empty() {
        return combined_scores;
    }

    let max_score = combined_scores
        .values()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut boosted = combined_scores;

    if is_symbol_query(query) {
        boost_symbol_definitions(&mut boosted, query, max_score, all_chunks);
    } else {
        boost_stem_matches(&mut boosted, query, max_score, all_chunks);
        boost_embedded_symbols(&mut boosted, query, max_score, all_chunks);
    }

    boosted
}

pub fn boost_multi_chunk_files(scores: &mut HashMap<usize, f32>, chunks: &[Chunk]) {
    if scores.is_empty() {
        return;
    }
    let max_score = scores.values().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_score == 0.0 || !max_score.is_finite() {
        return;
    }

    let mut file_sum: HashMap<&str, f32> = HashMap::new();
    let mut best_chunk: HashMap<&str, usize> = HashMap::new();
    for (&idx, &score) in scores.iter() {
        let Some(chunk) = chunks.get(idx) else {
            continue;
        };
        *file_sum.entry(chunk.file_path.as_str()).or_default() += score;
        let replace = match best_chunk
            .get(chunk.file_path.as_str())
            .and_then(|best| scores.get(best))
        {
            Some(best_score) => score > *best_score,
            None => true,
        };
        if replace {
            best_chunk.insert(chunk.file_path.as_str(), idx);
        }
    }

    let max_file_sum = file_sum.values().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_file_sum == 0.0 || !max_file_sum.is_finite() {
        return;
    }
    let boost_unit = max_score * FILE_COHERENCE_BOOST_FRAC;
    for (file_path, idx) in best_chunk {
        if let Some(score) = scores.get_mut(&idx) {
            *score += boost_unit * file_sum[file_path] / max_file_sum;
        }
    }
}

fn extract_symbol_name(query: &str) -> String {
    for separator in ["::", "\\", "->", "."] {
        if let Some((_, leaf)) = query.rsplit_once(separator) {
            return leaf.to_string();
        }
    }
    query.trim().to_string()
}

fn definition_pattern(symbol_name: &str, sql: bool) -> Regex {
    let escaped = regex::escape(symbol_name);
    let ns_prefix = r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.|::))*";
    let keywords = if sql {
        SQL_DEFINITION_KEYWORDS
    } else {
        DEFINITION_KEYWORDS
    };
    let body = keywords
        .iter()
        .map(|k| regex::escape(k))
        .collect::<Vec<_>>()
        .join("|");
    let pattern = format!(
        r"(?m)(?:^|\s)(?:{})\s+{}{}(?:\s|[<\(\{{:\[;]|$)",
        body, ns_prefix, escaped
    );
    if sql {
        RegexBuilder::new(&pattern)
            .case_insensitive(true)
            .multi_line(true)
            .build()
            .expect("valid SQL definition regex")
    } else {
        Regex::new(&pattern).expect("valid definition regex")
    }
}

fn chunk_defines_symbol(chunk: &Chunk, symbol_name: &str) -> bool {
    definition_pattern(symbol_name, false).is_match(&chunk.content)
        || definition_pattern(symbol_name, true).is_match(&chunk.content)
}

fn stem_matches(stem: &str, name: &str) -> bool {
    let stem_norm = stem.replace('_', "");
    stem == name
        || stem_norm == name
        || stem.trim_end_matches('s') == name
        || stem_norm.trim_end_matches('s') == name
}

fn definition_tier(chunk: &Chunk, names: &HashSet<String>, boost_unit: f32) -> f32 {
    if !names.iter().any(|name| chunk_defines_symbol(chunk, name)) {
        return 0.0;
    }
    let stem = Path::new(&chunk.file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if names
        .iter()
        .any(|name| stem_matches(&stem, &name.to_ascii_lowercase()))
    {
        boost_unit * 1.5
    } else {
        boost_unit
    }
}

fn scan_non_candidates<F>(
    boosted: &mut HashMap<usize, f32>,
    names: &HashSet<String>,
    boost_unit: f32,
    all_chunks: &[Chunk],
    stem_ok: F,
) where
    F: Fn(&str) -> bool,
{
    for (idx, chunk) in all_chunks.iter().enumerate() {
        if boosted.contains_key(&idx) {
            continue;
        }
        let stem = Path::new(&chunk.file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !stem_ok(&stem) {
            continue;
        }
        let tier = definition_tier(chunk, names, boost_unit);
        if tier > 0.0 {
            boosted.insert(idx, tier);
        }
    }
}

fn boost_symbol_definitions(
    boosted: &mut HashMap<usize, f32>,
    query: &str,
    max_score: f32,
    all_chunks: &[Chunk],
) {
    let symbol_name = extract_symbol_name(query);
    let mut names = HashSet::from([symbol_name.clone()]);
    if symbol_name != query.trim() {
        names.insert(query.trim().to_string());
    }

    let boost_unit = max_score * DEFINITION_BOOST_MULTIPLIER;
    for idx in boosted.keys().copied().collect::<Vec<_>>() {
        if let Some(chunk) = all_chunks.get(idx) {
            let tier = definition_tier(chunk, &names, boost_unit);
            if tier > 0.0 {
                if let Some(score) = boosted.get_mut(&idx) {
                    *score += tier;
                }
            }
        }
    }

    let symbol_lower = symbol_name.to_ascii_lowercase();
    scan_non_candidates(boosted, &names, boost_unit, all_chunks, |stem| {
        stem_matches(stem, &symbol_lower)
    });
}

fn boost_embedded_symbols(
    boosted: &mut HashMap<usize, f32>,
    query: &str,
    max_score: f32,
    all_chunks: &[Chunk],
) {
    let names: HashSet<String> = EMBEDDED_SYMBOL_RE
        .find_iter(query)
        .map(|m| m.as_str().to_string())
        .collect();
    if names.is_empty() {
        return;
    }

    let boost_unit = max_score * DEFINITION_BOOST_MULTIPLIER * EMBEDDED_SYMBOL_BOOST_SCALE;
    for idx in boosted.keys().copied().collect::<Vec<_>>() {
        if let Some(chunk) = all_chunks.get(idx) {
            let tier = definition_tier(chunk, &names, boost_unit);
            if tier > 0.0 {
                if let Some(score) = boosted.get_mut(&idx) {
                    *score += tier;
                }
            }
        }
    }

    let symbols_lower: HashSet<String> = names.iter().map(|s| s.to_ascii_lowercase()).collect();
    for (idx, chunk) in all_chunks.iter().enumerate() {
        if boosted.contains_key(&idx) {
            continue;
        }
        let stem = Path::new(&chunk.file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let stem_norm = stem.replace('_', "");
        if !symbols_lower.iter().any(|symbol_lower| {
            stem == *symbol_lower
                || stem_norm == *symbol_lower
                || (stem.len() >= EMBEDDED_STEM_MIN_LEN && symbol_lower.starts_with(&stem))
                || (stem_norm.len() >= EMBEDDED_STEM_MIN_LEN
                    && symbol_lower.starts_with(&stem_norm))
        }) {
            continue;
        }
        let tier = definition_tier(chunk, &names, boost_unit);
        if tier > 0.0 {
            boosted.insert(idx, tier);
        }
    }
}

fn count_keyword_matches(keywords: &HashSet<String>, parts: &HashSet<String>) -> usize {
    let exact_count = keywords.intersection(parts).count();
    if exact_count == keywords.len() {
        return exact_count;
    }
    let mut n_matches = exact_count;
    for keyword in keywords.difference(parts) {
        for part in parts {
            let (shorter, longer) = if keyword.len() <= part.len() {
                (keyword.as_str(), part.as_str())
            } else {
                (part.as_str(), keyword.as_str())
            };
            if shorter.len() >= 3 && longer.starts_with(shorter) {
                n_matches += 1;
                break;
            }
        }
    }
    n_matches
}

fn boost_stem_matches(
    boosted: &mut HashMap<usize, f32>,
    query: &str,
    max_score: f32,
    all_chunks: &[Chunk],
) {
    let keywords: HashSet<String> = KEYWORD_RE
        .find_iter(query)
        .map(|m| m.as_str().to_ascii_lowercase())
        .filter(|word| word.len() > 2 && !STOPWORDS.contains(&word.as_str()))
        .collect();
    if keywords.is_empty() {
        return;
    }

    let boost = max_score * STEM_BOOST_MULTIPLIER;
    let mut path_cache: HashMap<&str, HashSet<String>> = HashMap::new();
    for idx in boosted.keys().copied().collect::<Vec<_>>() {
        let Some(chunk) = all_chunks.get(idx) else {
            continue;
        };
        let parts = path_cache
            .entry(chunk.file_path.as_str())
            .or_insert_with(|| {
                let path = Path::new(&chunk.file_path);
                let mut parts: HashSet<String> = HashSet::new();
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    parts.extend(split_identifier(stem));
                }
                if let Some(parent) = path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                {
                    if parent != "." && parent != "/" && parent != ".." {
                        parts.extend(split_identifier(parent));
                    }
                }
                parts
            });
        let n_matches = count_keyword_matches(&keywords, parts);
        if n_matches > 0 {
            let match_ratio = n_matches as f32 / keywords.len() as f32;
            if match_ratio >= 0.10 {
                if let Some(score) = boosted.get_mut(&idx) {
                    *score += boost * match_ratio;
                }
            }
        }
    }
}

pub fn rerank_topk(
    scores: &HashMap<usize, f32>,
    chunks: &[Chunk],
    top_k: usize,
    penalise_paths: bool,
) -> Vec<(usize, f32)> {
    if scores.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut penalty_cache: HashMap<&str, f32> = HashMap::new();
    let mut penalised: HashMap<usize, f32> = HashMap::new();
    for (&idx, &score) in scores {
        if let Some(chunk) = chunks.get(idx) {
            let penalty = if penalise_paths {
                *penalty_cache
                    .entry(chunk.file_path.as_str())
                    .or_insert_with(|| file_path_penalty(&chunk.file_path))
            } else {
                1.0
            };
            penalised.insert(idx, score * penalty);
        }
    }

    let mut ranked: Vec<usize> = penalised.keys().copied().collect();
    ranked.sort_by(|&a, &b| {
        let sb = penalised.get(&b).copied().unwrap_or(0.0);
        let sa = penalised.get(&a).copied().unwrap_or(0.0);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| chunks[a].start_line.cmp(&chunks[b].start_line))
            .then_with(|| chunks[a].file_path.cmp(&chunks[b].file_path))
    });

    let mut file_selected: HashMap<&str, usize> = HashMap::new();
    let mut selected: Vec<(f32, usize)> = Vec::new();
    let mut min_selected = f32::INFINITY;

    for idx in ranked {
        let pen_score = penalised[&idx];
        if selected.len() >= top_k && pen_score <= min_selected {
            break;
        }

        let chunk = &chunks[idx];
        let already = *file_selected.get(chunk.file_path.as_str()).unwrap_or(&0);
        let mut eff_score = pen_score;
        if already >= FILE_SATURATION_THRESHOLD {
            let excess = already - FILE_SATURATION_THRESHOLD + 1;
            eff_score *= FILE_SATURATION_DECAY.powi(excess as i32);
        }

        selected.push((eff_score, idx));
        file_selected.insert(chunk.file_path.as_str(), already + 1);

        if selected.len() >= top_k {
            min_selected = selected
                .iter()
                .map(|(score, _)| *score)
                .fold(f32::INFINITY, f32::min);
        }
    }

    selected.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| chunks[a.1].start_line.cmp(&chunks[b.1].start_line))
            .then_with(|| chunks[a.1].file_path.cmp(&chunks[b.1].file_path))
    });
    selected
        .into_iter()
        .take(top_k)
        .map(|(score, idx)| (idx, score))
        .collect()
}

pub fn file_path_penalty(file_path: &str) -> f32 {
    let normalised = file_path.replace('\\', "/");
    let name = Path::new(file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let mut penalty = 1.0;
    if TEST_FILE_RE.is_match(&normalised) || TEST_DIR_RE.is_match(&normalised) {
        penalty *= STRONG_PENALTY;
    }
    if name == "__init__.py" || name == "package-info.java" {
        penalty *= MODERATE_PENALTY;
    }
    if COMPAT_DIR_RE.is_match(&normalised) {
        penalty *= STRONG_PENALTY;
    }
    if EXAMPLES_DIR_RE.is_match(&normalised) {
        penalty *= STRONG_PENALTY;
    }
    if TYPE_DEFS_RE.is_match(&normalised) {
        penalty *= MILD_PENALTY;
    }
    penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(content: &str, path: &str) -> Chunk {
        Chunk {
            content: content.to_string(),
            file_path: path.to_string(),
            start_line: 1,
            end_line: 1,
            language: None,
        }
    }

    #[test]
    fn resolves_alpha_like_python() {
        assert_eq!(resolve_alpha("MyService", None), 0.3);
        assert_eq!(resolve_alpha("how does routing work", None), 0.5);
        assert_eq!(resolve_alpha("MyService", Some(0.7)), 0.7);
    }

    #[test]
    fn boosts_defining_symbol_and_scans_non_candidates() {
        let chunks = vec![
            chunk("class MyService:\n    pass", "src/my_service.py"),
            chunk("x = 1", "src/other.py"),
        ];
        let scores = HashMap::from([(1usize, 0.5)]);
        let boosted = apply_query_boost(scores, "MyService", &chunks);
        assert!(boosted.contains_key(&0));
        assert!(boosted[&0] > boosted[&1]);
    }

    #[test]
    fn penalises_tests_and_reexports() {
        assert!(file_path_penalty("tests/test_auth.py") < 1.0);
        assert!(file_path_penalty("src/__init__.py") < 1.0);
    }
}
