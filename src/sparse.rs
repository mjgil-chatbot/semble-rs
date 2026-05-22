use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::tokens::tokenize;
use crate::types::Chunk;

/// bm25s.BM25 default parameters: method="lucene", k1=1.5, b=0.75.
const K1: f32 = 1.5;
const B: f32 = 0.75;

#[derive(Clone, Debug, Default)]
pub struct Bm25Index {
    doc_len: Vec<usize>,
    avg_doc_len: f32,
    idf: HashMap<String, f32>,
    postings: HashMap<String, Vec<(usize, usize)>>,
}

impl Bm25Index {
    pub fn new(docs: &[Vec<String>]) -> Self {
        let n_docs = docs.len();
        let doc_len: Vec<usize> = docs.iter().map(Vec::len).collect();
        let avg_doc_len = if n_docs == 0 {
            0.0
        } else {
            doc_len.iter().sum::<usize>() as f32 / n_docs as f32
        };

        let mut postings_maps: HashMap<String, HashMap<usize, usize>> = HashMap::new();
        for (doc_id, doc) in docs.iter().enumerate() {
            for token in doc {
                *postings_maps
                    .entry(token.clone())
                    .or_default()
                    .entry(doc_id)
                    .or_default() += 1;
            }
        }

        let mut idf = HashMap::new();
        let mut postings = HashMap::new();
        for (term, doc_freqs) in postings_maps {
            let df = doc_freqs.len() as f32;
            let n = n_docs as f32;
            // bm25s Lucene IDF: log(1 + (N - df + 0.5) / (df + 0.5))
            idf.insert(term.clone(), (1.0 + (n - df + 0.5) / (df + 0.5)).ln());
            postings.insert(term, doc_freqs.into_iter().collect());
        }

        Self {
            doc_len,
            avg_doc_len,
            idf,
            postings,
        }
    }

    pub fn scores(&self, query_tokens: &[String], selector: Option<&[usize]>) -> Vec<f32> {
        let mut scores = vec![0.0; self.doc_len.len()];
        if query_tokens.is_empty() || self.doc_len.is_empty() {
            return scores;
        }

        let allowed: Option<HashSet<usize>> = selector.map(|s| s.iter().copied().collect());
        let avg = self.avg_doc_len.max(1e-12);
        for token in query_tokens {
            let Some(postings) = self.postings.get(token) else {
                continue;
            };
            let idf = *self.idf.get(token).unwrap_or(&0.0);
            for &(doc_id, tf) in postings {
                if allowed.as_ref().is_some_and(|set| !set.contains(&doc_id)) {
                    continue;
                }
                let dl = self.doc_len[doc_id] as f32;
                let tf = tf as f32;
                // bm25s Lucene term-frequency component:
                // tf / (k1 * ((1 - b) + b * dl / avgdl) + tf)
                let tfc = tf / (K1 * ((1.0 - B) + B * dl / avg) + tf);
                scores[doc_id] += idf * tfc;
            }
        }
        scores
    }
}

/// Python `enrich_for_bm25`: content + file stem twice + up to last 3 parent dirs.
pub fn enrich_for_bm25(chunk: &Chunk) -> String {
    let path = Path::new(&chunk.file_path);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let mut dir_parts: Vec<String> = path
        .parent()
        .map(|p| {
            p.components()
                .filter_map(|c| c.as_os_str().to_str())
                .filter(|part| *part != "." && *part != "/")
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default();
    if dir_parts.len() > 3 {
        dir_parts = dir_parts.split_off(dir_parts.len() - 3);
    }
    format!(
        "{} {} {} {}",
        chunk.content,
        stem,
        stem,
        dir_parts.join(" ")
    )
}

pub fn bm25_docs(chunks: &[Chunk]) -> Vec<Vec<String>> {
    chunks
        .iter()
        .map(|chunk| tokenize(&enrich_for_bm25(chunk)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lucene_scores_matching_docs() {
        let docs = vec![tokenize("authenticate token"), tokenize("format name")];
        let idx = Bm25Index::new(&docs);
        let scores = idx.scores(&tokenize("authenticate"), None);
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn enriches_with_file_stem_twice_and_parent_tail() {
        let chunk = Chunk {
            content: "fn run() {}".to_string(),
            file_path: "a/b/c/auth_service.rs".to_string(),
            start_line: 1,
            end_line: 1,
            language: Some("rust".to_string()),
        };
        let enriched = enrich_for_bm25(&chunk);
        assert!(enriched.contains("auth_service auth_service a b c"));
    }
}
