use std::collections::HashMap;

use crate::dense::{Encoder, SemanticIndex};
use crate::ranking::{apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha};
use crate::sparse::Bm25Index;
use crate::tokens::tokenize;
use crate::types::{Chunk, Result, SearchMode, SearchResult};

const RRF_K: f32 = 60.0;

fn rrf_scores(scores: &HashMap<usize, f32>) -> HashMap<usize, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }
    let mut ranked: Vec<(usize, f32)> = scores.iter().map(|(&idx, &score)| (idx, score)).collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .enumerate()
        .map(|(rank, (idx, _))| (idx, 1.0 / (RRF_K + (rank + 1) as f32)))
        .collect()
}

pub fn search_semantic(
    query: &str,
    model: &dyn Encoder,
    semantic_index: &SemanticIndex,
    chunks: &[Chunk],
    top_k: usize,
    selector: Option<&[usize]>,
) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let query_embedding = model.encode_one(query)?;
    let results = semantic_index
        .query(&query_embedding, top_k, selector)
        .into_iter()
        .filter_map(|(idx, distance)| {
            chunks.get(idx).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                score: 1.0 - distance,
                source: SearchMode::Semantic,
            })
        })
        .collect();
    Ok(results)
}

pub fn search_bm25(
    query: &str,
    bm25_index: &Bm25Index,
    chunks: &[Chunk],
    top_k: usize,
    selector: Option<&[usize]>,
) -> Vec<SearchResult> {
    if query.trim().is_empty() || top_k == 0 {
        return Vec::new();
    }
    let query_tokens = tokenize(query);
    let scores = bm25_index.scores(&query_tokens, selector);
    sort_top_k(&scores, top_k)
        .into_iter()
        .filter(|&idx| scores[idx] > 0.0)
        .filter_map(|idx| {
            chunks.get(idx).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                score: scores[idx],
                source: SearchMode::Bm25,
            })
        })
        .collect()
}

fn sort_top_k(arr: &[f32], top_k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&a, &b| {
        arr[b]
            .partial_cmp(&arr[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    indices.truncate(top_k.min(indices.len()));
    indices
}

pub fn search_hybrid(
    query: &str,
    model: &dyn Encoder,
    semantic_index: &SemanticIndex,
    bm25_index: &Bm25Index,
    chunks: &[Chunk],
    top_k: usize,
    alpha: Option<f32>,
    selector: Option<&[usize]>,
) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let alpha_weight = resolve_alpha(query, alpha);
    let candidate_count = top_k * 5;

    let semantic = search_semantic(
        query,
        model,
        semantic_index,
        chunks,
        candidate_count,
        selector,
    )?;
    let semantic_scores: HashMap<usize, f32> = semantic
        .into_iter()
        .filter_map(|result| {
            chunks
                .iter()
                .position(|c| c == &result.chunk)
                .map(|idx| (idx, result.score))
        })
        .collect();

    let mut bm25_scores = HashMap::new();
    for result in search_bm25(query, bm25_index, chunks, candidate_count, selector) {
        if result.score != 0.0 {
            if let Some(idx) = chunks.iter().position(|c| c == &result.chunk) {
                bm25_scores.insert(idx, result.score);
            }
        }
    }

    let normalized_semantic = rrf_scores(&semantic_scores);
    let normalized_bm25 = rrf_scores(&bm25_scores);

    // Mirrors Python: all_candidates = sorted({*sem, *bm25}, key=lambda c: c.start_line)
    let mut all_candidates: Vec<usize> = normalized_semantic
        .keys()
        .chain(normalized_bm25.keys())
        .copied()
        .collect();
    all_candidates.sort_by_key(|&idx| (chunks[idx].start_line, idx));
    all_candidates.dedup();

    let mut combined_scores = HashMap::new();
    for idx in all_candidates {
        let score = alpha_weight * normalized_semantic.get(&idx).copied().unwrap_or(0.0)
            + (1.0 - alpha_weight) * normalized_bm25.get(&idx).copied().unwrap_or(0.0);
        combined_scores.insert(idx, score);
    }

    boost_multi_chunk_files(&mut combined_scores, chunks);
    let combined_scores = apply_query_boost(combined_scores, query, chunks);
    let ranked = rerank_topk(&combined_scores, chunks, top_k, alpha_weight < 1.0);

    Ok(ranked
        .into_iter()
        .filter_map(|(idx, score)| {
            chunks.get(idx).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                score,
                source: SearchMode::Hybrid,
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::{HashEmbedder, SemanticIndex};

    fn chunk(text: &str, path: &str) -> Chunk {
        Chunk {
            content: text.to_string(),
            file_path: path.to_string(),
            start_line: 1,
            end_line: 1,
            language: Some("python".to_string()),
        }
    }

    #[test]
    fn rrf_orders_by_score() {
        let scores = HashMap::from([(1, 0.5), (2, 0.9)]);
        let rrf = rrf_scores(&scores);
        assert!(rrf[&2] > rrf[&1]);
    }

    #[test]
    fn hybrid_returns_results() {
        let model = HashEmbedder::default();
        let chunks = vec![
            chunk("def authenticate(token): return True", "auth.py"),
            chunk("def format_name(x): return x", "utils.py"),
        ];
        let semantic = SemanticIndex::from_chunks(&model, &chunks).unwrap();
        let bm25 = Bm25Index::new(&crate::sparse::bm25_docs(&chunks));
        let results = search_hybrid(
            "authenticate",
            &model,
            &semantic,
            &bm25,
            &chunks,
            2,
            None,
            None,
        )
        .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].source, SearchMode::Hybrid);
    }
}
