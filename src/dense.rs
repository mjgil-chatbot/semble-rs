use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use model2vec_rs::model::StaticModel;

use crate::tokens::tokenize;
use crate::types::{Chunk, Result, SembleError};

/// Default model used by the Python implementation.
pub const DEFAULT_MODEL_NAME: &str = "minishlab/potion-code-16M";

/// Minimal encoder protocol matching the Python `Encoder` Protocol.
pub trait Encoder: Send + Sync {
    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    fn encode_one(&self, text: &str) -> Result<Vec<f32>> {
        let out = self.encode(&[text.to_string()])?;
        Ok(out.into_iter().next().unwrap_or_default())
    }
}

impl fmt::Debug for dyn Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("dyn Encoder")
    }
}

/// Rust-native Model2Vec encoder.
///
/// Uses `model2vec-rs::model::StaticModel::from_pretrained` so the default
/// path mirrors Python's `StaticModel.from_pretrained("minishlab/potion-code-16M")`.
pub struct Model2VecEncoder {
    model: StaticModel,
}

impl fmt::Debug for Model2VecEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Model2VecEncoder")
    }
}

impl Model2VecEncoder {
    pub fn load(model_path: Option<&str>) -> Result<Self> {
        let model_path = model_path.unwrap_or(DEFAULT_MODEL_NAME);
        let model =
            StaticModel::from_pretrained(model_path, None::<&str>, None::<bool>, None::<&str>)
                .map_err(|e| SembleError::Model(e.to_string()))?;
        Ok(Self { model })
    }
}

impl Encoder for Model2VecEncoder {
    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(self.model.encode(texts))
    }
}

/// Test-only / offline encoder used by unit tests and examples that should not
/// download a Hugging Face model. It is not used by the default SembleIndex path.
#[derive(Clone, Debug)]
pub struct HashEmbedder {
    dim: usize,
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self { dim: 256 }
    }
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }
}

impl Encoder for HashEmbedder {
    fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.encode_hash(t)).collect())
    }
}

impl HashEmbedder {
    fn encode_hash(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0; self.dim];
        for token in tokenize(text) {
            let h = hash64(&token);
            let idx = (h as usize) % self.dim;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }
        normalize(&mut vec);
        vec
    }
}

fn hash64<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec {
            *v /= norm;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SemanticIndex {
    vectors: Vec<Vec<f32>>,
}

impl SemanticIndex {
    pub fn new(mut vectors: Vec<Vec<f32>>) -> Self {
        for vector in &mut vectors {
            normalize(vector);
        }
        Self { vectors }
    }

    pub fn from_chunks(model: &dyn Encoder, chunks: &[Chunk]) -> Result<Self> {
        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        Ok(Self::new(model.encode(&texts)?))
    }

    /// Batched cosine-distance query compatible with Python's SelectableBasicBackend.
    /// Returns `(chunk_index, cosine_distance)` sorted by smallest distance.
    pub fn query(&self, vector: &[f32], k: usize, selector: Option<&[usize]>) -> Vec<(usize, f32)> {
        if k == 0 || self.vectors.is_empty() {
            return Vec::new();
        }
        let mut query = vector.to_vec();
        normalize(&mut query);

        let candidates: Box<dyn Iterator<Item = usize> + '_> = if let Some(selector) = selector {
            Box::new(selector.iter().copied())
        } else {
            Box::new(0..self.vectors.len())
        };

        let mut scored: Vec<(usize, f32)> = candidates
            .filter_map(|idx| {
                self.vectors
                    .get(idx)
                    .map(|doc| (idx, 1.0 - dot(&query, doc)))
            })
            .collect();
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k.min(scored.len()));
        scored
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embedder_produces_unit_vectors() {
        let text = vec!["hello_world helloWorld".to_string()];
        let emb = HashEmbedder::default().encode(&text).unwrap().remove(0);
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn semantic_query_returns_cosine_distance() {
        let idx = SemanticIndex::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let hits = idx.query(&[1.0, 0.0], 1, None);
        assert_eq!(hits[0].0, 0);
        assert!(hits[0].1 < 1e-6);
    }
}
