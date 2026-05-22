//! Rust port of the core `semble` code-search library.
//!
//! This version preserves the Python retrieval pipeline: tree-sitter-aware
//! chunking, Model2Vec/potion-code embeddings, BM25S-compatible Lucene BM25,
//! RRF fusion, and the same code-aware ranking boosts and penalties.

pub mod agent;
pub mod chunking;
pub mod dense;
pub mod file_walker;
pub mod files;
pub mod index;
pub mod ranking;
pub mod search;
pub mod sparse;
pub mod stats;
pub mod tokens;
pub mod types;
pub mod utils;

pub use dense::{Encoder, Model2VecEncoder, DEFAULT_MODEL_NAME};
pub use index::SembleIndex;
pub use types::{CallType, Chunk, IndexStats, SearchMode, SearchResult};
