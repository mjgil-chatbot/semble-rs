use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

/// Search strategy for [`SembleIndex::search`](crate::SembleIndex::search).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SearchMode {
    Hybrid,
    Semantic,
    Bm25,
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchMode::Hybrid => "hybrid",
            SearchMode::Semantic => "semantic",
            SearchMode::Bm25 => "bm25",
        }
    }
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "hybrid" => Ok(SearchMode::Hybrid),
            "semantic" => Ok(SearchMode::Semantic),
            "bm25" => Ok(SearchMode::Bm25),
            other => Err(format!("unknown search mode: {other}")),
        }
    }
}

/// Call type for token-savings tracking.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CallType {
    Search,
    FindRelated,
}

impl CallType {
    pub fn as_str(self) -> &'static str {
        match self {
            CallType::Search => "search",
            CallType::FindRelated => "find_related",
        }
    }
}

/// A single indexable unit of code.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Chunk {
    pub content: String,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: Option<String>,
}

impl Chunk {
    pub fn location(&self) -> String {
        format!("{}:{}-{}", self.file_path, self.start_line, self.end_line)
    }
}

/// A single search result with score and source mode.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub source: SearchMode,
}

/// Statistics about the current index state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IndexStats {
    pub indexed_files: usize,
    pub total_chunks: usize,
    pub languages: BTreeMap<String, usize>,
}

/// Errors returned by the Rust port.
#[derive(Debug)]
pub enum SembleError {
    Io(std::io::Error),
    Model(String),
    PathDoesNotExist(String),
    NotADirectory(String),
    NoSupportedFiles(String),
    Git(String),
    InvalidMode(String),
}

impl fmt::Display for SembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SembleError::Io(e) => write!(f, "{e}"),
            SembleError::Model(e) => write!(f, "model error: {e}"),
            SembleError::PathDoesNotExist(p) => write!(f, "Path does not exist: {p}"),
            SembleError::NotADirectory(p) => write!(f, "Path is not a directory: {p}"),
            SembleError::NoSupportedFiles(p) => write!(f, "No supported files found under {p}."),
            SembleError::Git(msg) => write!(f, "{msg}"),
            SembleError::InvalidMode(mode) => write!(f, "Unknown search mode: {mode}"),
        }
    }
}

impl std::error::Error for SembleError {}

impl From<std::io::Error> for SembleError {
    fn from(value: std::io::Error) -> Self {
        SembleError::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, SembleError>;
