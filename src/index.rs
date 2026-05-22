use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::chunking::chunk_source;
use crate::dense::{Encoder, HashEmbedder, Model2VecEncoder, SemanticIndex};
use crate::file_walker::walk_files;
use crate::files::{detect_language, get_extensions};
use crate::search::{search_bm25, search_hybrid, search_semantic};
use crate::sparse::{bm25_docs, Bm25Index};
use crate::stats::save_search_stats;
use crate::types::{CallType, Chunk, IndexStats, Result, SearchMode, SearchResult, SembleError};

const MAX_FILE_BYTES: u64 = 1_000_000;

pub trait RelatedSeed {
    fn as_chunk(&self) -> &Chunk;
}

impl RelatedSeed for Chunk {
    fn as_chunk(&self) -> &Chunk {
        self
    }
}

impl RelatedSeed for SearchResult {
    fn as_chunk(&self) -> &Chunk {
        &self.chunk
    }
}

#[derive(Debug)]
pub struct SembleIndex {
    pub chunks: Vec<Chunk>,
    model: Arc<dyn Encoder>,
    bm25_index: Bm25Index,
    semantic_index: SemanticIndex,
    root: Option<PathBuf>,
    file_sizes: BTreeMap<String, usize>,
    file_mapping: HashMap<String, Vec<usize>>,
    language_mapping: HashMap<String, Vec<usize>>,
}

impl SembleIndex {
    /// Build an index using the default Python model: minishlab/potion-code-16M.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path_with_options(path, false, None)
    }

    /// Build an index with default Model2Vec/potion embeddings and file-selection options.
    pub fn from_path_with_options(
        path: impl AsRef<Path>,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self> {
        let model: Arc<dyn Encoder> = Arc::new(Model2VecEncoder::load(None)?);
        Self::from_path_with_arc_encoder(path, model, include_text_files, extra_extensions)
    }

    /// Build an index using a caller-supplied model path or Hugging Face repository.
    pub fn from_path_with_model(
        path: impl AsRef<Path>,
        model_path: Option<&str>,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self> {
        let model: Arc<dyn Encoder> = Arc::new(Model2VecEncoder::load(model_path)?);
        Self::from_path_with_arc_encoder(path, model, include_text_files, extra_extensions)
    }

    /// Build an index with a custom encoder. This is primarily for tests/offline use.
    pub fn from_path_with_encoder<E>(
        path: impl AsRef<Path>,
        encoder: E,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self>
    where
        E: Encoder + 'static,
    {
        Self::from_path_with_arc_encoder(
            path,
            Arc::new(encoder),
            include_text_files,
            extra_extensions,
        )
    }

    pub fn from_path_with_arc_encoder(
        path: impl AsRef<Path>,
        model: Arc<dyn Encoder>,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(SembleError::PathDoesNotExist(path.display().to_string()));
        }
        if !path.is_dir() {
            return Err(SembleError::NotADirectory(path.display().to_string()));
        }
        let root = path.canonicalize()?;
        let (bm25_index, semantic_index, chunks) = create_index_from_path(
            &root,
            model.as_ref(),
            include_text_files,
            extra_extensions,
            &root,
        )?;
        Ok(Self::new(
            model,
            bm25_index,
            semantic_index,
            chunks,
            Some(root),
        ))
    }

    pub fn from_git(url: &str, git_ref: Option<&str>, include_text_files: bool) -> Result<Self> {
        Self::from_git_with_model(url, git_ref, None, include_text_files, None)
    }

    /// Clone a git repository and index it using a caller-supplied model path.
    pub fn from_git_with_model(
        url: &str,
        git_ref: Option<&str>,
        model_path: Option<&str>,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self> {
        let model: Arc<dyn Encoder> = Arc::new(Model2VecEncoder::load(model_path)?);
        Self::from_git_with_arc_encoder(url, git_ref, model, include_text_files, extra_extensions)
    }

    /// Clone a git repository and index it using a custom encoder.
    pub fn from_git_with_encoder<E>(
        url: &str,
        git_ref: Option<&str>,
        encoder: E,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self>
    where
        E: Encoder + 'static,
    {
        Self::from_git_with_arc_encoder(
            url,
            git_ref,
            Arc::new(encoder),
            include_text_files,
            extra_extensions,
        )
    }

    fn from_git_with_arc_encoder(
        url: &str,
        git_ref: Option<&str>,
        model: Arc<dyn Encoder>,
        include_text_files: bool,
        extra_extensions: Option<&[String]>,
    ) -> Result<Self> {
        let tmp = std::env::temp_dir().join(format!(
            "semble-rs-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&tmp)?;
        let mut cmd = Command::new("git");
        cmd.arg("clone").arg("--depth").arg("1");
        if let Some(git_ref) = git_ref {
            cmd.arg("--branch").arg(git_ref);
        }
        cmd.arg("--").arg(url).arg(&tmp);
        let output = cmd
            .output()
            .map_err(|e| SembleError::Git(format!("failed to run git clone: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = fs::remove_dir_all(&tmp);
            return Err(SembleError::Git(format!(
                "git clone failed: {}",
                stderr.trim()
            )));
        }
        let indexed =
            Self::from_path_with_arc_encoder(&tmp, model, include_text_files, extra_extensions);
        let _ = fs::remove_dir_all(&tmp);
        indexed
    }

    pub fn new(
        model: Arc<dyn Encoder>,
        bm25_index: Bm25Index,
        semantic_index: SemanticIndex,
        chunks: Vec<Chunk>,
        root: Option<PathBuf>,
    ) -> Self {
        let file_sizes = root
            .as_ref()
            .map(|root| compute_file_sizes(root, &chunks))
            .unwrap_or_default();
        let (file_mapping, language_mapping) = populate_mapping(&chunks);
        Self {
            model,
            chunks,
            bm25_index,
            semantic_index,
            root,
            file_sizes,
            file_mapping,
            language_mapping,
        }
    }

    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn stats(&self) -> IndexStats {
        let indexed_files = self.file_mapping.len();
        let total_chunks = self.chunks.len();
        let languages = self
            .language_mapping
            .iter()
            .map(|(lang, ids)| (lang.clone(), ids.len()))
            .collect::<BTreeMap<_, _>>();
        IndexStats {
            indexed_files,
            total_chunks,
            languages,
        }
    }

    pub fn search(&self, query: &str, top_k: usize, mode: SearchMode) -> Result<Vec<SearchResult>> {
        self.search_with_options(query, top_k, mode, None, None, None)
    }

    pub fn search_with_options(
        &self,
        query: &str,
        top_k: usize,
        mode: SearchMode,
        alpha: Option<f32>,
        filter_languages: Option<&[String]>,
        filter_paths: Option<&[String]>,
    ) -> Result<Vec<SearchResult>> {
        if self.chunks.is_empty() || query.trim().is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let selector = self.selector_vector(filter_languages, filter_paths);
        let selector_ref = selector.as_deref();
        let results = match mode {
            SearchMode::Bm25 => {
                search_bm25(query, &self.bm25_index, &self.chunks, top_k, selector_ref)
            }
            SearchMode::Semantic => search_semantic(
                query,
                self.model.as_ref(),
                &self.semantic_index,
                &self.chunks,
                top_k,
                selector_ref,
            )?,
            SearchMode::Hybrid => search_hybrid(
                query,
                self.model.as_ref(),
                &self.semantic_index,
                &self.bm25_index,
                &self.chunks,
                top_k,
                alpha,
                selector_ref,
            )?,
        };
        save_search_stats(&results, CallType::Search, &self.file_sizes);
        Ok(results)
    }

    pub fn find_related_chunk(&self, source: &Chunk, top_k: usize) -> Vec<SearchResult> {
        if top_k == 0 {
            return Vec::new();
        }
        let filter_languages = source.language.clone().map(|lang| vec![lang]);
        let selector = self.selector_vector(filter_languages.as_deref(), None);
        let selector_ref = selector.as_deref();
        let mut results = search_semantic(
            &source.content,
            self.model.as_ref(),
            &self.semantic_index,
            &self.chunks,
            top_k + 1,
            selector_ref,
        )
        .unwrap_or_default();
        results.retain(|r| &r.chunk != source);
        results.truncate(top_k);
        save_search_stats(&results, CallType::FindRelated, &self.file_sizes);
        results
    }

    pub fn find_related<S>(&self, source: &S, top_k: usize) -> Vec<SearchResult>
    where
        S: RelatedSeed + ?Sized,
    {
        self.find_related_chunk(source.as_chunk(), top_k)
    }

    pub fn find_related_result(&self, source: &SearchResult, top_k: usize) -> Vec<SearchResult> {
        self.find_related(source, top_k)
    }

    fn selector_vector(
        &self,
        filter_languages: Option<&[String]>,
        filter_paths: Option<&[String]>,
    ) -> Option<Vec<usize>> {
        let mut selector = Vec::new();
        if let Some(languages) = filter_languages {
            for language in languages {
                if let Some(ids) = self.language_mapping.get(language) {
                    selector.extend(ids.iter().copied());
                }
            }
        }
        if let Some(paths) = filter_paths {
            for path in paths {
                if let Some(ids) = self.file_mapping.get(path) {
                    selector.extend(ids.iter().copied());
                }
            }
        }
        if selector.is_empty() {
            None
        } else {
            selector.sort_unstable();
            selector.dedup();
            Some(selector)
        }
    }
}

pub fn create_index_from_path(
    root: &Path,
    model: &dyn Encoder,
    include_text_files: bool,
    extra_extensions: Option<&[String]>,
    display_root: &Path,
) -> Result<(Bm25Index, SemanticIndex, Vec<Chunk>)> {
    let extensions = get_extensions(include_text_files, extra_extensions);
    let files = walk_files(root, &extensions, None)?;
    let mut chunks = Vec::new();

    for file in files {
        let meta = match fs::metadata(&file) {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let Ok(source) = fs::read_to_string(&file) else {
            continue;
        };
        let rel = file
            .strip_prefix(display_root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        let language = detect_language(&file);
        chunks.extend(chunk_source(&source, &rel, language));
    }

    if chunks.is_empty() {
        return Err(SembleError::NoSupportedFiles(root.display().to_string()));
    }
    let semantic_index = SemanticIndex::from_chunks(model, &chunks)?;
    let bm25_index = Bm25Index::new(&bm25_docs(&chunks));
    Ok((bm25_index, semantic_index, chunks))
}

fn populate_mapping(
    chunks: &[Chunk],
) -> (HashMap<String, Vec<usize>>, HashMap<String, Vec<usize>>) {
    let mut file_mapping: HashMap<String, Vec<usize>> = HashMap::new();
    let mut language_mapping: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        file_mapping
            .entry(chunk.file_path.clone())
            .or_default()
            .push(idx);
        if let Some(language) = &chunk.language {
            language_mapping
                .entry(language.clone())
                .or_default()
                .push(idx);
        }
    }
    (file_mapping, language_mapping)
}

fn compute_file_sizes(root: &Path, chunks: &[Chunk]) -> BTreeMap<String, usize> {
    let mut sizes = BTreeMap::new();
    for chunk in chunks {
        if sizes.contains_key(&chunk.file_path) {
            continue;
        }
        if let Ok(text) = fs::read_to_string(root.join(&chunk.file_path)) {
            sizes.insert(chunk.file_path.clone(), text.len());
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_small_project_with_custom_encoder() {
        let root = std::env::temp_dir().join(format!(
            "semble-rs-index-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("auth.py"),
            "def authenticate(token):\n    return token == 'secret'\n",
        )
        .unwrap();
        let index =
            SembleIndex::from_path_with_encoder(&root, HashEmbedder::default(), false, None)
                .unwrap();
        assert_eq!(index.stats().indexed_files, 1);
        let results = index
            .search("authenticate token", 3, SearchMode::Hybrid)
            .unwrap();
        assert!(!results.is_empty());
        let _ = fs::remove_dir_all(root);
    }
}
