# semble-rs

Strict Rust port of the core [`semble`](https://github.com/MinishLab/semble) code-search workflow.

This repository includes the Rust library, the `semble` CLI, and lightweight
verification helpers that keep docs and metadata free of machine-local paths
and email addresses.

This port is designed to preserve the original Python retrieval pipeline rather than approximate it:

- `SembleIndex::from_path` / `SembleIndex::from_git`
- `search` with `hybrid`, `semantic`, and `bm25` modes
- `find-related` lookup for code near a file/line seed
- file walking and extension/language filtering
- tree-sitter-aware chunking via `tree-sitter-language-pack`
- Model2Vec static embeddings via `model2vec-rs`, defaulting to `minishlab/potion-code-16M`
- BM25S-compatible Lucene BM25 scoring (`k1=1.5`, `b=0.75`)
- RRF fusion, adaptive alpha, definition boosts, identifier-stem boosts, file-coherence boosts, path penalties, and file-saturation reranking
- token-savings stats and `semble init`

## Important parity note

The default constructor loads `minishlab/potion-code-16M` from Hugging Face through `model2vec-rs`:

```rust
let index = SembleIndex::from_path("./my-project")?;
```

For offline tests or applications that do not want model downloads, use a custom encoder:

```rust
use semble_rs::dense::HashEmbedder;
use semble_rs::SembleIndex;

let index = SembleIndex::from_path_with_encoder(
    "./my-project",
    HashEmbedder::default(),
    false,
    None,
)?;
```

`HashEmbedder` is provided only as a test/offline utility. It is not used by the default path and is not expected to reproduce the benchmark quality of the Python package.

## Build

```bash
cargo build --release
cargo test --release
python3 scripts/verify_repo_hygiene.py
```

The first default build resolves `model2vec-rs`, `tree-sitter-language-pack`, and their transitive dependencies from crates.io. The first default run may also download/cache `minishlab/potion-code-16M`.

## CLI

```bash
# Search a local repo
cargo run -- search "authentication flow" ./my-project

# Search with a specific mode and result count
cargo run -- search "save_pretrained" ./my-project --top-k 10 --mode hybrid

# Find related chunks around a known location
cargo run -- find-related src/auth.py 42 ./my-project

# Write .claude/agents/semble-search.md
cargo run -- init

# Show token-savings stats
cargo run -- savings --verbose
```

Install the binary with:

```bash
cargo install --path .
```

Then use `semble ...` directly.

## Python parity benchmark

The cross-implementation benchmark requires an explicit Python reference
checkout instead of assuming a machine-local path:

```bash
python3 scripts/benchmark_parity.py --python-repo ../semble
```

You can also set `SEMBLE_PYTHON_REPO` in the environment and omit the flag.

## Library example

```rust
use semble_rs::{SearchMode, SembleIndex};

fn main() -> semble_rs::types::Result<()> {
    let index = SembleIndex::from_path("./my-project")?;
    let results = index.search("save model to disk", 3, SearchMode::Hybrid)?;

    for result in results {
        println!("{} score={:.3}", result.chunk.location(), result.score);
    }
    Ok(())
}
```

## License

This repository is licensed under ISC. See [LICENSE](LICENSE).

The preserved upstream MIT text is kept in
[THIRD_PARTY_LICENSE](THIRD_PARTY_LICENSE).
