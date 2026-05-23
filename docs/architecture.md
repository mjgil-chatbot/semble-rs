# Architecture

`semble-rs` is still implemented as a single Cargo package, but the code is organized around three logical crate boundaries so parity work stays isolated:

## Logical crates

`semble-core`
Contains retrieval behavior and the public library API.
This includes `src/chunking.rs`, `src/dense.rs`, `src/file_walker.rs`, `src/files.rs`, `src/index.rs`, `src/ranking.rs`, `src/search.rs`, `src/sparse.rs`, `src/tokens.rs`, `src/types.rs`, and `src/utils.rs`.
Anything that changes ranking, indexing, chunking, embeddings, filters, or public library types belongs here.

`semble-cli`
Contains the user-facing command surface in `src/main.rs` plus the bundled agent template in `src/agent.rs`.
CLI-specific parity work belongs here: subcommand parsing, exact stdout/stderr messages, and command-level behavior such as `init` and `savings`.

`semble-verification`
Contains parity and regression checks.
Today that lives in `tests/` and `scripts/benchmark_parity.py`.
Behavioral assertions that should gate `cargo test` belong in `tests/`.
Cross-implementation checks that need the Python reference checkout stay in `scripts/`.
The parity benchmark requires `--python-repo` or `SEMBLE_PYTHON_REPO` instead
of assuming a machine-local checkout path.

## Current parity strategy

The Rust crate uses the Python `semble` project as the reference implementation for the core CLI workflow:

- `search`
- `find-related`
- `init`
- `savings`
- public library search and related-search entrypoints

Parity-sensitive behavior is enforced in two layers:

- Rust tests cover API-level behavior that should remain stable without external dependencies.
- `scripts/benchmark_parity.py` runs the built Rust CLI and the Python `semble` CLI against the same synthetic fixture, normalizes score text, and checks both output shape and repeated command latency.

## Benchmark scope

The benchmark script is intentionally end-to-end and CLI-focused.
It measures the user-visible commands rather than internal helper functions so regressions in parsing, indexing, ranking, or formatting are caught together.
The current cases are:

- `search --mode bm25`
- `search --mode hybrid`
- `find-related`

The script fails when normalized command output diverges or when Rust median latency regresses beyond the configured ratio threshold versus the Python reference.

## Generated-artifact exclusions

Repo-root indexing now ignores the same generated artifact classes that caused
the `servolink` failure investigation to wander into Cargo depfiles and other
non-source trees. The default walker excludes:

- `target/`
- `target-*/`
- `*-target/`
- `.lightweight-test/`

Those defaults are implemented in `src/file_walker.rs` so CLI searches stay
focused on repo-owned source inputs instead of build outputs.
