use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use semble_rs::dense::HashEmbedder;
use semble_rs::utils::resolve_chunk;
use semble_rs::{SearchMode, SembleIndex};

#[test]
fn end_to_end_search_and_find_related() {
    let root = std::env::temp_dir().join(format!(
        "semble-rs-e2e-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("auth.py"),
        "def authenticate(token):\n    return token == 'secret'\n\ndef login(username, password):\n    return authenticate(password)\n",
    )
    .unwrap();
    fs::write(
        root.join("utils.py"),
        "def format_name(first, last):\n    return f'{first} {last}'\n",
    )
    .unwrap();

    let index =
        SembleIndex::from_path_with_encoder(&root, HashEmbedder::default(), false, None).unwrap();
    let results = index
        .search("authenticate token", 3, SearchMode::Hybrid)
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.chunk.file_path == "auth.py"));

    let chunk = resolve_chunk(&index.chunks, "auth.py", 1).unwrap();
    let related = index.find_related_chunk(chunk, 3);
    assert!(related.len() <= 3);

    let _ = fs::remove_dir_all(root);
}
