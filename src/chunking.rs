use crate::files::all_languages;
use crate::types::Chunk;

const DESIRED_CHUNK_LENGTH_CHARS: usize = 1500;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkBoundary {
    start: usize, // character offset, inclusive
    end: usize,   // character offset, exclusive
}

/// Check whether a language is in Semble's known language map.
pub fn is_supported_language(language: &str) -> bool {
    all_languages().contains(language)
}

/// Chunk pre-read source text.
///
/// Mirrors Python's `chunk_source`: try tree-sitter for known languages, fall
/// back to line chunking when the parser cannot be loaded, and convert byte
/// ranges from tree-sitter into Python-style character offsets before slicing.
pub fn chunk_source(source: &str, file_path: &str, language: Option<&str>) -> Vec<Chunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }

    let boundaries = language
        .filter(|lang| is_supported_language(lang))
        .and_then(|lang| chunk(source, lang, DESIRED_CHUNK_LENGTH_CHARS))
        .unwrap_or_else(|| chunk_lines(source, DESIRED_CHUNK_LENGTH_CHARS));

    boundaries
        .into_iter()
        .filter_map(|b| {
            let end_index = b.end.saturating_sub(1).max(b.start);
            let text = char_slice_inclusive(source, b.start, end_index)?;
            Some(Chunk {
                content: text,
                file_path: file_path.to_string(),
                start_line: count_newlines_before_char(source, b.start) + 1,
                end_line: count_newlines_before_char(source, end_index) + 1,
                language: language.map(ToOwned::to_owned),
            })
        })
        .collect()
}

fn chunk(text: &str, language: &str, desired_length: usize) -> Option<Vec<ChunkBoundary>> {
    if text.trim().is_empty() {
        return Some(Vec::new());
    }

    let mut parser = match tree_sitter_language_pack::get_parser(language) {
        Ok(parser) => parser,
        Err(_) => return None,
    };
    let tree = parser.parse(text)?;
    let root = tree.root_node();
    let bytes = text.as_bytes();

    let mut chunks = Vec::new();
    for boundary in merge_node(root, desired_length) {
        chunks.push(ChunkBoundary {
            start: byte_to_char_offset(bytes, boundary.start)?,
            end: byte_to_char_offset(bytes, boundary.end)?,
        });
    }
    Some(chunks)
}

fn merge_adjacent_chunks(chunks: &[ChunkBoundary], desired_length: usize) -> Vec<ChunkBoundary> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut current_start = chunks[0].start;
    let mut current_end = chunks[0].end;
    let mut current_length = current_end.saturating_sub(current_start);

    for group in &chunks[1..] {
        let length = group.end.saturating_sub(group.start);
        if current_length + length > desired_length {
            merged.push(ChunkBoundary {
                start: current_start,
                end: current_end,
            });
            current_start = group.start;
            current_end = group.end;
            current_length = length;
            continue;
        }
        current_end = group.end;
        current_length += length;
    }

    merged.push(ChunkBoundary {
        start: current_start,
        end: current_end,
    });
    merged
}

fn merge_node_inner(
    node: tree_sitter_language_pack::Node,
    desired_length: usize,
) -> Vec<ChunkBoundary> {
    let child_count = node.child_count();
    if child_count == 0 {
        return vec![ChunkBoundary {
            start: node.start_byte(),
            end: node.end_byte(),
        }];
    }

    let mut groups = Vec::new();
    let mut index = 0usize;
    while index < child_count {
        let Some(mut child) = node.child(index as u32) else {
            break;
        };
        let start = child.start_byte();
        let mut end = child.end_byte();
        let mut length = child.end_byte().saturating_sub(child.start_byte());
        index += 1;

        if length > desired_length {
            groups.extend(merge_node_inner(child, desired_length));
            continue;
        }

        while index < child_count {
            let Some(next_child) = node.child(index as u32) else {
                break;
            };
            let child_length = next_child
                .end_byte()
                .saturating_sub(next_child.start_byte());
            if length + child_length > desired_length {
                break;
            }
            child = next_child;
            end = child.end_byte();
            length += child_length;
            index += 1;
        }

        groups.push(ChunkBoundary { start, end });
    }
    groups
}

fn merge_node(node: tree_sitter_language_pack::Node, desired_length: usize) -> Vec<ChunkBoundary> {
    let raw_chunks = merge_node_inner(node, desired_length);
    merge_adjacent_chunks(&raw_chunks, desired_length)
}

fn chunk_lines(text: &str, desired_length: usize) -> Vec<ChunkBoundary> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut lines_as_groups = Vec::new();
    let mut index = 0usize;
    for line in text.split_inclusive('\n') {
        let len = line.chars().count();
        lines_as_groups.push(ChunkBoundary {
            start: index,
            end: index + len,
        });
        index += len;
    }
    if index < text.chars().count() {
        let total = text.chars().count();
        lines_as_groups.push(ChunkBoundary {
            start: index,
            end: total,
        });
    }
    merge_adjacent_chunks(&lines_as_groups, desired_length)
}

fn byte_to_char_offset(bytes: &[u8], byte_offset: usize) -> Option<usize> {
    let prefix = bytes.get(..byte_offset)?;
    std::str::from_utf8(prefix).ok().map(|s| s.chars().count())
}

fn char_slice_inclusive(source: &str, start: usize, end: usize) -> Option<String> {
    if end < start {
        return Some(String::new());
    }
    let mut out = String::new();
    for (idx, ch) in source.chars().enumerate() {
        if idx < start {
            continue;
        }
        if idx > end {
            break;
        }
        out.push(ch);
    }
    Some(out)
}

fn count_newlines_before_char(source: &str, char_end_exclusive: usize) -> usize {
    source
        .chars()
        .take(char_end_exclusive)
        .filter(|&ch| ch == '\n')
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_source_with_lines() {
        let chunks = chunk_source("def a():\n    pass\n", "a.py", Some("not_a_real_language"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 2);
        assert_eq!(chunks[0].location(), "a.py:1-2");
    }

    #[test]
    fn line_chunking_uses_character_offsets() {
        let chunks = chunk_source("αβ\nγδ\n", "u.txt", None);
        assert_eq!(chunks[0].content, "αβ\nγδ\n");
        assert_eq!(chunks[0].end_line, 2);
    }
}
