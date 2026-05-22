/// Split a single identifier into lowercase sub-tokens via snake_case and
/// camelCase/PascalCase boundaries. The original lowercase token is preserved
/// when it decomposes into multiple parts.
pub fn split_identifier(token: &str) -> Vec<String> {
    let lower = token.to_ascii_lowercase();
    let parts: Vec<String> = if token.contains('_') {
        lower
            .split('_')
            .filter(|p| !p.is_empty())
            .map(|p| p.to_string())
            .collect()
    } else {
        split_camel(token)
            .into_iter()
            .map(|p| p.to_ascii_lowercase())
            .filter(|p| !p.is_empty())
            .collect()
    };

    if parts.len() >= 2 {
        let mut out = Vec::with_capacity(parts.len() + 1);
        out.push(lower);
        out.extend(parts);
        out
    } else {
        vec![lower]
    }
}

fn split_camel(token: &str) -> Vec<String> {
    let chars: Vec<(usize, char)> = token.char_indices().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut starts = vec![0usize];
    for i in 1..chars.len() {
        let prev = chars[i - 1].1;
        let cur = chars[i].1;
        let next = chars.get(i + 1).map(|(_, c)| *c);

        let boundary = (cur.is_ascii_uppercase()
            && (prev.is_ascii_lowercase() || prev.is_ascii_digit()))
            || (cur.is_ascii_uppercase()
                && prev.is_ascii_uppercase()
                && next.is_some_and(|n| n.is_ascii_lowercase()))
            || (cur.is_ascii_digit() && !prev.is_ascii_digit())
            || (!cur.is_ascii_digit() && prev.is_ascii_digit());
        if boundary {
            starts.push(chars[i].0);
        }
    }

    let mut parts = Vec::new();
    for (i, start) in starts.iter().copied().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(token.len());
        let part = &token[start..end];
        if !part.is_empty() {
            parts.push(part.to_string());
        }
    }
    parts
}

/// Split text into lowercase identifier-like tokens for BM25 indexing.
/// Compound identifiers are expanded into sub-tokens for partial matches.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        let is_ident = ch.is_ascii_alphanumeric() || ch == '_';
        if let Some(s) = start {
            if !is_ident {
                let raw = &text[s..idx];
                if is_identifier_start(raw.chars().next()) {
                    out.extend(split_identifier(raw));
                }
                start = None;
            }
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            start = Some(idx);
        }
    }

    if let Some(s) = start {
        let raw = &text[s..];
        if is_identifier_start(raw.chars().next()) {
            out.extend(split_identifier(raw));
        }
    }

    out
}

fn is_identifier_start(ch: Option<char>) -> bool {
    ch.is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_identifiers() {
        assert_eq!(
            split_identifier("HandlerStack"),
            vec!["handlerstack", "handler", "stack"]
        );
        assert_eq!(
            split_identifier("getHTTPResponse"),
            vec!["gethttpresponse", "get", "http", "response"]
        );
        assert_eq!(split_identifier("my_func"), vec!["my_func", "my", "func"]);
        assert_eq!(split_identifier("simple"), vec!["simple"]);
    }

    #[test]
    fn tokenizes_text() {
        assert_eq!(
            tokenize("def getHTTPResponse(): pass")[..4],
            ["def", "gethttpresponse", "get", "http"]
        );
    }
}
