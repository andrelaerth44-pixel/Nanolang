use std::{env, fs, process};

fn scan_braces(line: &str) -> (usize, usize) {
    let mut opens = 0usize;
    let mut closes = 0usize;
    let mut chars = line.chars().peekable();
    let mut string = false;
    let mut escape = false;
    while let Some(c) = chars.next() {
        if string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                string = false;
            }
            continue;
        }
        if c == '"' {
            string = true;
            continue;
        }
        if c == '#' {
            break;
        }
        match c {
            '{' => opens += 1,
            '}' => closes += 1,
            _ => {}
        }
    }
    (opens, closes)
}


fn normalize_inline(line: &str) -> String {
    let mut tokens = Vec::<String>::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c == '#' {
            let comment: String = chars[i..].iter().collect();
            if !tokens.is_empty() {
                return format!("{} {}", join_tokens(&tokens), comment.trim());
            }
            return comment.trim().to_string();
        }

        if c == '"' {
            let start = i;
            i += 1;
            let mut escape = false;
            while i < chars.len() {
                let ch = chars[i];
                i += 1;
                if escape {
                    escape = false;
                } else if ch == '\\' {
                    escape = true;
                } else if ch == '"' {
                    break;
                }
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        if c.is_ascii_alphanumeric() || c == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        if i + 1 < chars.len() {
            let two = [c, chars[i + 1]];
            if matches!(
                two,
                ['=', '='] | ['!', '='] | ['>', '='] | ['<', '='] | ['&', '&'] | ['|', '|']
            ) {
                tokens.push(two.iter().collect());
                i += 2;
                continue;
            }
        }

        tokens.push(c.to_string());
        i += 1;
    }

    join_tokens(&tokens)
}

fn token_is_value(token: &str) -> bool {
    matches!(token.as_bytes().first(),
        Some(b'0'..=b'9') | Some(b'_')
    ) || token.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        || token.starts_with('"')
        || matches!(token, ")" | "]" | "}")
}

fn is_operator(token: &str) -> bool {
    matches!(token, "=" | "==" | "!=" | ">" | ">=" | "<" | "<=" | "&&" | "||" | "+" | "-" | "*" | "/" | "%")
}

fn join_tokens(tokens: &[String]) -> String {
    let mut out = String::new();

    for (index, token) in tokens.iter().enumerate() {
        let prev = tokens.get(index.wrapping_sub(1)).map(String::as_str);
        let next = tokens.get(index + 1).map(String::as_str);

        let mut need_space = false;

        if let Some(prev) = prev {
            if token == "," {
                need_space = false;
            } else if prev == "," || prev == ":" {
                need_space = true;
            } else if token == ":" {
                need_space = false;
            } else if token == "." || prev == "." {
                need_space = false;
            } else if token == ")" || token == "]" || token == "}" {
                need_space = false;
            } else if prev == "(" || prev == "[" {
                need_space = false;
            } else if token == "(" {
                need_space = false;
            } else if is_operator(token) {
                let unary = matches!(token.as_str(), "+" | "-")
                    && !token_is_value(prev)
                    && prev != ")" && prev != "]" && prev != "}";
                need_space = !unary;
            } else if is_operator(prev) {
                let unary_prev = matches!(prev, "+" | "-")
                    && !token_is_value(tokens.get(index.saturating_sub(2)).map(String::as_str).unwrap_or(""));
                need_space = !unary_prev;
            } else {
                need_space = token_is_value(prev) && (token_is_value(token) || token == "(");
                if token == "{" && prev != "=" {
                    need_space = true;
                }
            }
        }

        if need_space && !out.ends_with(' ') {
            out.push(' ');
        }
        out.push_str(token);

        if token == "," || token == ":" {
            out.push(' ');
        }
        let _ = next;
    }

    out.trim().to_string()
}

fn format_source(src: &str) -> String {
    let mut out = String::new();
    let mut indent = 0usize;
    let mut last_blank = false;

    for raw in src.lines() {
        let line = raw.trim();

        if line.is_empty() {
            if !last_blank && !out.is_empty() {
                out.push('\n');
            }
            last_blank = true;
            continue;
        }
        last_blank = false;

        let starts_with_close = line.starts_with('}');
        if starts_with_close {
            indent = indent.saturating_sub(1);
        }

        for _ in 0..indent {
            out.push_str("    ");
        }
        out.push_str(&normalize_inline(line));
        out.push('\n');

        let (opens, closes) = scan_braces(line);
        let non_leading_closes = closes.saturating_sub(if starts_with_close { 1 } else { 0 });
        indent = indent
            .saturating_add(opens)
            .saturating_sub(non_leading_closes);
    }

    if out.is_empty() {
        String::new()
    } else {
        out.trim_end_matches([' ', '\t', '\n']).to_string() + "\n"
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("uso: nano-fmt <arquivo.nano>");
        process::exit(2);
    }
    let path = &args[1];
    let src = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Nano fmt: {e}");
        process::exit(1)
    });
    if let Err(e) = fs::write(path, format_source(&src)) {
        eprintln!("Nano fmt: {e}");
        process::exit(1)
    }
}

#[cfg(test)]
mod tests {
    use super::{format_source, normalize_inline};

    #[test]
    fn normalizes_inline_spacing() {
        assert_eq!(
            normalize_inline("x=1+2*3"),
            "x = 1 + 2 * 3"
        );
        assert_eq!(
            normalize_inline("if x>=10&&x<20 {"),
            "if x >= 10 && x < 20 {"
        );
        assert_eq!(
            normalize_inline("f(a,b,{x:1,y:2})"),
            "f(a, b, {x: 1, y: 2})"
        );
        assert_eq!(
            normalize_inline("x=-4"),
            "x = -4"
        );
    }

    #[test]
    fn formats_blocks() {
        assert_eq!(
            format_source("if true {\nprint 1\n}\n"),
            "if true {\n    print 1\n}\n"
        );
    }

    #[test]
    fn ignores_braces_inside_strings_and_comments() {
        let src = "print \"{x}\"\n# }\nif true {\nprint \"}\"\n}\n";
        assert_eq!(
            format_source(src),
            "print \"{x}\"\n# }\nif true {\n    print \"}\"\n}\n"
        );
    }

    #[test]
    fn is_idempotent() {
        let src = "function main() {\nprint 1\nif true {\nprint 2\n}\n}\n";
        let once = format_source(src);
        assert_eq!(format_source(&once), once);
    }

    #[test]
    fn keeps_else_at_block_depth() {
        assert_eq!(
            format_source("if true {\nprint 1\n} else {\nprint 2\n}\n"),
            "if true {\n    print 1\n} else {\n    print 2\n}\n"
        );
    }
}
