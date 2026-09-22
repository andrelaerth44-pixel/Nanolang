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

        let (_, closes) = scan_braces(line);
        let starts_with_close = line.starts_with('}');
        let leading_dedent = if starts_with_close { 1 } else { closes.min(indent) };
        if starts_with_close {
            indent = indent.saturating_sub(1);
        }

        for _ in 0..indent {
            out.push_str("    ");
        }
        out.push_str(line);
        out.push('\n');

        let (opens, closes) = scan_braces(line);
        indent = indent
            .saturating_add(opens)
            .saturating_sub(closes.max(leading_dedent));
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
    use super::format_source;

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
}
