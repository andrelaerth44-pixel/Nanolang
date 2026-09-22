use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{self, Read, Write},
};

#[derive(Default)]
struct Server {
    documents: HashMap<String, String>,
    shutdown: bool,
}

#[derive(Clone)]
struct Diagnostic {
    line: usize,
    character: usize,
    end_line: usize,
    end_character: usize,
    message: String,
    severity: u8,
}

fn main() {
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input).unwrap_or(0);
    let mut out = io::stdout();
    let mut server = Server::default();
    let mut pos = 0usize;

    while pos < input.len() {
        let Some(offset) = input[pos..]
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
        else {
            break;
        };
        let header_end = pos + offset;
        let header = String::from_utf8_lossy(&input[pos..header_end]);
        let len = header
            .lines()
            .find_map(|line| {
                line.split_once(':')
                    .filter(|(k, _)| k.eq_ignore_ascii_case("Content-Length"))
                    .and_then(|(_, v)| v.trim().parse::<usize>().ok())
            })
            .unwrap_or(0);
        let body_start = header_end + 4;
        if body_start + len > input.len() {
            break;
        }

        let body = &input[body_start..body_start + len];
        let Ok(request) = serde_json::from_slice::<Value>(body) else {
            pos = body_start + len;
            continue;
        };

        let id = request.get("id").cloned();
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("");

        if method.starts_with("$/") || id.is_none() {
            if let Some(notification) = handle_notification(&mut server, &request) {
                write_lsp(&mut out, &notification);
            }
        } else {
            let result = handle_request(&mut server, method, &request);
            let response = json!({
                "jsonrpc": "2.0",
                "id": id.unwrap(),
                "result": result
            });
            write_lsp(&mut out, &response);
        }

        if server.shutdown && method == "exit" {
            break;
        }
        pos = body_start + len;
    }
}

fn handle_notification(server: &mut Server, request: &Value) -> Option<Value> {
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialized" | "$/cancelRequest" => return None,
        "textDocument/didOpen" => {
            let Some(item) = request.pointer("/params/textDocument") else { return None; };
            let uri = item.get("uri").and_then(Value::as_str).unwrap_or("").to_string();
            let text = item.get("text").and_then(Value::as_str).unwrap_or("").to_string();
            server.documents.insert(uri.clone(), text.clone());
            return Some(publish_diagnostics(&uri, &text));
        }
        "textDocument/didChange" => {
            let Some(params) = request.get("params") else { return None; };
            let uri = params
                .pointer("/textDocument/uri")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let Some(changes) = params.get("contentChanges").and_then(Value::as_array) else { return None; };
            if let Some(change) = changes.last() {
                if let Some(text) = change.get("text").and_then(Value::as_str) {
                    server.documents.insert(uri.clone(), text.to_string());
                    return Some(publish_diagnostics(&uri, text));
                }
            }
            None
        }
        "textDocument/didClose" => {
            if let Some(uri) = request.pointer("/params/textDocument/uri").and_then(Value::as_str) {
                server.documents.remove(uri);
            }
            None
        }
        _ => None,
    }
}

fn publish_diagnostics(uri: &str, text: &str) -> Value {
    let diagnostics: Vec<Value> = analyze(text).into_iter().map(|d| json!({
        "range": {
            "start": { "line": d.line, "character": d.character },
            "end": { "line": d.end_line, "character": d.end_character }
        },
        "severity": d.severity,
        "source": "nano",
        "message": d.message
    })).collect();

    json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": uri,
            "diagnostics": diagnostics
        }
    })
}

fn handle_request(server: &mut Server, method: &str, request: &Value) -> Value {
    match method {
        "initialize" => json!({
            "capabilities": {
                "textDocumentSync": { "openClose": true, "change": 1 },
                "completionProvider": {
                    "triggerCharacters": [".", "_"]
                },
                "hoverProvider": true,
                "documentFormattingProvider": true,
                "codeActionProvider": true
            },
            "serverInfo": {
                "name": "nano-lsp",
                "version": "0.2.0"
            }
        }),
        "shutdown" => {
            server.shutdown = true;
            Value::Null
        }
        "exit" => Value::Null,
        "textDocument/completion" => completion(server, request),
        "textDocument/hover" => hover(server, request),
        "textDocument/formatting" => formatting(server, request),
        "textDocument/codeAction" => code_actions(server, request),
        "textDocument/diagnostic" => document_diagnostic(server, request),
        _ => Value::Null,
    }
}

fn completion(server: &Server, request: &Value) -> Value {
    let text = document_text(server, request);
    let words = [
        ("function", 14, "declara uma função"),
        ("if", 14, "condição"),
        ("else", 14, "ramo alternativo"),
        ("while", 14, "laço while"),
        ("for", 14, "laço for"),
        ("in", 14, "iterador"),
        ("return", 14, "retorna de uma função"),
        ("true", 14, "booleano verdadeiro"),
        ("false", 14, "booleano falso"),
        ("use", 14, "carrega um módulo"),
        ("print", 3, "imprime um valor"),
        ("len", 3, "tamanho de texto, lista, objeto ou tensor"),
        ("range", 3, "gera números de 0 até limite"),
        ("tensor", 3, "cria um tensor"),
        ("parameter", 3, "cria um parâmetro treinável"),
        ("zeros", 3, "cria um tensor preenchido com zero"),
        ("shape", 3, "retorna o shape do tensor"),
        ("matmul", 3, "multiplicação de matrizes"),
        ("sum", 3, "redução soma"),
        ("mean", 3, "redução média"),
        ("grad", 3, "gradiente automático"),
        ("step", 3, "atualização SGD"),
        ("adam", 3, "atualização Adam"),
        ("cast", 3, "conversão de dtype"),
        ("dtype", 3, "dtype do tensor"),
        ("device", 3, "dispositivo do tensor"),
        ("backend", 3, "backend ativo"),
        ("memory_bytes", 3, "memória do tensor"),
    ];

    let mut items = Vec::new();
    for (label, kind, detail) in words {
        items.push(json!({
            "label": label,
            "kind": kind,
            "detail": detail,
            "insertText": label
        }));
    }

    let mut seen_functions = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("function ") {
            if let Some(name) = rest.split(['(', ' ', '\t']).next() {
                if !name.is_empty() {
                    seen_functions.push(name.to_string());
                }
            }
        }
    }
    for name in seen_functions {
        items.push(json!({
            "label": name,
            "kind": 3,
            "detail": "função do arquivo",
            "insertText": name
        }));
    }

    json!({
        "isIncomplete": false,
        "items": items
    })
}

fn hover(server: &Server, request: &Value) -> Value {
    let text = document_text(server, request);
    let word = word_at_position(&text, request);
    let Some(word) = word else { return Value::Null; };

    let description = match word.as_str() {
        "function" => "Declara uma função Nano: function nome(arg1, arg2) { ... }",
        "if" => "Executa um bloco quando a expressão é verdadeira.",
        "for" => "Itera uma variável sobre uma List.",
        "tensor" => "Cria um Tensor a partir de dados e shape.",
        "parameter" => "Cria um Tensor treinável com requires_grad.",
        "matmul" => "Calcula multiplicação matricial com o backend ativo.",
        "grad" => "Calcula o gradiente de uma loss em relação a um parâmetro.",
        "adam" => "Aplica uma atualização Adam ao parâmetro.",
        "step" => "Aplica uma atualização SGD ao parâmetro.",
        "std.fs" => "Módulo de filesystem do runtime Nano.",
        "std.net" => "Módulo de rede do runtime Nano.",
        "std.async" => "Módulo de concorrência e tarefas do runtime Nano.",
        _ => {
            if text.lines().any(|line| line.trim_start().starts_with(&format!("function {word}"))) {
                "Função definida neste documento."
            } else {
                return Value::Null;
            }
        }
    };

    json!({
        "contents": {
            "kind": "markdown",
            "value": format!("**{}**\n\n{}", word, description)
        }
    })
}

fn formatting(server: &Server, request: &Value) -> Value {
    let text = document_text(server, request);
    let formatted = format_source(&text);
    if formatted == text {
        return json!([]);
    }

    let (last_line, last_char) = end_position(&text);
    json!([{
        "range": {
            "start": { "line": 0, "character": 0 },
            "end": { "line": last_line, "character": last_char }
        },
        "newText": formatted
    }])
}

fn code_actions(server: &Server, request: &Value) -> Value {
    let text = document_text(server, request);
    let diagnostics = analyze(&text);
    let mut actions = Vec::new();

    if diagnostics.iter().any(|d| d.message.contains("bloco não terminado")) {
        let (line, character) = end_position(&text);
        actions.push(json!({
            "title": "Adicionar '}' faltando",
            "kind": "quickfix",
            "edit": {
                "changes": {
                    document_uri(request): [{
                        "range": {
                            "start": { "line": line, "character": character },
                            "end": { "line": line, "character": character }
                        },
                        "newText": "\n}"
                    }]
                }
            }
        }));
    }

    for (line_no, line) in text.lines().enumerate() {
        if let Some(column) = line.find("//") {
            actions.push(json!({
                "title": "Trocar comentário // por #",
                "kind": "quickfix",
                "edit": {
                    "changes": {
                        document_uri(request): [{
                            "range": {
                                "start": { "line": line_no, "character": column },
                                "end": { "line": line_no, "character": column + 2 }
                            },
                            "newText": "#"
                        }]
                    }
                }
            }));
            break;
        }
    }

    actions.push(json!({
        "title": "Formatar documento com Nano Formatter",
        "kind": "source.formatDocument",
        "edit": {
            "changes": {
                document_uri(request): [{
                    "range": {
                        "start": { "line": 0, "character": 0 },
                        "end": {
                            "line": end_position(&text).0,
                            "character": end_position(&text).1
                        }
                    },
                    "newText": format_source(&text)
                }]
            }
        }
    }));

    Value::Array(actions)
}

fn document_diagnostic(server: &Server, request: &Value) -> Value {
    let text = document_text(server, request);
    let items: Vec<Value> = analyze(&text).into_iter().map(|d| {
        json!({
            "range": {
                "start": { "line": d.line, "character": d.character },
                "end": { "line": d.end_line, "character": d.end_character }
            },
            "severity": d.severity,
            "source": "nano",
            "message": d.message
        })
    }).collect();

    json!({ "kind": "full", "items": items })
}

fn analyze(src: &str) -> Vec<Diagnostic> {
    let mut result = Vec::new();
    let mut stack: Vec<(char, usize, usize)> = Vec::new();

    for (line_no, line) in src.lines().enumerate() {
        let mut in_string = false;
        let mut escape = false;
        for (column, ch) in line.chars().enumerate() {
            if in_string {
                if escape {
                    escape = false;
                } else if ch == '\\' {
                    escape = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            if ch == '"' {
                in_string = true;
                continue;
            }
            if ch == '#' {
                break;
            }
            if ch == '/' && line.chars().nth(column + 1) == Some('/') {
                result.push(Diagnostic {
                    line: line_no,
                    character: column,
                    end_line: line_no,
                    end_character: column + 2,
                    message: "Nano usa # para comentários; // não faz parte da sintaxe atual.".into(),
                    severity: 2,
                });
                break;
            }
            match ch {
                '(' | '[' | '{' => stack.push((ch, line_no, column)),
                ')' | ']' | '}' => {
                    let expected = match ch { ')' => '(', ']' => '[', _ => '{' };
                    if stack.last().map(|x| x.0) != Some(expected) {
                        result.push(Diagnostic {
                            line: line_no,
                            character: column,
                            end_line: line_no,
                            end_character: column + 1,
                            message: format!("'{ch}' não fecha o bloco esperado."),
                            severity: 1,
                        });
                    } else {
                        stack.pop();
                    }
                }
                _ => {}
            }
        }
    }

    for (open, line, column) in stack {
        let close = match open { '(' => ')', '[' => ']', _ => '}' };
        result.push(Diagnostic {
            line,
            character: column,
            end_line: line,
            end_character: column + 1,
            message: format!("bloco não terminado: falta '{close}'."),
            severity: 1,
        });
    }
    result
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

        let (_, _closes) = scan_braces(line);
        let starts_with_close = line.starts_with('}');
        if starts_with_close {
            indent = indent.saturating_sub(1);
        }

        for _ in 0..indent {
            out.push_str("    ");
        }
        out.push_str(line);
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

fn scan_braces(line: &str) -> (usize, usize) {
    let mut opens = 0usize;
    let mut closes = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for c in line.chars() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
        } else if c == '#' {
            break;
        } else if c == '{' {
            opens += 1;
        } else if c == '}' {
            closes += 1;
        }
    }
    (opens, closes)
}

fn document_text(server: &Server, request: &Value) -> String {
    document_uri_value(request)
        .and_then(|uri| server.documents.get(&uri).cloned())
        .unwrap_or_default()
}

fn document_uri(request: &Value) -> String {
    document_uri_value(request).unwrap_or_default()
}

fn document_uri_value(request: &Value) -> Option<String> {
    request
        .pointer("/params/textDocument/uri")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn word_at_position(text: &str, request: &Value) -> Option<String> {
    let line = request
        .pointer("/params/position/line")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let character = request
        .pointer("/params/position/character")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let row = text.lines().nth(line)?;
    let chars: Vec<char> = row.chars().collect();
    let mut start = character.min(chars.len());
    let mut end = start;
    while start > 0 && is_word(chars[start - 1]) { start -= 1; }
    while end < chars.len() && is_word(chars[end]) { end += 1; }
    if start == end { None } else { Some(chars[start..end].iter().collect()) }
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn end_position(text: &str) -> (usize, usize) {
    let line = text.lines().count().saturating_sub(1);
    let last = text.lines().last().unwrap_or("");
    (line, last.chars().count())
}

fn write_lsp(out: &mut impl Write, body: &Value) {
    let encoded = serde_json::to_string(body).unwrap();
    let _ = write!(out, "Content-Length: {}\r\n\r\n{}", encoded.len(), encoded);
    let _ = out.flush();
}
