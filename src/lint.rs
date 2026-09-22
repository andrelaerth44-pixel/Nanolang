use std::collections::{HashMap, HashSet};

use crate::{Expr, Stmt, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Severity {
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) struct Diagnostic {
    pub(crate) severity: Severity,
    pub(crate) message: String,
}

pub(crate) fn lint(program: &[Stmt]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut functions = HashMap::<String, usize>::new();
    let mut calls = HashSet::<String>::new();

    for stmt in program {
        collect_function_decls(stmt, &mut functions);
        collect_calls_stmt(stmt, &mut calls);
    }

    for (name, count) in &functions {
        if *count > 1 {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                message: format!("função '{name}' declarada mais de uma vez"),
            });
        }
        if name == "main" && *count > 1 {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                message: "main deve ser única".into(),
            });
        }
    }

    for (name, _) in &functions {
        if name != "main" && !calls.contains(name) {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                message: format!("função '{name}' não é chamada"),
            });
        }
    }

    lint_block(program, false, &mut diagnostics);
    diagnostics
}

fn collect_function_decls(stmt: &Stmt, functions: &mut HashMap<String, usize>) {
    match stmt {
        Stmt::Function(name, _, body) => {
            *functions.entry(name.clone()).or_default() += 1;
            for child in body {
                collect_function_decls(child, functions);
            }
        }
        Stmt::If(_, yes, no) => {
            for child in yes.iter().chain(no.iter()) {
                collect_function_decls(child, functions);
            }
        }
        Stmt::While(_, body) | Stmt::For(_, _, body) => {
            for child in body {
                collect_function_decls(child, functions);
            }
        }
        _ => {}
    }
}

fn collect_calls_stmt(stmt: &Stmt, calls: &mut HashSet<String>) {
    match stmt {
        Stmt::Assign(_, expr)
        | Stmt::Print(expr)
        | Stmt::Expr(expr)
        | Stmt::Return(expr) => collect_calls_expr(expr, calls),
        Stmt::If(cond, yes, no) => {
            collect_calls_expr(cond, calls);
            for child in yes.iter().chain(no.iter()) {
                collect_calls_stmt(child, calls);
            }
        }
        Stmt::While(cond, body) => {
            collect_calls_expr(cond, calls);
            for child in body {
                collect_calls_stmt(child, calls);
            }
        }
        Stmt::For(_, iterable, body) => {
            collect_calls_expr(iterable, calls);
            for child in body {
                collect_calls_stmt(child, calls);
            }
        }
        Stmt::Function(_, _, body) => {
            for child in body {
                collect_calls_stmt(child, calls);
            }
        }
        Stmt::Use(_) | Stmt::Break => {}
    }
}

fn collect_calls_expr(expr: &Expr, calls: &mut HashSet<String>) {
    match expr {
        Expr::Call(name, args) => {
            calls.insert(name.clone());
            for arg in args {
                collect_calls_expr(arg, calls);
            }
        }
        Expr::Binary(left, _, right) => {
            collect_calls_expr(left, calls);
            collect_calls_expr(right, calls);
        }
        Expr::Unary(_, inner) => collect_calls_expr(inner, calls),
        Expr::List(items) => {
            for item in items {
                collect_calls_expr(item, calls);
            }
        }
        Expr::Object(fields) => {
            for (_, value) in fields {
                collect_calls_expr(value, calls);
            }
        }
        Expr::Index(target, index) => {
            collect_calls_expr(target, calls);
            collect_calls_expr(index, calls);
        }
        Expr::Field(target, _) => collect_calls_expr(target, calls),
        Expr::Value(_) | Expr::Var(_) => {}
    }
}

fn lint_block(stmts: &[Stmt], inside_loop: bool, diagnostics: &mut Vec<Diagnostic>) {
    if stmts.is_empty() {
        diagnostics.push(Diagnostic {
            severity: Severity::Warning,
            message: "bloco vazio".into(),
        });
        return;
    }

    let mut terminated = false;

    for stmt in stmts {
        if terminated {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                message: "código inalcançável após terminador do bloco".into(),
            });
            break;
        }

        match stmt {
            Stmt::Return(_) | Stmt::Break if inside_loop => {
                if matches!(stmt, Stmt::Break) {
                    terminated = true;
                }
            }
            Stmt::Return(_) => {
                terminated = true;
            }
            Stmt::Break => {
                terminated = true;
            }
            Stmt::If(cond, yes, no) => {
                if let Some(value) = constant_truthy(cond) {
                    if *value {
                        diagnostics.push(Diagnostic {
                            severity: Severity::Warning,
                            message: "condição do if é sempre verdadeira".into(),
                        });
                    } else {
                        diagnostics.push(Diagnostic {
                            severity: Severity::Warning,
                            message: "condição do if é sempre falsa".into(),
                        });
                    }
                }
                lint_block(yes, inside_loop, diagnostics);
                if !no.is_empty() {
                    lint_block(no, inside_loop, diagnostics);
                }
            }
            Stmt::While(cond, body) => {
                if matches!(constant_truthy(cond), Some(false)) {
                    diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "while nunca será executado".into(),
                    });
                }
                lint_block(body, true, diagnostics);
            }
            Stmt::For(_, iterable, body) => {
                if matches!(iterable, Expr::List(items) if items.is_empty()) {
                    diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "for sobre lista vazia não executa".into(),
                    });
                }
                lint_block(body, true, diagnostics);
            }
            Stmt::Function(_, params, body) => {
                let used = used_variables(body);
                for param in params {
                    if !used.contains(param) {
                        diagnostics.push(Diagnostic {
                            severity: Severity::Warning,
                            message: format!("parâmetro '{param}' não é usado"),
                        });
                    }
                }
                lint_block(body, false, diagnostics);
            }
            Stmt::Assign(_, expr) | Stmt::Print(expr) | Stmt::Expr(expr) | Stmt::Return(expr) => {
                lint_expr(expr, diagnostics);
            }
            Stmt::Use(_) => {}
        }
    }
}

fn lint_expr(expr: &Expr, diagnostics: &mut Vec<Diagnostic>) {
    match expr {
        Expr::Binary(_, op, right) => {
            if matches!(op, crate::Op::Div | crate::Op::Mod)
                && matches!(right.as_ref(), Expr::Value(Value::Number(value)) if *value == 0.0)
            {
                diagnostics.push(Diagnostic {
                    severity: Severity::Warning,
                    message: "divisão ou resto por zero detectado estaticamente".into(),
                });
            }
            if let crate::Op::Add = op {
                if matches!(right.as_ref(), Expr::Value(Value::Text(text)) if text.is_empty()) {
                    diagnostics.push(Diagnostic {
                        severity: Severity::Warning,
                        message: "concatenação com texto vazio".into(),
                    });
                }
            }
            lint_expr(right, diagnostics);
        }
        Expr::Unary(_, inner) => lint_expr(inner, diagnostics),
        Expr::Call(_, args) => {
            for arg in args {
                lint_expr(arg, diagnostics);
            }
        }
        Expr::List(items) => {
            for item in items {
                lint_expr(item, diagnostics);
            }
        }
        Expr::Object(fields) => {
            for (_, value) in fields {
                lint_expr(value, diagnostics);
            }
        }
        Expr::Index(target, index) => {
            lint_expr(target, diagnostics);
            lint_expr(index, diagnostics);
        }
        Expr::Field(target, _) => lint_expr(target, diagnostics),
        Expr::Value(_) | Expr::Var(_) => {}
    }
}

fn constant_truthy(expr: &Expr) -> Option<&bool> {
    match expr {
        Expr::Value(Value::Boolean(value)) => Some(value),
        _ => None,
    }
}

fn used_variables(body: &[Stmt]) -> HashSet<String> {
    let mut used = HashSet::new();
    for stmt in body {
        used_stmt(stmt, &mut used);
    }
    used
}

fn used_stmt(stmt: &Stmt, used: &mut HashSet<String>) {
    match stmt {
        Stmt::Assign(_, expr)
        | Stmt::Print(expr)
        | Stmt::Expr(expr)
        | Stmt::Return(expr) => used_expr(expr, used),
        Stmt::If(cond, yes, no) => {
            used_expr(cond, used);
            for child in yes.iter().chain(no.iter()) { used_stmt(child, used); }
        }
        Stmt::While(cond, body) => {
            used_expr(cond, used);
            for child in body { used_stmt(child, used); }
        }
        Stmt::For(_, iterable, body) => {
            used_expr(iterable, used);
            for child in body { used_stmt(child, used); }
        }
        Stmt::Function(_, params, body) => {
            for param in params { used.remove(param); }
            for child in body { used_stmt(child, used); }
        }
        Stmt::Break | Stmt::Use(_) => {}
    }
}

fn used_expr(expr: &Expr, used: &mut HashSet<String>) {
    match expr {
        Expr::Var(name) => { used.insert(name.clone()); }
        Expr::Binary(left, _, right) => { used_expr(left, used); used_expr(right, used); }
        Expr::Unary(_, inner) => used_expr(inner, used),
        Expr::Call(_, args) => for arg in args { used_expr(arg, used); },
        Expr::List(items) => for item in items { used_expr(item, used); },
        Expr::Object(fields) => for (_, value) in fields { used_expr(value, used); },
        Expr::Index(target, index) => { used_expr(target, used); used_expr(index, used); }
        Expr::Field(target, _) => used_expr(target, used),
        Expr::Value(_) => {}
    }
}
