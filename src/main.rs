use std::{env, fs, process};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String), Number(f64), Text(String),
    True, False, Function, Print, If, Else, Return,
    Plus, Minus, Star, Slash, Equal, EqualEqual, BangEqual,
    Greater, GreaterEqual, Less, LessEqual,
    LeftParen, RightParen, LeftBrace, RightBrace, LeftBracket, RightBracket, Comma, Colon, Dot, Eof,
}

struct Lexer { src: Vec<char>, pos: usize }

impl Lexer {
    fn new(src: &str) -> Self { Self { src: src.chars().collect(), pos: 0 } }
    fn peek(&self) -> Option<char> { self.src.get(self.pos).copied() }
    fn advance(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() { self.pos += 1; }
        c
    }
    fn lex(&mut self) -> Result<Vec<Token>, String> {
        let mut out = Vec::new();
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\n' | '\r' | '\t' => { self.advance(); }
                '#' => { while !matches!(self.peek(), None | Some('\n')) { self.advance(); } }
                '"' => out.push(self.string()?),
                '0'..='9' => out.push(self.number()),
                'a'..='z' | 'A'..='Z' | '_' => out.push(self.ident()),
                '+' => { self.advance(); out.push(Token::Plus); }
                '-' => { self.advance(); out.push(Token::Minus); }
                '*' => { self.advance(); out.push(Token::Star); }
                '/' => { self.advance(); out.push(Token::Slash); }
                '(' => { self.advance(); out.push(Token::LeftParen); }
                ')' => { self.advance(); out.push(Token::RightParen); }
                '{' => { self.advance(); out.push(Token::LeftBrace); }
                '}' => { self.advance(); out.push(Token::RightBrace); }
                '[' => { self.advance(); out.push(Token::LeftBracket); }
                ']' => { self.advance(); out.push(Token::RightBracket); }
                ',' => { self.advance(); out.push(Token::Comma); }
                ':' => { self.advance(); out.push(Token::Colon); }
                '.' => { self.advance(); out.push(Token::Dot); }
                '=' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); out.push(Token::EqualEqual); }
                    else { out.push(Token::Equal); }
                }
                '!' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); out.push(Token::BangEqual); }
                    else { return Err("Nano: '!' isolado não é válido na v0.1".into()); }
                }
                '>' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); out.push(Token::GreaterEqual); }
                    else { out.push(Token::Greater); }
                }
                '<' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); out.push(Token::LessEqual); }
                    else { out.push(Token::Less); }
                }
                _ => return Err(format!("Nano: caractere inesperado '{c}'")),
            }
        }
        out.push(Token::Eof);
        Ok(out)
    }
    fn string(&mut self) -> Result<Token, String> {
        self.advance();
        let mut v = String::new();
        while let Some(c) = self.peek() {
            self.advance();
            match c {
                '"' => return Ok(Token::Text(v)),
                '\\' => match self.advance() {
                    Some('n') => v.push('\n'),
                    Some('t') => v.push('\t'),
                    Some('"') => v.push('"'),
                    Some('\\') => v.push('\\'),
                    Some(x) => return Err(format!("Nano: escape inválido \\{x}")),
                    None => return Err("Nano: string não terminada".into()),
                },
                _ => v.push(c),
            }
        }
        Err("Nano: string não terminada".into())
    }
    fn number(&mut self) -> Token {
        let start = self.pos;
        while matches!(self.peek(), Some('0'..='9' | '.')) { self.advance(); }
        let s: String = self.src[start..self.pos].iter().collect();
        Token::Number(s.parse().unwrap_or(0.0))
    }
    fn ident(&mut self) -> Token {
        let start = self.pos;
        while matches!(self.peek(), Some('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) { self.advance(); }
        match self.src[start..self.pos].iter().collect::<String>().as_str() {
            "true" => Token::True, "false" => Token::False,
            "function" => Token::Function,
            "print" => Token::Print, "if" => Token::If,
            "else" => Token::Else, "return" => Token::Return,
            s => Token::Ident(s.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
enum Value { Number(f64), Text(String), Boolean(bool), List(Vec<Value>), Object(HashMap<String, Value>), Null }

impl Value {
    fn truthy(&self) -> bool {
        match self {
            Self::Boolean(v) => *v,
            Self::Number(v) => *v != 0.0,
            Self::Text(v) => !v.is_empty(),
            Self::List(v) => !v.is_empty(),
            Self::Object(v) => !v.is_empty(),
            Self::Null => false,
        }
    }
    fn show(&self) -> String {
        match self {
            Self::Number(v) if v.fract() == 0.0 => format!("{}", *v as i64),
            Self::Number(v) => v.to_string(),
            Self::Text(v) => v.clone(),
            Self::Boolean(v) => v.to_string(),
            Self::List(v) => format!("[{}]", v.iter().map(|x| x.show()).collect::<Vec<_>>().join(", ")),
            Self::Object(v) => {
                let mut items = v.iter().map(|(k, val)| format!("{}: {}", k, val.show())).collect::<Vec<_>>();
                items.sort();
                format!("{{{}}}", items.join(", "))
            }
            Self::Null => "null".into(),
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Value(Value), Var(String), List(Vec<Expr>), Object(Vec<(String, Expr)>),
    Binary(Box<Expr>, Op, Box<Expr>), Call(String, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>), Field(Box<Expr>, String),
}

#[derive(Debug, Clone, Copy)]
enum Op { Add, Sub, Mul, Div, Eq, Ne, Gt, Ge, Lt, Le }

#[derive(Debug, Clone)]
enum Stmt {
    Assign(String, Expr), Print(Expr), Expr(Expr),
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    Function(String, Vec<String>, Vec<Stmt>),
    Return(Expr),
}

struct Parser { tokens: Vec<Token>, pos: usize }

impl Parser {
    fn new(tokens: Vec<Token>) -> Self { Self { tokens, pos: 0 } }
    fn peek(&self) -> &Token { &self.tokens[self.pos] }
    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos + 1 < self.tokens.len() { self.pos += 1; }
        t
    }
    fn expect(&mut self, wanted: Token) -> Result<(), String> {
        if *self.peek() == wanted { self.advance(); Ok(()) }
        else { Err(format!("Nano: esperado {:?}, encontrado {:?}", wanted, self.peek())) }
    }
    fn program(&mut self) -> Result<Vec<Stmt>, String> {
        let mut out = Vec::new();
        while !matches!(self.peek(), Token::Eof) { out.push(self.statement()?); }
        Ok(out)
    }
    fn statement(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Token::Function => self.function(),
            Token::Print => { self.advance(); Ok(Stmt::Print(self.expression()?)) }
            Token::If => self.if_stmt(),
            Token::Return => { self.advance(); Ok(Stmt::Return(self.expression()?)) }
            Token::Ident(name) => {
                let name = name.clone();
                if matches!(self.tokens.get(self.pos + 1), Some(Token::Equal)) {
                    self.advance(); self.advance();
                    Ok(Stmt::Assign(name, self.expression()?))
                } else { Ok(Stmt::Expr(self.expression()?)) }
            }
            _ => Ok(Stmt::Expr(self.expression()?)),
        }
    }
    fn function(&mut self) -> Result<Stmt, String> {
        self.advance();
        let name = match self.advance() {
            Token::Ident(v) => v,
            x => return Err(format!("Nano: nome da função esperado, encontrado {:?}", x)),
        };
        self.expect(Token::LeftParen)?;
        let mut params = Vec::new();
        if !matches!(self.peek(), Token::RightParen) {
            loop {
                match self.advance() {
                    Token::Ident(v) => params.push(v),
                    x => return Err(format!("Nano: parâmetro esperado, encontrado {:?}", x)),
                }
                if matches!(self.peek(), Token::RightParen) { break; }
                self.expect(Token::Comma)?;
            }
        }
        self.expect(Token::RightParen)?;
        Ok(Stmt::Function(name, params, self.block()?))
    }
    fn if_stmt(&mut self) -> Result<Stmt, String> {
        self.advance();
        let cond = self.expression()?;
        let yes = self.block()?;
        let no = if matches!(self.peek(), Token::Else) { self.advance(); self.block()? } else { Vec::new() };
        Ok(Stmt::If(cond, yes, no))
    }
    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(Token::LeftBrace)?;
        let mut out = Vec::new();
        while !matches!(self.peek(), Token::RightBrace) {
            if matches!(self.peek(), Token::Eof) { return Err("Nano: bloco não terminado".into()); }
            out.push(self.statement()?);
        }
        self.expect(Token::RightBrace)?;
        Ok(out)
    }
    fn expression(&mut self) -> Result<Expr, String> { self.equality() }
    fn equality(&mut self) -> Result<Expr, String> {
        let mut left = self.compare()?;
        loop {
            let op = match self.peek() {
                Token::EqualEqual => Op::Eq, Token::BangEqual => Op::Ne, _ => break,
            };
            self.advance();
            left = Expr::Binary(Box::new(left), op, Box::new(self.compare()?));
        }
        Ok(left)
    }
    fn compare(&mut self) -> Result<Expr, String> {
        let mut left = self.term()?;
        loop {
            let op = match self.peek() {
                Token::Greater => Op::Gt, Token::GreaterEqual => Op::Ge,
                Token::Less => Op::Lt, Token::LessEqual => Op::Le, _ => break,
            };
            self.advance();
            left = Expr::Binary(Box::new(left), op, Box::new(self.term()?));
        }
        Ok(left)
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut left = self.factor()?;
        loop {
            let op = match self.peek() { Token::Plus => Op::Add, Token::Minus => Op::Sub, _ => break };
            self.advance();
            left = Expr::Binary(Box::new(left), op, Box::new(self.factor()?));
        }
        Ok(left)
    }
    fn factor(&mut self) -> Result<Expr, String> {
        let mut left = self.primary()?;
        loop {
            let op = match self.peek() { Token::Star => Op::Mul, Token::Slash => Op::Div, _ => break };
            self.advance();
            left = Expr::Binary(Box::new(left), op, Box::new(self.primary()?));
        }
        Ok(left)
    }
    fn primary(&mut self) -> Result<Expr, String> {
        let mut expr = match self.advance() {
            Token::Number(v) => Expr::Value(Value::Number(v)),
            Token::Text(v) => Expr::Value(Value::Text(v)),
            Token::True => Expr::Value(Value::Boolean(true)),
            Token::False => Expr::Value(Value::Boolean(false)),
            Token::Ident(name) => Expr::Var(name),
            Token::LeftParen => {
                let e = self.expression()?;
                self.expect(Token::RightParen)?;
                e
            }
            Token::LeftBracket => {
                let mut items = Vec::new();
                if !matches!(self.peek(), Token::RightBracket) {
                    loop {
                        items.push(self.expression()?);
                        if matches!(self.peek(), Token::RightBracket) { break; }
                        self.expect(Token::Comma)?;
                    }
                }
                self.expect(Token::RightBracket)?;
                Expr::List(items)
            }
            Token::LeftBrace => {
                let mut fields = Vec::new();
                if !matches!(self.peek(), Token::RightBrace) {
                    loop {
                        let key = match self.advance() {
                            Token::Ident(v) => v,
                            Token::Text(v) => v,
                            x => return Err(format!("Nano: chave esperada no objeto, encontrado {:?}", x)),
                        };
                        self.expect(Token::Colon)?;
                        fields.push((key, self.expression()?));
                        if matches!(self.peek(), Token::RightBrace) { break; }
                        self.expect(Token::Comma)?;
                    }
                }
                self.expect(Token::RightBrace)?;
                Expr::Object(fields)
            }
            x => return Err(format!("Nano: expressão inesperada {:?}", x)),
        };

        loop {
            match self.peek() {
                Token::LeftParen => {
                    self.advance();
                    let name = match expr {
                        Expr::Var(n) => n,
                        _ => return Err("Nano: chamada deve usar o nome de uma função na v0.2".into()),
                    };
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Token::RightParen) {
                        loop {
                            args.push(self.expression()?);
                            if matches!(self.peek(), Token::RightParen) { break; }
                            self.expect(Token::Comma)?;
                        }
                    }
                    self.expect(Token::RightParen)?;
                    expr = Expr::Call(name, args);
                }
                Token::LeftBracket => {
                    self.advance();
                    let index = self.expression()?;
                    self.expect(Token::RightBracket)?;
                    expr = Expr::Index(Box::new(expr), Box::new(index));
                }
                Token::Dot => {
                    self.advance();
                    let name = match self.advance() {
                        Token::Ident(v) => v,
                        x => return Err(format!("Nano: campo esperado depois de '.', encontrado {:?}", x)),
                    };
                    expr = Expr::Field(Box::new(expr), name);
                }
                _ => break,
            }
        }
        Ok(expr)
    }
}

struct Runtime {
    vars: HashMap<String, Value>,
    functions: HashMap<String, (Vec<String>, Vec<Stmt>)>,
}

impl Runtime {
    fn new() -> Self { Self { vars: HashMap::new(), functions: HashMap::new() } }

    fn run(&mut self, program: &[Stmt]) -> Result<(), String> {
        for stmt in program {
            if self.exec(stmt)?.is_some() { break; }
        }
        Ok(())
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<Option<Value>, String> {
        match stmt {
            Stmt::Assign(n, e) => { let v = self.eval(e)?; self.vars.insert(n.clone(), v); Ok(None) }
            Stmt::Print(e) => { println!("{}", self.eval(e)?.show()); Ok(None) }
            Stmt::Expr(e) => { self.eval(e)?; Ok(None) }
            Stmt::Function(n, p, b) => { self.functions.insert(n.clone(), (p.clone(), b.clone())); Ok(None) }
            Stmt::Return(e) => Ok(Some(self.eval(e)?)),
            Stmt::If(c, yes, no) => {
                let body = if self.eval(c)?.truthy() { yes } else { no };
                for s in body {
                    if let Some(v) = self.exec(s)? { return Ok(Some(v)); }
                }
                Ok(None)
            }
        }
    }

    fn eval(&mut self, e: &Expr) -> Result<Value, String> {
        match e {
            Expr::Value(v) => Ok(v.clone()),
            Expr::Var(n) => self.vars.get(n).cloned().ok_or_else(|| format!("Nano: variável '{n}' não definida")),
            Expr::List(items) => {
                let mut values = Vec::new();
                for item in items { values.push(self.eval(item)?); }
                Ok(Value::List(values))
            },
            Expr::Object(fields) => {
                let mut values = HashMap::new();
                for (key, value) in fields { values.insert(key.clone(), self.eval(value)?); }
                Ok(Value::Object(values))
            },
            Expr::Binary(a, op, b) => {
                let left = self.eval(a)?;
                let right = self.eval(b)?;
                self.binary(left, *op, right)
            },
            Expr::Field(target, name) => {
                match self.eval(target)? {
                    Value::Object(values) => values.get(name).cloned()
                        .ok_or_else(|| format!("Nano: campo '{name}' não existe")),
                    _ => Err("Nano: '.' requer um objeto".into()),
                }
            },
            Expr::Index(target, index) => {
                let value = self.eval(target)?;
                let key = self.eval(index)?;
                match (value, key) {
                    (Value::List(values), Value::Number(n)) => {
                        if n < 0.0 || n.fract() != 0.0 { return Err("Nano: índice deve ser um número inteiro".into()); }
                        values.get(n as usize).cloned().ok_or_else(|| "Nano: índice fora do limite".into())
                    }
                    (Value::Object(values), Value::Text(key)) => values.get(&key).cloned()
                        .ok_or_else(|| format!("Nano: chave '{key}' não existe")),
                    _ => Err("Nano: indexação requer lista[número] ou objeto[texto]".into()),
                }
            },
            Expr::Call(name, args) => {
                if name == "len" {
                    if args.len() != 1 { return Err("Nano: len() recebe 1 argumento".into()); }
                    return match self.eval(&args[0])? {
                        Value::Text(v) => Ok(Value::Number(v.chars().count() as f64)),
                        Value::List(v) => Ok(Value::Number(v.len() as f64)),
                        Value::Object(v) => Ok(Value::Number(v.len() as f64)),
                        _ => Err("Nano: len() requer texto, lista ou objeto".into()),
                    };
                }
                let (params, body) = self.functions.get(name).cloned().ok_or_else(|| format!("Nano: função '{name}' não definida"))?;
                if params.len() != args.len() { return Err(format!("Nano: '{name}' esperava {} argumentos", params.len())); }
                let saved = self.vars.clone();
                for (p, a) in params.iter().zip(args) {
                    let v = self.eval(a)?;
                    self.vars.insert(p.clone(), v);
                }
                let mut result = Value::Null;
                for s in &body {
                    if let Some(v) = self.exec(s)? { result = v; break; }
                }
                self.vars = saved;
                Ok(result)
            }
        }
    }

    fn binary(&self, a: Value, op: Op, b: Value) -> Result<Value, String> {
        match op {
            Op::Add => match (a, b) {
                (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x + y)),
                (Value::Text(x), Value::Text(y)) => Ok(Value::Text(x + &y)),
                (Value::Text(x), y) => Ok(Value::Text(x + &y.show())),
                (x, Value::Text(y)) => Ok(Value::Text(x.show() + &y)),
                (Value::List(mut x), Value::List(y)) => { x.extend(y); Ok(Value::List(x)) },
                _ => Err("Nano: '+' requer números, texto ou listas compatíveis".into()),
            },
            Op::Sub => num(a,b,|x,y| x-y),
            Op::Mul => num(a,b,|x,y| x*y),
            Op::Div => num(a,b,|x,y| x/y),
            Op::Eq => Ok(Value::Boolean(a.show() == b.show())),
            Op::Ne => Ok(Value::Boolean(a.show() != b.show())),
            Op::Gt => cmp(a,b,|x,y| x>y),
            Op::Ge => cmp(a,b,|x,y| x>=y),
            Op::Lt => cmp(a,b,|x,y| x<y),
            Op::Le => cmp(a,b,|x,y| x<=y),
        }
    }
}

fn num(a: Value, b: Value, f: fn(f64,f64)->f64) -> Result<Value,String> {
    match (a,b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Number(f(x,y))),
        _ => Err("Nano: operação requer números".into()),
    }
}

fn cmp(a: Value, b: Value, f: fn(f64,f64)->bool) -> Result<Value,String> {
    match (a,b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Boolean(f(x,y))),
        _ => Err("Nano: comparação requer números".into()),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = match args.as_slice() {
        [_, command] if command == "run" => "main.nano".to_string(),
        [_, command, file] if command == "run" => file.clone(),
        _ => {
            eprintln!("Nano 0.1 — uso: nano run [arquivo.nano]");
            process::exit(2);
        }
    };
    let source = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => { eprintln!("Nano: não foi possível ler '{path}': {e}"); process::exit(1); }
    };
    let tokens = match Lexer::new(&source).lex() {
        Ok(t) => t,
        Err(e) => { eprintln!("{e}"); process::exit(1); }
    };
    let program = match Parser::new(tokens).program() {
        Ok(p) => p,
        Err(e) => { eprintln!("{e}"); process::exit(1); }
    };
    if let Err(e) = Runtime::new().run(&program) {
        eprintln!("{e}");
        process::exit(1);
    }
}
