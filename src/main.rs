mod backend;
mod dtype;
mod gpu;
mod memory;
mod ir;
mod native;
mod npu;

use std::{cell::RefCell, env, fs, path::Path, process, rc::Rc, sync::atomic::{AtomicU64, Ordering}};
use std::collections::HashMap;
use dtype::DType;
use half::{bf16, f16};

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String), Number(f64), Text(String),
    True, False, Use, Function, Print, If, Else, Return, While, For, In,
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
            "use" => Token::Use, "function" => Token::Function,
            "print" => Token::Print, "if" => Token::If,
            "else" => Token::Else, "return" => Token::Return,
            "while" => Token::While, "for" => Token::For, "in" => Token::In,
            s => Token::Ident(s.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorOpKind {
    Add, Sub, Mul, Div,
}

type TensorRef = Rc<RefCell<Tensor>>;

#[derive(Debug, Clone)]
enum TensorOp {
    Leaf,
    Elementwise(TensorOpKind, TensorRef, TensorRef),
    Matmul(TensorRef, TensorRef),
    FusedMulAdd(TensorRef, TensorRef, TensorRef),
    Sum(TensorRef),
    Mean(TensorRef),
}

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

fn next_tensor_id() -> u64 {
    NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone)]
enum TensorStorage {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    Remote { len: usize, dtype: DType },
}

impl TensorStorage {
    fn from_f32(dtype: DType, data: Vec<f32>) -> Self {
        match dtype {
            DType::F32 => Self::F32(data),
            DType::F16 => Self::F16(data.into_iter().map(f16::from_f32).collect()),
            DType::BF16 => Self::BF16(data.into_iter().map(bf16::from_f32).collect()),
        }
    }

    fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::F32(v) => v.clone(),
            Self::F16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::BF16(v) => v.iter().map(|x| x.to_f32()).collect(),
            Self::Remote { .. } => panic!("Nano: dados do tensor ainda estão somente na GPU"),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::Remote { len, .. } => *len,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Self::F32(v) => v.len() * 4,
            Self::F16(v) => v.len() * 2,
            Self::BF16(v) => v.len() * 2,
            Self::Remote { len, dtype } => len * dtype.bytes(),
        }
    }
}

#[derive(Debug)]
struct Tensor {
    id: u64,
    storage: TensorStorage,
    shape: Vec<usize>,
    dtype: DType,
    requires_grad: bool,
    device: backend::BackendKind,
    op: TensorOp,
    host_valid: bool,
}

impl Tensor {
    fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<TensorRef, String> {
        Self::new_on(data, shape, requires_grad, backend::BackendKind::Cpu)
    }

    fn new_on(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool, device: backend::BackendKind) -> Result<TensorRef, String> {
        Self::new_with_dtype_on(data, shape, requires_grad, device, DType::F32)
    }

    fn new_with_dtype_on(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        device: backend::BackendKind,
        dtype: DType,
    ) -> Result<TensorRef, String> {
        let expected = shape.iter().copied().product::<usize>();
        if expected != data.len() {
            return Err(format!("Nano: tensor tem {} valores, mas a forma exige {}", data.len(), expected));
        }
        Ok(Rc::new(RefCell::new(Self {
            id: next_tensor_id(),
            storage: TensorStorage::from_f32(dtype, data),
            shape,
            dtype,
            requires_grad,
            device,
            op: TensorOp::Leaf,
            host_valid: true,
        })))
    }

    fn derived_on(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool, device: backend::BackendKind, op: TensorOp) -> Result<TensorRef, String> {
        Self::derived_dtype_on(data, shape, requires_grad, device, DType::F32, op)
    }

    fn derived_dtype_on(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        device: backend::BackendKind,
        dtype: DType,
        op: TensorOp,
    ) -> Result<TensorRef, String> {
        Self::derived_dtype_on_with_id(next_tensor_id(), data, shape, requires_grad, device, dtype, op)
    }

    fn derived_dtype_on_with_id(
        id: u64,
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        device: backend::BackendKind,
        dtype: DType,
        op: TensorOp,
    ) -> Result<TensorRef, String> {
        let expected=shape.iter().copied().product::<usize>();
        if expected!=data.len(){return Err(format!("Nano: tensor derivado tem {} valores, mas a forma exige {}",data.len(),expected));}
        Ok(Rc::new(RefCell::new(Self{id,storage:TensorStorage::from_f32(dtype,data),shape,dtype,requires_grad,device,op,host_valid:true})))
    }
    fn remote_with_id(
        id: u64,
        shape: Vec<usize>,
        requires_grad: bool,
        device: backend::BackendKind,
        dtype: DType,
        op: TensorOp,
    ) -> Result<TensorRef, String> {
        if device != backend::BackendKind::Gpu {
            return Err("Nano: tensor remoto exige backend GPU".into());
        }
        let expected=shape.iter().copied().product::<usize>();
        Ok(Rc::new(RefCell::new(Self{
            id,
            storage:TensorStorage::Remote{len:expected,dtype},
            shape,
            dtype,
            requires_grad,
            device,
            op,
            host_valid:false,
        })))
    }

    fn data_f32(&self) -> Vec<f32> { self.storage.to_f32() }
    fn set_data_f32(&mut self, data: Vec<f32>) { self.storage = TensorStorage::from_f32(self.dtype, data); self.host_valid = true; }
    fn mark_host_stale(&mut self) { self.host_valid = false; }
    fn data_len(&self) -> usize { self.storage.len() }
    fn memory_bytes(&self) -> usize { self.storage.bytes() }

    fn show(&self) -> String {
        format!(
            "tensor(shape={:?}, dtype={}, device={}, bytes={})",
            self.shape,
            self.dtype.name(),
            self.device.name(),
            self.memory_bytes()
        )
    }
}

#[derive(Debug, Clone)]
enum Value {
    Number(f64), Text(String), Boolean(bool),
    List(Vec<Value>), Object(HashMap<String, Value>), Tensor(TensorRef), Null
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Object(a), Self::Object(b)) => a == b,
            (Self::Tensor(a), Self::Tensor(b)) => a.borrow().id == b.borrow().id,
            (Self::Null, Self::Null) => true,
            _ => false,
        }
    }
}

impl Value {
    fn truthy(&self) -> bool {
        match self {
            Self::Boolean(v) => *v,
            Self::Number(v) => *v != 0.0,
            Self::Text(v) => !v.is_empty(),
            Self::List(v) => !v.is_empty(),
            Self::Object(v) => !v.is_empty(),
            Self::Tensor(v) => v.borrow().data_len() > 0,
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
            Self::Tensor(v) => v.borrow().show(),
            Self::Null => "null".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Number, Text, Boolean, List, Object, Tensor, Null, Any,
}

impl Type {
    fn name(self) -> &'static str {
        match self {
            Self::Number => "Number",
            Self::Text => "Text",
            Self::Boolean => "Boolean",
            Self::List => "List",
            Self::Object => "Object",
            Self::Tensor => "Tensor",
            Self::Null => "Null",
            Self::Any => "Any",
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
    While(Expr, Vec<Stmt>),
    For(String, Expr, Vec<Stmt>),
    Use(String), Function(String, Vec<String>, Vec<Stmt>),
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
            Token::Use => {
                self.advance();
                match self.advance() {
                    Token::Text(path) => Ok(Stmt::Use(path)),
                    x => Err(format!("Nano: caminho do módulo esperado, encontrado {:?}", x)),
                }
            }
            Token::Function => self.function(),
            Token::Print => { self.advance(); Ok(Stmt::Print(self.expression()?)) }
            Token::If => self.if_stmt(),
            Token::While => self.while_stmt(),
            Token::For => self.for_stmt(),
            Token::Return => { self.advance(); Ok(Stmt::Return(self.expression()?)) }
            Token::Ident(name) => {
                let name = name.clone();
                if matches!(self.tokens.get(self.pos + 1), Some(Token::Equal)) {
                    self.advance();
                    self.advance();
                    Ok(Stmt::Assign(name, self.expression()?))
                } else {
                    Ok(Stmt::Expr(self.expression()?))
                }
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
        let no = if matches!(self.peek(), Token::Else) {
            self.advance();
            self.block()?
        } else {
            Vec::new()
        };
        Ok(Stmt::If(cond, yes, no))
    }
    fn while_stmt(&mut self) -> Result<Stmt, String> {
        self.advance();
        let cond = self.expression()?;
        Ok(Stmt::While(cond, self.block()?))
    }

    fn for_stmt(&mut self) -> Result<Stmt, String> {
        self.advance();
        let name = match self.advance() {
            Token::Ident(v) => v,
            x => return Err(format!("Nano: variável do for esperada, encontrado {:?}", x)),
        };
        self.expect(Token::In)?;
        let iterable = self.expression()?;
        Ok(Stmt::For(name, iterable, self.block()?))
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

struct Semantic {
    vars: HashMap<String, Type>,
    functions: HashMap<String, (usize, Type)>,
}

impl Semantic {
    fn new() -> Self {
        Self { vars: HashMap::new(), functions: HashMap::new() }
    }

    fn check(&mut self, program: &[Stmt]) -> Result<(), String> {
        for stmt in program {
            if let Stmt::Function(name, params, _) = stmt {
                self.functions.entry(name.clone()).or_insert((params.len(), Type::Any));
            }
        }

        for stmt in program {
            if let Stmt::Function(name, params, body) = stmt {
                self.check_function(name, params, body)?;
            }
        }

        for stmt in program {
            if !matches!(stmt, Stmt::Function(_, _, _)) {
                self.check_stmt(stmt)?;
            }
        }
        Ok(())
    }

    fn check_function(&mut self, name: &str, params: &[String], body: &[Stmt]) -> Result<(), String> {
        let saved = self.vars.clone();
        for param in params {
            self.vars.insert(param.clone(), Type::Any);
        }

        let mut return_type = Type::Null;
        let mut saw_return = false;
        for stmt in body {
            self.check_stmt_with_return(stmt, &mut return_type, &mut saw_return)?;
        }

        let inferred = if saw_return { return_type } else { Type::Null };
        if let Some((_, stored)) = self.functions.get_mut(name) {
            *stored = inferred;
        }

        self.vars = saved;
        Ok(())
    }

    fn check_stmt_with_return(
        &mut self,
        stmt: &Stmt,
        return_type: &mut Type,
        saw_return: &mut bool,
    ) -> Result<(), String> {
        match stmt {
            Stmt::Return(expr) => {
                let ty = self.expr_type(expr)?;
                if !*saw_return {
                    *return_type = ty;
                    *saw_return = true;
                } else {
                    *return_type = Self::merge(*return_type, ty, "retorno de função")?;
                }
                Ok(())
            }
            Stmt::If(cond, yes, no) => {
                let cond_type = self.expr_type(cond)?;
                self.expect_type(cond_type, &[Type::Boolean, Type::Number, Type::Text, Type::List, Type::Object, Type::Null, Type::Any], "condição")?;
                for s in yes { self.check_stmt_with_return(s, return_type, saw_return)?; }
                for s in no { self.check_stmt_with_return(s, return_type, saw_return)?; }
                Ok(())
            }
            Stmt::While(cond, body) => {
                let cond_type = self.expr_type(cond)?;
                self.expect_type(cond_type, &[Type::Boolean, Type::Number, Type::Text, Type::List, Type::Object, Type::Null, Type::Any], "condição")?;
                for s in body { self.check_stmt_with_return(s, return_type, saw_return)?; }
                Ok(())
            }
            Stmt::For(name, iterable, body) => {
                self.expr_type(iterable)?;
                let previous = self.vars.insert(name.clone(), Type::Any);
                for s in body { self.check_stmt_with_return(s, return_type, saw_return)?; }
                match previous {
                    Some(ty) => { self.vars.insert(name.clone(), ty); }
                    None => { self.vars.remove(name); }
                }
                Ok(())
            }
            _ => self.check_stmt(stmt),
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Assign(name, expr) => {
                let ty = self.expr_type(expr)?;
                if let Some(previous) = self.vars.get(name).copied() {
                    let merged = Self::merge(previous, ty, &format!("variável '{name}'"))?;
                    self.vars.insert(name.clone(), merged);
                } else {
                    self.vars.insert(name.clone(), ty);
                }
                Ok(())
            }
            Stmt::Print(expr) | Stmt::Expr(expr) => { self.expr_type(expr)?; Ok(()) }
            Stmt::If(cond, yes, no) => {
                self.expr_type(cond)?;
                for s in yes { self.check_stmt(s)?; }
                for s in no { self.check_stmt(s)?; }
                Ok(())
            }
            Stmt::While(cond, body) => {
                self.expr_type(cond)?;
                for s in body { self.check_stmt(s)?; }
                Ok(())
            }
            Stmt::For(name, iterable, body) => {
                self.expr_type(iterable)?;
                let previous = self.vars.insert(name.clone(), Type::Any);
                for s in body { self.check_stmt(s)?; }
                match previous {
                    Some(ty) => { self.vars.insert(name.clone(), ty); }
                    None => { self.vars.remove(name); }
                }
                Ok(())
            }
            Stmt::Use(_) => Ok(()),
            Stmt::Function(_, _, _) => Ok(()),
            Stmt::Return(expr) => { self.expr_type(expr)?; Ok(()) }
        }
    }

    fn expr_type(&mut self, expr: &Expr) -> Result<Type, String> {
        match expr {
            Expr::Value(v) => Ok(match v {
                Value::Number(_) => Type::Number,
                Value::Text(_) => Type::Text,
                Value::Boolean(_) => Type::Boolean,
                Value::List(_) => Type::List,
                Value::Object(_) => Type::Object,
                Value::Tensor(_) => Type::Tensor,
                Value::Null => Type::Null,
            }),
            Expr::Var(name) => self.vars.get(name).copied().ok_or_else(|| format!("Nano: variável '{name}' não definida")),
            Expr::List(items) => {
                for item in items { self.expr_type(item)?; }
                Ok(Type::List)
            }
            Expr::Object(fields) => {
                for (_, value) in fields { self.expr_type(value)?; }
                Ok(Type::Object)
            }
            Expr::Binary(a, op, b) => {
                let left = self.expr_type(a)?;
                let right = self.expr_type(b)?;
                match op {
                    Op::Eq | Op::Ne => Ok(Type::Boolean),
                    Op::Add => {
                        if left == Type::Any || right == Type::Any { return Ok(Type::Any); }
                        match (left, right) {
                            (Type::Number, Type::Number) => Ok(Type::Number),
                            (Type::Text, _) | (_, Type::Text) => Ok(Type::Text),
                            (Type::List, Type::List) => Ok(Type::List),
                            _ => Err(format!("Nano: '+' não aceita {} + {}", left.name(), right.name())),
                        }
                    }
                    Op::Sub | Op::Mul | Op::Div => {
                        Self::numeric_result(left, right, "operação aritmética")
                    }
                    Op::Gt | Op::Ge | Op::Lt | Op::Le => {
                        self.expect_numeric(left, right, "comparação")?;
                        Ok(Type::Boolean)
                    }
                }
            }
            Expr::Field(target, name) => {
                let target_type = self.expr_type(target)?;
                match target_type {
                    Type::Object | Type::Any => Ok(Type::Any),
                    _ => Err(format!("Nano: '.' requer Object, recebido {}", target_type.name())),
                }
                .map_err(|e| if e.contains("Object") && !name.is_empty() { e } else { e })
            }
            Expr::Index(target, index) => {
                let target_type = self.expr_type(target)?;
                let index_type = self.expr_type(index)?;
                match target_type {
                    Type::Any => Ok(Type::Any),
                    Type::List => {
                        if index_type == Type::Number || index_type == Type::Any {
                            Ok(Type::Any)
                        } else {
                            Err(format!("Nano: lista requer índice Number, recebido {}", index_type.name()))
                        }
                    }
                    Type::Object => {
                        if index_type == Type::Text || index_type == Type::Any {
                            Ok(Type::Any)
                        } else {
                            Err(format!("Nano: Object requer chave Text, recebido {}", index_type.name()))
                        }
                    }
                    Type::Tensor => {
                        if index_type == Type::Number || index_type == Type::Any {
                            Ok(Type::Any)
                        } else {
                            Err(format!("Nano: Tensor requer índice Number, recebido {}", index_type.name()))
                        }
                    }
                    _ => Err(format!("Nano: indexação requer List, Object ou Tensor, recebido {}", target_type.name())),
                }
            }
            Expr::Call(name, args) => {
                for arg in args { self.expr_type(arg)?; }
                if name == "len" {
                    if args.len() != 1 { return Err("Nano: len() recebe 1 argumento".into()); }
                    return Ok(Type::Number);
                }
                if name == "to_text" {
                    if args.len() != 1 { return Err("Nano: to_text() recebe 1 argumento".into()); }
                    return Ok(Type::Text);
                }
                if name == "to_number" {
                    if args.len() != 1 { return Err("Nano: to_number() recebe 1 argumento".into()); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Text && ty != Type::Any {
                        return Err(format!("Nano: to_number() requer Text, recebido {}", ty.name()));
                    }
                    return Ok(Type::Number);
                }
                if matches!(name, "upper" | "lower" | "trim") {
                    if args.len() != 1 { return Err(format!("Nano: {name}() recebe 1 Text")); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Text && ty != Type::Any {
                        return Err(format!("Nano: {name}() requer Text, recebido {}", ty.name()));
                    }
                    return Ok(Type::Text);
                }
                if matches!(name, "contains" | "starts_with" | "ends_with") {
                    if args.len() != 2 { return Err(format!("Nano: {name}() recebe 2 Text")); }
                    let a = self.expr_type(&args[0])?;
                    let b = self.expr_type(&args[1])?;
                    if (a != Type::Text && a != Type::Any) || (b != Type::Text && b != Type::Any) {
                        return Err(format!("Nano: {name}() requer Text, Text"));
                    }
                    return Ok(Type::Boolean);
                }
                if name == "replace" {
                    if args.len() != 3 { return Err("Nano: replace() recebe texto, antigo e novo".into()); }
                    for arg in args {
                        let ty = self.expr_type(arg)?;
                        if ty != Type::Text && ty != Type::Any {
                            return Err("Nano: replace() requer Text, Text, Text".into());
                        }
                    }
                    return Ok(Type::Text);
                }
                if name == "substring" || name == "char_at" {
                    let expected = if name == "char_at" { 2 } else { 3 };
                    if args.len() != expected { return Err(format!("Nano: {name}() recebe {expected} argumentos")); }
                    let text = self.expr_type(&args[0])?;
                    if text != Type::Text && text != Type::Any {
                        return Err(format!("Nano: {name}() requer Text como primeiro argumento"));
                    }
                    for arg in &args[1..] {
                        let ty = self.expr_type(arg)?;
                        if ty != Type::Number && ty != Type::Any {
                            return Err(format!("Nano: {name}() requer índices Number"));
                        }
                    }
                    return Ok(Type::Text);
                }
                if name == "split" {
                    if args.len() != 2 { return Err("Nano: split() recebe texto e separador".into()); }
                    for arg in args {
                        let ty = self.expr_type(arg)?;
                        if ty != Type::Text && ty != Type::Any {
                            return Err("Nano: split() requer Text, Text".into());
                        }
                    }
                    return Ok(Type::List);
                }
                if name == "join" {
                    if args.len() != 2 { return Err("Nano: join() recebe lista e separador".into()); }
                    let list = self.expr_type(&args[0])?;
                    let sep = self.expr_type(&args[1])?;
                    if (list != Type::List && list != Type::Any) || (sep != Type::Text && sep != Type::Any) {
                        return Err("Nano: join() requer List, Text".into());
                    }
                    return Ok(Type::Text);
                }
                if name == "append" {
                    if args.len() != 2 { return Err("Nano: append() recebe lista e valor".into()); }
                    let list = self.expr_type(&args[0])?;
                    if list != Type::List && list != Type::Any {
                        return Err("Nano: append() requer List".into());
                    }
                    return Ok(Type::List);
                }
                if name == "backend" {
                    if !args.is_empty() { return Err("Nano: backend() não recebe argumentos".into()); }
                    return Ok(Type::Text);
                }
                if name == "dtype" {
                    if args.len() != 1 { return Err("Nano: dtype() recebe 1 tensor".into()); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Tensor && ty != Type::Any {
                        return Err(format!("Nano: dtype() requer Tensor, recebido {}", ty.name()));
                    }
                    return Ok(Type::Text);
                }
                if name == "memory_bytes" {
                    if args.len() != 1 { return Err("Nano: memory_bytes() recebe 1 tensor".into()); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Tensor && ty != Type::Any {
                        return Err(format!("Nano: memory_bytes() requer Tensor, recebido {}", ty.name()));
                    }
                    return Ok(Type::Number);
                }
                if name == "cast" {
                    if args.len() != 2 { return Err("Nano: cast() recebe tensor e dtype".into()); }
                    let ty = self.expr_type(&args[0])?;
                    let dtype_ty = self.expr_type(&args[1])?;
                    if (ty != Type::Tensor && ty != Type::Any) || dtype_ty != Type::Text {
                        return Err("Nano: cast() requer Tensor e Text".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "device" {
                    if args.len() != 1 { return Err("Nano: device() recebe 1 tensor".into()); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Tensor && ty != Type::Any {
                        return Err(format!("Nano: device() requer Tensor, recebido {}", ty.name()));
                    }
                    return Ok(Type::Text);
                }
                if name == "tensor" {
                    if args.len() != 2 { return Err("Nano: tensor() recebe dados e shape".into()); }
                    let data_type = self.expr_type(&args[0])?;
                    let shape_type = self.expr_type(&args[1])?;
                    if data_type != Type::List || shape_type != Type::List {
                        return Err("Nano: tensor() requer lista de dados e lista de shape".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "zeros" {
                    if args.len() != 1 { return Err("Nano: zeros() recebe shape".into()); }
                    if self.expr_type(&args[0])? != Type::List {
                        return Err("Nano: zeros() requer lista de shape".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "shape" {
                    if args.len() != 1 { return Err("Nano: shape() recebe 1 tensor".into()); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Tensor && ty != Type::Any {
                        return Err(format!("Nano: shape() requer Tensor, recebido {}", ty.name()));
                    }
                    return Ok(Type::List);
                }
                if name == "range" {
                    if args.len() != 1 { return Err("Nano: range() recebe 1 argumento".into()); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Number && ty != Type::Any {
                        return Err(format!("Nano: range() requer Number, recebido {}", ty.name()));
                    }
                    return Ok(Type::List);
                }
                if name == "parameter" {
                    if args.len() != 2 { return Err("Nano: parameter() recebe dados e shape".into()); }
                    let data_type = self.expr_type(&args[0])?;
                    let shape_type = self.expr_type(&args[1])?;
                    if data_type != Type::List || shape_type != Type::List {
                        return Err("Nano: parameter() requer listas de dados e shape".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "matmul" {
                    if args.len() != 2 { return Err("Nano: matmul() recebe 2 tensores".into()); }
                    let left = self.expr_type(&args[0])?;
                    let right = self.expr_type(&args[1])?;
                    if (left != Type::Tensor && left != Type::Any) || (right != Type::Tensor && right != Type::Any) {
                        return Err("Nano: matmul() requer Tensor, Tensor".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "sum" || name == "mean" {
                    if args.len() != 1 { return Err(format!("Nano: {name}() recebe 1 tensor")); }
                    let ty = self.expr_type(&args[0])?;
                    if ty != Type::Tensor && ty != Type::Any {
                        return Err(format!("Nano: {name}() requer Tensor, recebido {}", ty.name()));
                    }
                    return Ok(Type::Tensor);
                }
                if name == "grad" {
                    if args.len() != 2 { return Err("Nano: grad() recebe loss e parâmetro".into()); }
                    let loss = self.expr_type(&args[0])?;
                    let param = self.expr_type(&args[1])?;
                    if (loss != Type::Tensor && loss != Type::Any) || (param != Type::Tensor && param != Type::Any) {
                        return Err("Nano: grad() requer Tensor, Tensor".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "adam" {
                    if args.len() != 3 { return Err("Nano: adam() recebe parâmetro, gradiente e taxa".into()); }
                    let param = self.expr_type(&args[0])?;
                    let grad = self.expr_type(&args[1])?;
                    let rate = self.expr_type(&args[2])?;
                    if (param != Type::Tensor && param != Type::Any) || (grad != Type::Tensor && grad != Type::Any) || (rate != Type::Number && rate != Type::Any) {
                        return Err("Nano: adam() requer Tensor, Tensor, Number".into());
                    }
                    return Ok(Type::Tensor);
                }
                if name == "step" {
                    if args.len() != 3 { return Err("Nano: step() recebe parâmetro, gradiente e taxa".into()); }
                    let param = self.expr_type(&args[0])?;
                    let grad = self.expr_type(&args[1])?;
                    let rate = self.expr_type(&args[2])?;
                    if (param != Type::Tensor && param != Type::Any) || (grad != Type::Tensor && grad != Type::Any) || (rate != Type::Number && rate != Type::Any) {
                        return Err("Nano: step() requer Tensor, Tensor, Number".into());
                    }
                    return Ok(Type::Tensor);
                }
                match self.functions.get(name) {
                    Some((expected, return_type)) if *expected != args.len() => {
                        Err(format!("Nano: '{name}' esperava {} argumentos", expected))
                    }
                    Some((_, return_type)) => Ok(*return_type),
                    None => Ok(Type::Any),
                }
            }
        }
    }

    fn merge(a: Type, b: Type, label: &str) -> Result<Type, String> {
        if a == b { Ok(a) }
        else if a == Type::Any || b == Type::Any { Ok(Type::Any) }
        else { Err(format!("Nano: tipo incompatível em {label}: {} e {}", a.name(), b.name())) }
    }

    fn numeric_result(a: Type, b: Type, label: &str) -> Result<Type, String> {
        if a == Type::Number && b == Type::Number { Ok(Type::Number) }
        else if a == Type::Any || b == Type::Any { Ok(Type::Any) }
        else { Err(format!("Nano: {label} requer Number, recebido {} e {}", a.name(), b.name())) }
    }

    fn expect_numeric(&self, a: Type, b: Type, label: &str) -> Result<(), String> {
        if (a == Type::Number || a == Type::Any) && (b == Type::Number || b == Type::Any) {
            Ok(())
        } else {
            Err(format!("Nano: {label} requer Number, recebido {} e {}", a.name(), b.name()))
        }
    }

    fn expect_type(&self, _actual: Type, _allowed: &[Type], _label: &str) -> Result<(), String> {
        Ok(())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliCommand {
    Run,
    Check,
    BuildNative,
}

fn parse_cli(
    args: &[String],
) -> Result<(CliCommand, String, Option<backend::BackendKind>, Option<DType>, Option<String>), String> {
    let command = match args.get(1).map(String::as_str) {
        Some("run") => CliCommand::Run,
        Some("check") => CliCommand::Check,
        Some("build") => CliCommand::BuildNative,
        _ => return Err("uso: nano run|check|build --native [--output arquivo] [--backend cpu|gpu|npu] [--dtype f32|f16|bf16] [arquivo.nano]".into()),
    };

    let mut path = None;
    let mut selected_backend = None;
    let mut selected_dtype = None;
    let mut output = None;
    let mut native_requested = false;
    let mut index = 2;

    while index < args.len() {
        match args[index].as_str() {
            "--native" => native_requested = true,
            "-o" | "--output" => {
                index += 1;
                output = Some(
                    args.get(index)
                        .ok_or_else(|| "Nano: --output requer um caminho".to_string())?
                        .clone()
                );
            }
            value if value.starts_with("--output=") => {
                output = Some(value.trim_start_matches("--output=").to_string());
            }
            "--backend" => {
                index += 1;
                let value = args.get(index)
                    .ok_or_else(|| "Nano: --backend requer cpu ou gpu".to_string())?;
                selected_backend = Some(
                    backend::BackendKind::parse(value).map_err(|e| e.to_string())?
                );
            }
            value if value.starts_with("--backend=") => {
                selected_backend = Some(
                    backend::BackendKind::parse(value.trim_start_matches("--backend="))
                        .map_err(|e| e.to_string())?
                );
            }
            "--dtype" => {
                index += 1;
                let value = args.get(index)
                    .ok_or_else(|| "Nano: --dtype requer f32, f16 ou bf16".to_string())?;
                selected_dtype = Some(DType::parse(value)?);
            }
            value if value.starts_with("--dtype=") => {
                selected_dtype = Some(DType::parse(value.trim_start_matches("--dtype="))?);
            }
            value if value.starts_with('-') => {
                return Err(format!("Nano: opção desconhecida '{value}'"));
            }
            value => {
                if path.replace(value.to_string()).is_some() {
                    return Err("Nano: apenas um arquivo .nano pode ser informado".into());
                }
            }
        }
        index += 1;
    }

    if matches!(command, CliCommand::BuildNative) && !native_requested {
        return Err("Nano: use 'nano build --native [arquivo.nano]' para o backend CPU nativo".into());
    }
    if !matches!(command, CliCommand::BuildNative) && native_requested {
        return Err("Nano: --native só é válido com 'build'".into());
    }

    if matches!(command, CliCommand::BuildNative) && output.is_none() {
        let source_path = path.as_deref().unwrap_or("main.nano");
        let stem = Path::new(source_path)
            .file_stem()
            .and_then(|v| v.to_str())
            .unwrap_or("nano_app");
        output = Some(stem.to_string());
    }

    Ok((
        command,
        path.unwrap_or_else(|| "main.nano".into()),
        selected_backend,
        selected_dtype,
        output,
    ))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (command, path, cli_backend, cli_dtype, output) = match parse_cli(&args) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("Nano 0.9 — {e}");
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

    let mut semantic = Semantic::new();
    if let Err(e) = semantic.check(&program) {
        eprintln!("{e}");
        process::exit(1);
    }

    let mut compiler = ir::Compiler::new();
    let ir_program = match compiler.compile(&program) {
        Ok(program) => program,
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    };

    let mut optimizer = ir::Optimizer::new();
    let ir_program = optimizer.optimize_program(ir_program);

    if command == CliCommand::Check {
        return;
    }

    if command == CliCommand::BuildNative {
        let output_path = output
            .expect("build --native sempre define caminho de saída");
        let output = Path::new(&output_path);
        if let Err(e) = native::build(&ir_program, output) {
            eprintln!("{e}");
            process::exit(1);
        }
        println!("Nano: executável nativo criado em '{}'", output.display());
        return;
    }

    let backend_kind = match cli_backend {
        Some(kind) => kind,
        None => match env::var("NANO_BACKEND") {
        Ok(value) => match backend::BackendKind::parse(&value) {
            Ok(kind) => kind,
            Err(e) => {
                eprintln!("Nano: {e}");
                process::exit(1);
            }
        },
            Err(_) => backend::BackendKind::Cpu,
        },
    };

    let dtype = match cli_dtype {
        Some(dtype) => dtype,
        None => match env::var("NANO_DTYPE") {
            Ok(value) => match DType::parse(&value) {
                Ok(dtype) => dtype,
                Err(e) => { eprintln!("Nano: {e}"); process::exit(1); }
            },
            Err(_) => DType::F32,
        },
    };

    let mut runtime = match ir::IrRuntime::with_backend_and_dtype(backend_kind, dtype) {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("{e}");
            process::exit(1);
        }
    };

    if let Err(e) = runtime.run(&ir_program) {
        eprintln!("{e}");
        process::exit(1);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_to_run_and_main() {
        let args = vec!["nano".into(), "run".into()];
        let (command, path, backend, dtype, output) = parse_cli(&args).unwrap();
        assert_eq!(command, CliCommand::Run);
        assert_eq!(path, "main.nano");
        assert_eq!(backend, None);
        assert_eq!(dtype, None);
        assert_eq!(output, None);
    }

    #[test]
    fn cli_supports_check_and_backend() {
        let args = vec![
            "nano".into(),
            "check".into(),
            "--backend=cpu".into(),
            "examples/tensor.nano".into(),
        ];
        let (command, path, backend, dtype, output) = parse_cli(&args).unwrap();
        assert_eq!(command, CliCommand::Check);
        assert_eq!(path, "examples/tensor.nano");
        assert_eq!(backend, Some(backend::BackendKind::Cpu));
        assert_eq!(dtype, None);
        assert_eq!(output, None);
    }

    #[test]
    fn cli_rejects_unknown_options() {
        let args = vec!["nano".into(), "run".into(), "--wat".into()];
        assert!(parse_cli(&args).is_err());
    }

    #[test]
    fn cli_supports_native_build() {
        let args = vec![
            "nano".into(),
            "build".into(),
            "--native".into(),
            "--output=app".into(),
            "program.nano".into(),
        ];
        let (command, path, backend, dtype, output) = parse_cli(&args).unwrap();
        assert_eq!(command, CliCommand::BuildNative);
        assert_eq!(path, "program.nano");
        assert_eq!(backend, None);
        assert_eq!(dtype, None);
        assert_eq!(output, Some("app".into()));
    }

}
