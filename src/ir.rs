use std::collections::HashMap;
use std::fs;

use super::{Expr, Lexer, Op, Parser, Semantic, Stmt, Value};

#[derive(Debug, Clone)]
pub(crate) enum IrInst {
    Const(Value),
    Load(String),
    Store(String),
    Binary(Op),
    MakeList(usize),
    MakeObject(Vec<String>),
    Index,
    Field(String),
    Call(String, usize),
    Print,
    Pop,
    JumpIfFalse(usize),
    Jump(usize),
    Return,
    Use(String),
}

#[derive(Debug, Clone)]
pub(crate) struct IrFunction {
    pub(crate) params: Vec<String>,
    pub(crate) code: Vec<IrInst>,
}

#[derive(Debug, Clone)]
pub(crate) struct IrProgram {
    pub(crate) code: Vec<IrInst>,
    pub(crate) functions: HashMap<String, IrFunction>,
}

pub(crate) struct Compiler;

pub(crate) struct Optimizer;

impl Optimizer {
    pub(crate) fn new() -> Self { Self }

    pub(crate) fn optimize_program(&mut self, mut program: IrProgram) -> IrProgram {
        program.code = self.optimize_code(program.code);
        for function in program.functions.values_mut() {
            function.code = self.optimize_code(function.code.clone());
        }
        program
    }

    fn optimize_code(&self, code: Vec<IrInst>) -> Vec<IrInst> {
        let original_len = code.len();
        let mut out: Vec<IrInst> = Vec::with_capacity(original_len);
        let mut map: Vec<usize> = vec![0; original_len + 1];

        for (old_index, inst) in code.into_iter().enumerate() {
            let mut folded = false;

            if let IrInst::Binary(op) = &inst {
                if out.len() >= 2 {
                    let right = out[out.len() - 1].clone();
                    let left = out[out.len() - 2].clone();
                    if let (IrInst::Const(a), IrInst::Const(b)) = (left, right) {
                        if let Ok(value) = binary(a, *op, b) {
                            let new_index = out.len() - 2;
                            out.truncate(new_index);
                            out.push(IrInst::Const(value));
                            map[old_index.saturating_sub(2)] = new_index;
                            map[old_index.saturating_sub(1)] = new_index;
                            map[old_index] = new_index;
                            folded = true;
                        }
                    }
                }
            }

            if !folded {
                let new_index = out.len();
                map[old_index] = new_index;
                out.push(inst);
            }
        }

        map[original_len] = out.len();

        let output_len = out.len();
        for inst in &mut out {
            match inst {
                IrInst::Jump(target) | IrInst::JumpIfFalse(target) => {
                    *target = map.get(*target).copied().unwrap_or(output_len);
                }
                _ => {}
            }
        }

        out
    }
}

impl Compiler {
    pub(crate) fn new() -> Self { Self }

    pub(crate) fn compile(&mut self, program: &[Stmt]) -> Result<IrProgram, String> {
        let mut functions = HashMap::new();
        let mut code = Vec::new();

        for stmt in program {
            if let Stmt::Function(name, params, body) = stmt {
                let mut function_code = Vec::new();
                for stmt in body {
                    self.compile_stmt(stmt, &mut function_code)?;
                }
                if !matches!(function_code.last(), Some(IrInst::Return)) {
                    function_code.push(IrInst::Const(Value::Null));
                    function_code.push(IrInst::Return);
                }
                functions.insert(
                    name.clone(),
                    IrFunction { params: params.clone(), code: function_code },
                );
            }
        }

        for stmt in program {
            if !matches!(stmt, Stmt::Function(_, _, _)) {
                self.compile_stmt(stmt, &mut code)?;
            }
        }

        Ok(IrProgram { code, functions })
    }

    fn compile_stmt(&mut self, stmt: &Stmt, code: &mut Vec<IrInst>) -> Result<(), String> {
        match stmt {
            Stmt::Assign(name, expr) => {
                self.compile_expr(expr, code)?;
                code.push(IrInst::Store(name.clone()));
            }
            Stmt::Print(expr) => {
                self.compile_expr(expr, code)?;
                code.push(IrInst::Print);
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr, code)?;
                code.push(IrInst::Pop);
            }
            Stmt::Return(expr) => {
                self.compile_expr(expr, code)?;
                code.push(IrInst::Return);
            }
            Stmt::Use(path) => code.push(IrInst::Use(path.clone())),
            Stmt::Function(_, _, _) => {}
            Stmt::If(cond, yes, no) => {
                self.compile_expr(cond, code)?;
                let jump_false = code.len();
                code.push(IrInst::JumpIfFalse(usize::MAX));

                for stmt in yes {
                    self.compile_stmt(stmt, code)?;
                }

                let jump_end = code.len();
                code.push(IrInst::Jump(usize::MAX));
                let else_start = code.len();
                code[jump_false] = IrInst::JumpIfFalse(else_start);

                for stmt in no {
                    self.compile_stmt(stmt, code)?;
                }

                let end = code.len();
                code[jump_end] = IrInst::Jump(end);
            }
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &Expr, code: &mut Vec<IrInst>) -> Result<(), String> {
        match expr {
            Expr::Value(value) => code.push(IrInst::Const(value.clone())),
            Expr::Var(name) => code.push(IrInst::Load(name.clone())),
            Expr::List(items) => {
                for item in items {
                    self.compile_expr(item, code)?;
                }
                code.push(IrInst::MakeList(items.len()));
            }
            Expr::Object(fields) => {
                let mut keys = Vec::with_capacity(fields.len());
                for (key, value) in fields {
                    keys.push(key.clone());
                    self.compile_expr(value, code)?;
                }
                code.push(IrInst::MakeObject(keys));
            }
            Expr::Binary(left, op, right) => {
                self.compile_expr(left, code)?;
                self.compile_expr(right, code)?;
                code.push(IrInst::Binary(*op));
            }
            Expr::Call(name, args) => {
                for arg in args {
                    self.compile_expr(arg, code)?;
                }
                code.push(IrInst::Call(name.clone(), args.len()));
            }
            Expr::Index(target, index) => {
                self.compile_expr(target, code)?;
                self.compile_expr(index, code)?;
                code.push(IrInst::Index);
            }
            Expr::Field(target, name) => {
                self.compile_expr(target, code)?;
                code.push(IrInst::Field(name.clone()));
            }
        }
        Ok(())
    }
}

pub(crate) struct IrRuntime {
    vars: HashMap<String, Value>,
    functions: HashMap<String, IrFunction>,
}

impl IrRuntime {
    pub(crate) fn new() -> Self {
        Self { vars: HashMap::new(), functions: HashMap::new() }
    }

    pub(crate) fn run(&mut self, program: &IrProgram) -> Result<(), String> {
        self.functions.extend(program.functions.clone());
        let _ = self.execute_code(&program.code)?;
        Ok(())
    }

    fn execute_code(&mut self, code: &[IrInst]) -> Result<Option<Value>, String> {
        let mut stack: Vec<Value> = Vec::new();
        let mut ip = 0usize;

        while ip < code.len() {
            match &code[ip] {
                IrInst::Const(value) => stack.push(value.clone()),
                IrInst::Load(name) => {
                    let value = self.vars.get(name).cloned()
                        .ok_or_else(|| format!("Nano: variável '{name}' não definida"))?;
                    stack.push(value);
                }
                IrInst::Store(name) => {
                    let value = stack.pop().ok_or_else(|| "Nano IR: stack vazia em Store".to_string())?;
                    self.vars.insert(name.clone(), value);
                }
                IrInst::Binary(op) => {
                    let right = stack.pop().ok_or_else(|| "Nano IR: stack vazia no operando direito".to_string())?;
                    let left = stack.pop().ok_or_else(|| "Nano IR: stack vazia no operando esquerdo".to_string())?;
                    stack.push(binary(left, *op, right)?);
                }
                IrInst::MakeList(count) => {
                    let mut values = pop_n(&mut stack, *count)?;
                    values.reverse();
                    stack.push(Value::List(values));
                }
                IrInst::MakeObject(keys) => {
                    let mut values = pop_n(&mut stack, keys.len())?;
                    values.reverse();
                    let mut object = HashMap::new();
                    for (key, value) in keys.iter().cloned().zip(values) {
                        object.insert(key, value);
                    }
                    stack.push(Value::Object(object));
                }
                IrInst::Index => {
                    let index = stack.pop().ok_or_else(|| "Nano IR: stack vazia no índice".to_string())?;
                    let target = stack.pop().ok_or_else(|| "Nano IR: stack vazia no alvo".to_string())?;
                    stack.push(index_value(target, index)?);
                }
                IrInst::Field(name) => {
                    let target = stack.pop().ok_or_else(|| "Nano IR: stack vazia no campo".to_string())?;
                    let value = match target {
                        Value::Object(values) => values.get(name).cloned()
                            .ok_or_else(|| format!("Nano: campo '{name}' não existe"))?,
                        _ => return Err("Nano: '.' requer um objeto".into()),
                    };
                    stack.push(value);
                }
                IrInst::Call(name, count) => {
                    let mut args = pop_n(&mut stack, *count)?;
                    args.reverse();
                    let value = self.call(name, args)?;
                    stack.push(value);
                }
                IrInst::Print => {
                    let value = stack.pop().ok_or_else(|| "Nano IR: stack vazia em Print".to_string())?;
                    println!("{}", value.show());
                }
                IrInst::Pop => {
                    stack.pop().ok_or_else(|| "Nano IR: stack vazia em Pop".to_string())?;
                }
                IrInst::JumpIfFalse(target) => {
                    let condition = stack.pop().ok_or_else(|| "Nano IR: stack vazia em JumpIfFalse".to_string())?;
                    if !condition.truthy() {
                        ip = *target;
                        continue;
                    }
                }
                IrInst::Jump(target) => {
                    ip = *target;
                    continue;
                }
                IrInst::Return => {
                    return Ok(Some(stack.pop().unwrap_or(Value::Null)));
                }
                IrInst::Use(path) => self.load_module(path)?,
            }
            ip += 1;
        }

        Ok(None)
    }

    fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        if name == "len" {
            if args.len() != 1 {
                return Err("Nano: len() recebe 1 argumento".into());
            }
            return match &args[0] {
                Value::Text(v) => Ok(Value::Number(v.chars().count() as f64)),
                Value::List(v) => Ok(Value::Number(v.len() as f64)),
                Value::Object(v) => Ok(Value::Number(v.len() as f64)),
                Value::Tensor(v) => Ok(Value::Number(v.data.len() as f64)),
                _ => Err("Nano: len() requer texto, lista, objeto ou tensor".into()),
            };
        }

        if name == "tensor" {
            if args.len() != 2 {
                return Err("Nano: tensor() recebe dados e shape".into());
            }
            let data = list_numbers(&args[0], "dados")?;
            let shape = list_shape(&args[1])?;
            return Ok(Value::Tensor(super::Tensor::new(data, shape)?));
        }

        if name == "zeros" {
            if args.len() != 1 {
                return Err("Nano: zeros() recebe shape".into());
            }
            let shape = list_shape(&args[0])?;
            let size = shape.iter().copied().product::<usize>();
            return Ok(Value::Tensor(super::Tensor::new(vec![0.0; size], shape)?));
        }

        if name == "shape" {
            if args.len() != 1 {
                return Err("Nano: shape() recebe 1 tensor".into());
            }
            return match &args[0] {
                Value::Tensor(t) => Ok(Value::List(
                    t.shape.iter().map(|v| Value::Number(*v as f64)).collect()
                )),
                _ => Err("Nano: shape() requer Tensor".into()),
            };
        }

        if name == "matmul" {
            if args.len() != 2 {
                return Err("Nano: matmul() recebe 2 tensores".into());
            }
            return matmul_values(&args[0], &args[1]);
        }

        let function = self.functions.get(name).cloned()
            .ok_or_else(|| format!("Nano: função '{name}' não definida"))?;

        if function.params.len() != args.len() {
            return Err(format!("Nano: '{name}' esperava {} argumentos", function.params.len()));
        }

        let saved = self.vars.clone();
        for (param, value) in function.params.iter().zip(args) {
            self.vars.insert(param.clone(), value);
        }

        let result = self.execute_code(&function.code)?.unwrap_or(Value::Null);
        self.vars = saved;
        Ok(result)
    }

    fn load_module(&mut self, path: &str) -> Result<(), String> {
        let source = fs::read_to_string(path)
            .map_err(|e| format!("Nano: não foi possível carregar módulo '{path}': {e}"))?;
        let tokens = Lexer::new(&source).lex()?;
        let program = Parser::new(tokens).program()?;

        let mut semantic = Semantic::new();
        semantic.check(&program)?;

        let mut compiler = Compiler::new();
        let module = compiler.compile(&program)?;
        self.functions.extend(module.functions);

        let _ = self.execute_code(&module.code)?;
        Ok(())
    }
}

fn pop_n(stack: &mut Vec<Value>, count: usize) -> Result<Vec<Value>, String> {
    if stack.len() < count {
        return Err("Nano IR: stack insuficiente".into());
    }
    let start = stack.len() - count;
    Ok(stack.drain(start..).collect())
}

fn binary(a: Value, op: Op, b: Value) -> Result<Value, String> {
    match op {
        Op::Add => match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x + y)),
            (Value::Text(x), Value::Text(y)) => Ok(Value::Text(x + &y)),
            (Value::Text(x), y) => Ok(Value::Text(x + &y.show())),
            (x, Value::Text(y)) => Ok(Value::Text(x.show() + &y)),
            (Value::List(mut x), Value::List(y)) => { x.extend(y); Ok(Value::List(x)) },
            _ => Err("Nano: '+' requer números, texto ou listas compatíveis".into()),
        },
        Op::Sub => num(a, b, |x, y| x - y),
        Op::Mul => num(a, b, |x, y| x * y),
        Op::Div => num(a, b, |x, y| x / y),
        Op::Eq => Ok(Value::Boolean(a == b)),
        Op::Ne => Ok(Value::Boolean(a != b)),
        Op::Gt => cmp(a, b, |x, y| x > y),
        Op::Ge => cmp(a, b, |x, y| x >= y),
        Op::Lt => cmp(a, b, |x, y| x < y),
        Op::Le => cmp(a, b, |x, y| x <= y),
    }
}

fn index_value(target: Value, index: Value) -> Result<Value, String> {
    match (target, index) {
        (Value::List(values), Value::Number(n)) => {
            if n < 0.0 || n.fract() != 0.0 {
                return Err("Nano: índice deve ser um número inteiro".into());
            }
            values.get(n as usize).cloned()
                .ok_or_else(|| "Nano: índice fora do limite".into())
        }
        (Value::Object(values), Value::Text(key)) => values.get(&key).cloned()
            .ok_or_else(|| format!("Nano: chave '{key}' não existe")),
        _ => Err("Nano: indexação requer lista[número] ou objeto[texto]".into()),
    }
}

fn num(a: Value, b: Value, f: fn(f64, f64) -> f64) -> Result<Value, String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Number(f(x, y))),
        _ => Err("Nano: operação requer números".into()),
    }
}

fn cmp(a: Value, b: Value, f: fn(f64, f64) -> bool) -> Result<Value, String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Boolean(f(x, y))),
        _ => Err("Nano: comparação requer números".into()),
    }
}


fn list_numbers(value: &Value, label: &str) -> Result<Vec<f32>, String> {
    match value {
        Value::List(items) => items.iter().map(|item| match item {
            Value::Number(v) => Ok(*v as f32),
            _ => Err(format!("Nano: tensor() requer Number nos {label}")),
        }).collect(),
        _ => Err(format!("Nano: tensor() requer lista de {label}")),
    }
}

fn list_shape(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::List(items) => items.iter().map(|item| match item {
            Value::Number(v) if *v >= 0.0 && v.fract() == 0.0 => Ok(*v as usize),
            _ => Err("Nano: shape deve ser uma lista de números inteiros".into()),
        }).collect(),
        _ => Err("Nano: shape deve ser uma lista".into()),
    }
}

fn matmul_values(a: &Value, b: &Value) -> Result<Value, String> {
    let (left, right) = match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => (a, b),
        _ => return Err("Nano: matmul() requer Tensor, Tensor".into()),
    };

    if left.shape.len() != 2 || right.shape.len() != 2 {
        return Err("Nano: matmul() nesta versão requer tensores 2D".into());
    }

    let (m, k) = (left.shape[0], left.shape[1]);
    let (k2, n) = (right.shape[0], right.shape[1]);
    if k != k2 {
        return Err(format!("Nano: matmul() incompatível: {}x{} com {}x{}", m, k, k2, n));
    }

    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for x in 0..k {
                sum += left.data[i * k + x] * right.data[x * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    Ok(Value::Tensor(super::Tensor::new(out, vec![m, n])?))
}
