use std::collections::HashMap;
use std::fs;

use super::{backend, Expr, Lexer, Op, Parser, Semantic, Stmt, TensorOp, TensorOpKind, TensorRef, Value};
use backend::{ElementwiseOp, TensorBackend};

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
    IterInit,
    IterNext(String, usize),
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

#[derive(Debug, Clone)]
struct IterState {
    values: Vec<Value>,
    index: usize,
}

#[derive(Debug, Clone)]
struct AdamState {
    step: u64,
    m: Vec<f32>,
    v: Vec<f32>,
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
                IrInst::IterNext(_, target) => {
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
            Stmt::While(cond, body) => {
                let start = code.len();
                self.compile_expr(cond, code)?;
                let exit = code.len();
                code.push(IrInst::JumpIfFalse(usize::MAX));
                for stmt in body {
                    self.compile_stmt(stmt, code)?;
                }
                code.push(IrInst::Jump(start));
                let end = code.len();
                code[exit] = IrInst::JumpIfFalse(end);
            }
            Stmt::For(name, iterable, body) => {
                self.compile_expr(iterable, code)?;
                code.push(IrInst::IterInit);
                let check = code.len();
                code.push(IrInst::IterNext(name.clone(), usize::MAX));
                for stmt in body {
                    self.compile_stmt(stmt, code)?;
                }
                code.push(IrInst::Jump(check));
                let end = code.len();
                code[check] = IrInst::IterNext(name.clone(), end);
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
    adam: HashMap<u64, AdamState>,
    backend: Box<dyn TensorBackend>,
}

impl IrRuntime {
    pub(crate) fn new() -> Self {
        Self::with_backend(backend::BackendKind::Cpu).expect("CPU backend must be available")
    }

    pub(crate) fn with_backend(kind: backend::BackendKind) -> Result<Self, String> {
        let backend = backend::create(kind)
            .map_err(|e| format!("Nano: backend {}: {}", kind.name(), e))?;
        Ok(Self {
            vars: HashMap::new(),
            functions: HashMap::new(),
            adam: HashMap::new(),
            backend,
        })
    }

    pub(crate) fn run(&mut self, program: &IrProgram) -> Result<(), String> {
        self.functions.extend(program.functions.clone());
        let _ = self.execute_code(&program.code)?;
        Ok(())
    }

    fn execute_code(&mut self, code: &[IrInst]) -> Result<Option<Value>, String> {
        let mut stack: Vec<Value> = Vec::new();
        let mut iterators: Vec<IterState> = Vec::new();
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
                    stack.push(self.binary_value(left, *op, right)?);
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
                IrInst::IterInit => {
                    let iterable = stack.pop().ok_or_else(|| "Nano IR: stack vazia em IterInit".to_string())?;
                    match iterable {
                        Value::List(values) => iterators.push(IterState { values, index: 0 }),
                        _ => return Err("Nano: for requer List".into()),
                    }
                }
                IrInst::IterNext(name, target) => {
                    let item = match iterators.last_mut() {
                        Some(iter) if iter.index < iter.values.len() => {
                            let value = iter.values[iter.index].clone();
                            iter.index += 1;
                            Some(value)
                        }
                        Some(_) => {
                            iterators.pop();
                            None
                        }
                        None => return Err("Nano IR: IterNext sem IterInit".into()),
                    };

                    match item {
                        Some(value) => { self.vars.insert(name.clone(), value); }
                        None => {
                            ip = *target;
                            continue;
                        }
                    }
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
                Value::Tensor(v) => Ok(Value::Number(v.borrow().data.len() as f64)),
                _ => Err("Nano: len() requer texto, lista, objeto ou tensor".into()),
            };
        }

        if name == "backend" {
            if !args.is_empty() {
                return Err("Nano: backend() não recebe argumentos".into());
            }
            return Ok(Value::Text(self.backend.kind().name().into()));
        }

        if name == "device" {
            if args.len() != 1 {
                return Err("Nano: device() recebe 1 tensor".into());
            }
            return match &args[0] {
                Value::Tensor(tensor) => Ok(Value::Text(tensor.borrow().device.name().into())),
                _ => Err("Nano: device() requer Tensor".into()),
            };
        }

        if name == "tensor" {
            if args.len() != 2 {
                return Err("Nano: tensor() recebe dados e shape".into());
            }
            let data = list_numbers(&args[0], "dados")?;
            let shape = list_shape(&args[1])?;
            return Ok(Value::Tensor(super::Tensor::new_on(data, shape, false, self.backend.kind())?));
        }

        if name == "parameter" {
            if args.len() != 2 {
                return Err("Nano: parameter() recebe dados e shape".into());
            }
            let data = list_numbers(&args[0], "dados")?;
            let shape = list_shape(&args[1])?;
            return Ok(Value::Tensor(super::Tensor::new_on(data, shape, true, self.backend.kind())?));
        }

        if name == "zeros" {
            if args.len() != 1 {
                return Err("Nano: zeros() recebe shape".into());
            }
            let shape = list_shape(&args[0])?;
            let size = shape.iter().copied().product::<usize>();
            return Ok(Value::Tensor(super::Tensor::new_on(vec![0.0; size], shape, false, self.backend.kind())?));
        }

        if name == "shape" {
            if args.len() != 1 {
                return Err("Nano: shape() recebe 1 tensor".into());
            }
            return match &args[0] {
                Value::Tensor(t) => Ok(Value::List(
                    t.borrow().shape.iter().map(|v| Value::Number(*v as f64)).collect()
                )),
                _ => Err("Nano: shape() requer Tensor".into()),
            };
        }

        if name == "matmul" {
            if args.len() != 2 {
                return Err("Nano: matmul() recebe 2 tensores".into());
            }
            return matmul_values(&args[0], &args[1], self.backend.as_ref());
        }

        if name == "sum" || name == "mean" {
            if args.len() != 1 {
                return Err(format!("Nano: {name}() recebe 1 tensor"));
            }
            return reduce_value(&args[0], name == "mean", self.backend.as_ref());
        }

        if name == "grad" {
            if args.len() != 2 {
                return Err("Nano: grad() recebe loss e parâmetro".into());
            }
            return gradient_value(&args[0], &args[1]);
        }

        if name == "step" {
            if args.len() != 3 {
                return Err("Nano: step() recebe parâmetro, gradiente e taxa".into());
            }
            return step_value(&args[0], &args[1], &args[2]);
        }

        if name == "adam" {
            if args.len() != 3 {
                return Err("Nano: adam() recebe parâmetro, gradiente e taxa".into());
            }
            return self.adam_value(&args[0], &args[1], &args[2]);
        }

        if name == "range" {
            if args.len() != 1 {
                return Err("Nano: range() recebe 1 argumento".into());
            }
            let limit = match &args[0] {
                Value::Number(v) if *v >= 0.0 && v.fract() == 0.0 => *v as usize,
                _ => return Err("Nano: range() requer Number inteiro não negativo".into()),
            };
            return Ok(Value::List((0..limit).map(|v| Value::Number(v as f64)).collect()));
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

    fn adam_value(&mut self, parameter: &Value, gradient: &Value, rate: &Value) -> Result<Value, String> {
        let param = match parameter {
            Value::Tensor(t) => std::rc::Rc::clone(t),
            _ => return Err("Nano: adam() requer parâmetro Tensor".into()),
        };
        let grad = match gradient {
            Value::Tensor(t) => std::rc::Rc::clone(t),
            _ => return Err("Nano: adam() requer gradiente Tensor".into()),
        };
        let lr = match rate {
            Value::Number(v) => *v as f32,
            _ => return Err("Nano: adam() requer taxa Number".into()),
        };

        let (id, len) = {
            let p = param.borrow();
            let g = grad.borrow();
            if p.shape != g.shape {
                return Err("Nano: parâmetro e gradiente precisam ter o mesmo shape".into());
            }
            (p.id, p.data.len())
        };

        let state = self.adam.entry(id).or_insert_with(|| AdamState {
            step: 0,
            m: vec![0.0; len],
            v: vec![0.0; len],
        });

        if state.m.len() != len || state.v.len() != len {
            return Err("Nano: estado Adam incompatível com o parâmetro".into());
        }

        state.step += 1;
        let t = state.step as f32;
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;

        {
            let mut p = param.borrow_mut();
            let g = grad.borrow();
            for i in 0..p.data.len() {
                state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g.data[i];
                state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g.data[i] * g.data[i];
                let m_hat = state.m[i] / (1.0 - beta1.powf(t));
                let v_hat = state.v[i] / (1.0 - beta2.powf(t));
                p.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        Ok(Value::Tensor(std::rc::Rc::clone(&param)))
    }

    fn binary_value(&mut self, a: Value, op: Op, b: Value) -> Result<Value, String> {
        match (&a, &b, op) {
            (Value::Tensor(left), Value::Tensor(right), Op::Add | Op::Sub | Op::Mul | Op::Div) => {
                let kind = match op {
                    Op::Add => TensorOpKind::Add,
                    Op::Sub => TensorOpKind::Sub,
                    Op::Mul => TensorOpKind::Mul,
                    Op::Div => TensorOpKind::Div,
                    _ => unreachable!(),
                };
                Ok(Value::Tensor(tensor_elementwise(left, right, kind, self.backend.as_ref())?))
            }
            (Value::Tensor(tensor), Value::Number(n), Op::Mul) |
            (Value::Number(n), Value::Tensor(tensor), Op::Mul) => {
                let scalar = super::Tensor::new_on(vec![*n as f32; tensor.borrow().data.len()], tensor.borrow().shape.clone(), false, self.backend.kind())?;
                Ok(Value::Tensor(tensor_elementwise(tensor, &scalar, TensorOpKind::Mul, self.backend.as_ref())?))
            }
            _ => binary(a, op, b),
        }
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

fn matmul_values(a: &Value, b: &Value, backend: &dyn TensorBackend) -> Result<Value, String> {
    let (left, right) = match (a, b) {
        (Value::Tensor(a), Value::Tensor(b)) => (a, b),
        _ => return Err("Nano: matmul() requer Tensor, Tensor".into()),
    };

    let (lshape, rshape, requires_grad) = {
        let l = left.borrow();
        let r = right.borrow();
        if l.device != backend.kind() || r.device != backend.kind() {
            return Err(format!(
                "Nano: tensors estão no dispositivo {} e o backend ativo é {}",
                if l.device != backend.kind() { l.device.name() } else { r.device.name() },
                backend.kind().name()
            ));
        }
        if l.device != r.device {
            return Err("Nano: matmul() requer tensors no mesmo dispositivo".into());
        }
        (l.shape.clone(), r.shape.clone(), l.requires_grad || r.requires_grad)
    };

    if lshape.len() != 2 || rshape.len() != 2 {
        return Err("Nano: matmul() nesta versão requer tensores 2D".into());
    }

    let (m, k) = (lshape[0], lshape[1]);
    let (k2, n) = (rshape[0], rshape[1]);
    if k != k2 {
        return Err(format!("Nano: matmul() incompatível: {}x{} com {}x{}", m, k, k2, n));
    }

    let out = {
        let l = left.borrow();
        let r = right.borrow();
        backend.matmul(&l.data, &l.shape, &r.data, &r.shape)
            .map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?
    };

    Ok(Value::Tensor(super::Tensor::derived_on(
        out,
        vec![m, n],
        requires_grad,
        backend.kind(),
        TensorOp::Matmul(std::rc::Rc::clone(left), std::rc::Rc::clone(right)),
    )?))
}


fn tensor_elementwise(
    a: &TensorRef,
    b: &TensorRef,
    op: TensorOpKind,
    backend: &dyn TensorBackend,
) -> Result<TensorRef, String> {
    let left = a.borrow();
    let right = b.borrow();
    if left.shape != right.shape {
        return Err("Nano: Tensor elementwise requer shapes iguais".into());
    }
    if left.device != backend.kind() || right.device != backend.kind() {
        return Err(format!(
            "Nano: tensors estão no dispositivo errado para o backend {}",
            backend.kind().name()
        ));
    }
    if left.device != right.device {
        return Err("Nano: Tensor elementwise requer o mesmo dispositivo".into());
    }

    let backend_op = match op {
        TensorOpKind::Add => ElementwiseOp::Add,
        TensorOpKind::Sub => ElementwiseOp::Sub,
        TensorOpKind::Mul => ElementwiseOp::Mul,
        TensorOpKind::Div => ElementwiseOp::Div,
    };
    let data = backend.elementwise(&left.data, &right.data, &left.shape, backend_op)
        .map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;

    let requires_grad = left.requires_grad || right.requires_grad;
    let shape = left.shape.clone();
    drop(left);
    drop(right);

    Ok(super::Tensor::derived_on(
        data,
        shape,
        requires_grad,
        backend.kind(),
        TensorOp::Elementwise(op, std::rc::Rc::clone(a), std::rc::Rc::clone(b)),
    )?)
}

fn reduce_value(
    value: &Value,
    mean: bool,
    backend: &dyn TensorBackend,
) -> Result<Value, String> {
    let tensor = match value {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: redução requer Tensor".into()),
    };
    let borrowed = tensor.borrow();
    if borrowed.device != backend.kind() {
        return Err(format!(
            "Nano: tensor está no dispositivo {}, mas o backend ativo é {}",
            borrowed.device.name(),
            backend.kind().name()
        ));
    }
    let result = backend.reduce(&borrowed.data, mean)
        .map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;
    let op = if mean { TensorOp::Mean(std::rc::Rc::clone(&tensor)) } else { TensorOp::Sum(std::rc::Rc::clone(&tensor)) };
    let requires_grad = borrowed.requires_grad;
    drop(borrowed);
    Ok(Value::Tensor(super::Tensor::derived_on(
        vec![result],
        vec![1],
        requires_grad,
        backend.kind(),
        op,
    )?))
}

fn gradient_value(loss: &Value, parameter: &Value) -> Result<Value, String> {
    let loss_ref = match loss {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: grad() requer Tensor como loss".into()),
    };
    let param_ref = match parameter {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: grad() requer Tensor como parâmetro".into()),
    };

    let output_shape = param_ref.borrow().shape.clone();
    let mut grads: HashMap<u64, Vec<f32>> = HashMap::new();
    let upstream = vec![1.0_f32; loss_ref.borrow().data.len()];
    backward(&loss_ref, upstream, &mut grads)?;

    let id = param_ref.borrow().id;
    let data = grads.get(&id).cloned().unwrap_or_else(|| vec![0.0; param_ref.borrow().data.len()]);
    let device = param_ref.borrow().device;
    Ok(Value::Tensor(super::Tensor::new_on(data, output_shape, false, device)?))
}

fn step_value(parameter: &Value, gradient: &Value, rate: &Value) -> Result<Value, String> {
    let param = match parameter {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: step() requer parâmetro Tensor".into()),
    };
    let grad = match gradient {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: step() requer gradiente Tensor".into()),
    };
    let lr = match rate {
        Value::Number(v) => *v as f32,
        _ => return Err("Nano: step() requer taxa Number".into()),
    };

    let mut p = param.borrow_mut();
    let g = grad.borrow();
    if p.shape != g.shape {
        return Err("Nano: parâmetro e gradiente precisam ter o mesmo shape".into());
    }
    for (value, delta) in p.data.iter_mut().zip(&g.data) {
        *value -= lr * delta;
    }
    drop(g);
    drop(p);
    Ok(Value::Tensor(param))
}

fn backward(
    node: &TensorRef,
    upstream: Vec<f32>,
    grads: &mut HashMap<u64, Vec<f32>>,
) -> Result<(), String> {
    let (id, op) = {
        let value = node.borrow();
        (value.id, value.op.clone())
    };

    let entry = grads.entry(id).or_insert_with(|| vec![0.0; upstream.len()]);
    for (slot, value) in entry.iter_mut().zip(upstream.iter()) {
        *slot += *value;
    }

    match op {
        TensorOp::Leaf => Ok(()),
        TensorOp::Elementwise(kind, left, right) => {
            let (ldata, rdata) = {
                let l = left.borrow();
                let r = right.borrow();
                (l.data.clone(), r.data.clone())
            };
            match kind {
                TensorOpKind::Add => {
                    backward(&left, upstream.clone(), grads)?;
                    backward(&right, upstream, grads)?;
                }
                TensorOpKind::Sub => {
                    backward(&left, upstream.clone(), grads)?;
                    backward(&right, upstream.into_iter().map(|v| -v).collect(), grads)?;
                }
                TensorOpKind::Mul => {
                    let lg = upstream.iter().zip(&rdata).map(|(u, r)| u * r).collect();
                    let rg = upstream.iter().zip(&ldata).map(|(u, l)| u * l).collect();
                    backward(&left, lg, grads)?;
                    backward(&right, rg, grads)?;
                }
                TensorOpKind::Div => {
                    let lg = upstream.iter().zip(&rdata).map(|(u, r)| u / r).collect();
                    let rg = upstream.iter().zip(&ldata).zip(&rdata).map(|((u, l), r)| -u * l / (r * r)).collect();
                    backward(&left, lg, grads)?;
                    backward(&right, rg, grads)?;
                }
            }
            Ok(())
        }
        TensorOp::Sum(input) => {
            let scalar = upstream.first().copied().unwrap_or(0.0);
            let size = input.borrow().data.len();
            backward(&input, vec![scalar; size], grads)
        }
        TensorOp::Mean(input) => {
            let scalar = upstream.first().copied().unwrap_or(0.0);
            let size = input.borrow().data.len().max(1);
            backward(&input, vec![scalar / size as f32; size], grads)
        }
        TensorOp::Matmul(left, right) => {
            let upstream_shape = node.borrow().shape.clone();
            if upstream_shape.len() != 2 {
                return Err("Nano: grad matmul requer saída 2D".into());
            }
            let (m, n) = (upstream_shape[0], upstream_shape[1]);
            let (ldata, lshape, rdata, rshape) = {
                let l = left.borrow();
                let r = right.borrow();
                (l.data.clone(), l.shape.clone(), r.data.clone(), r.shape.clone())
            };
            if lshape.len() != 2 || rshape.len() != 2 {
                return Err("Nano: grad matmul requer entradas 2D".into());
            }
            let k = lshape[1];
            if lshape[0] != m || rshape[1] != n || rshape[0] != k {
                return Err("Nano: shapes incompatíveis no grad matmul".into());
            }

            let mut lg = vec![0.0_f32; m * k];
            for i in 0..m {
                for x in 0..k {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += upstream[i * n + j] * rdata[x * n + j];
                    }
                    lg[i * k + x] = sum;
                }
            }

            let mut rg = vec![0.0_f32; k * n];
            for x in 0..k {
                for j in 0..n {
                    let mut sum = 0.0;
                    for i in 0..m {
                        sum += ldata[i * k + x] * upstream[i * n + j];
                    }
                    rg[x * n + j] = sum;
                }
            }

            backward(&left, lg, grads)?;
            backward(&right, rg, grads)?;
            Ok(())
        }
    }
}
