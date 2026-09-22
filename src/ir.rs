use std::collections::{HashMap, HashSet};
use crate::dtype::DType;
use crate::memory::MemoryPlanner;
use crate::ui;
use crate::sync_runtime;
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::thread::{self, JoinHandle};
use std::sync::mpsc::{self, Sender, Receiver};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use serde_json::Value as JsonValue;

use super::{backend, qualified_name, Expr, Lexer, Op, Parser, Semantic, SourceSpan, Stmt, TensorOp, TensorOpKind, TensorRef, Value};
use backend::{ElementwiseOp, TensorBackend};

#[derive(Debug, Clone)]
pub(crate) enum IrInst {
    Const(Value),
    Load(String),
    Store(String),
    SetIndex,
    SetField(String),
    Binary(Op),
    Unary(crate::UnaryOp),
    FusedMulAdd,
    MakeList(usize),
    MakeObject(Vec<String>),
    Index,
    Field(String),
    Call(String, usize),
    CallValue(usize),
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
enum TaskValue {
    Number(f64),
    Text(String),
    Boolean(bool),
    List(Vec<TaskValue>),
    Object(HashMap<String, TaskValue>),
    Null,
    Function(String),
}

#[derive(Debug, Clone)]
enum TaskInst {
    Const(TaskValue),
    Load(String),
    Store(String),
    SetIndex,
    SetField(String),
    Binary(Op),
    Unary(crate::UnaryOp),
    FusedMulAdd,
    MakeList(usize),
    MakeObject(Vec<String>),
    Index,
    Field(String),
    Call(String, usize),
    CallValue(usize),
    Print,
    Pop,
    JumpIfFalse(usize),
    Jump(usize),
    IterInit,
    IterNext(String, usize),
    Return,
    Use(String),
}

impl TaskInst {
    fn from_ir(inst: &IrInst) -> Result<Self, String> {
        Ok(match inst {
            IrInst::Const(value) => Self::Const(TaskValue::from_value(value.clone())?),
            IrInst::Load(name) => Self::Load(name.clone()),
            IrInst::Store(name) => Self::Store(name.clone()),
            IrInst::SetIndex => Self::SetIndex,
            IrInst::SetField(name) => Self::SetField(name.clone()),
            IrInst::Binary(op) => Self::Binary(*op),
            IrInst::Unary(op) => Self::Unary(*op),
            IrInst::FusedMulAdd => Self::FusedMulAdd,
            IrInst::MakeList(count) => Self::MakeList(*count),
            IrInst::MakeObject(keys) => Self::MakeObject(keys.clone()),
            IrInst::Index => Self::Index,
            IrInst::Field(name) => Self::Field(name.clone()),
            IrInst::Call(name, count) => Self::Call(name.clone(), *count),
            IrInst::CallValue(count) => Self::CallValue(*count),
            IrInst::Print => Self::Print,
            IrInst::Pop => Self::Pop,
            IrInst::JumpIfFalse(target) => Self::JumpIfFalse(*target),
            IrInst::Jump(target) => Self::Jump(*target),
            IrInst::IterInit => Self::IterInit,
            IrInst::IterNext(name, target) => Self::IterNext(name.clone(), *target),
            IrInst::Return => Self::Return,
            IrInst::Use(path) => Self::Use(path.clone()),
        })
    }

    fn into_ir(self) -> IrInst {
        match self {
            Self::Const(value) => IrInst::Const(value.into_value()),
            Self::Load(name) => IrInst::Load(name),
            Self::Store(name) => IrInst::Store(name),
            Self::SetIndex => IrInst::SetIndex,
            Self::SetField(name) => IrInst::SetField(name),
            Self::Binary(op) => IrInst::Binary(op),
            Self::Unary(op) => IrInst::Unary(op),
            Self::FusedMulAdd => IrInst::FusedMulAdd,
            Self::MakeList(count) => IrInst::MakeList(count),
            Self::MakeObject(keys) => IrInst::MakeObject(keys),
            Self::Index => IrInst::Index,
            Self::Field(name) => IrInst::Field(name),
            Self::Call(name, count) => IrInst::Call(name, count),
            Self::CallValue(count) => IrInst::CallValue(count),
            Self::Print => IrInst::Print,
            Self::Pop => IrInst::Pop,
            Self::JumpIfFalse(target) => IrInst::JumpIfFalse(target),
            Self::Jump(target) => IrInst::Jump(target),
            Self::IterInit => IrInst::IterInit,
            Self::IterNext(name, target) => IrInst::IterNext(name, target),
            Self::Return => IrInst::Return,
            Self::Use(path) => IrInst::Use(path),
        }
    }
}

#[derive(Debug, Clone)]
struct TaskFunction {
    params: Vec<String>,
    code: Vec<TaskInst>,
}

impl TaskFunction {
    fn from_ir(function: &IrFunction) -> Result<Self, String> {
        let mut code = Vec::with_capacity(function.code.len());
        for inst in &function.code {
            code.push(TaskInst::from_ir(inst)?);
        }
        Ok(Self { params: function.params.clone(), code })
    }

    fn into_ir(self) -> IrFunction {
        IrFunction {
            params: self.params,
            code: self.code.into_iter().map(TaskInst::into_ir).collect(),
        }
    }
}

impl TaskValue {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::Number(v) => Ok(Self::Number(v)),
            Value::Text(v) => Ok(Self::Text(v)),
            Value::Boolean(v) => Ok(Self::Boolean(v)),
            Value::Null => Ok(Self::Null),
            Value::Function(name) => Ok(Self::Function(name)),
            Value::List(values) => values.into_iter().map(Self::from_value).collect::<Result<Vec<_>, _>>().map(Self::List),
            Value::Object(values) => values.into_iter()
                .map(|(key, value)| Self::from_value(value).map(|value| (key, value)))
                .collect::<Result<HashMap<_, _>, _>>()
                .map(Self::Object),
            Value::Tensor(_) => Err("Nano: tarefa não pode devolver Tensor entre threads".into()),
        }
    }

    fn into_value(self) -> Value {
        match self {
            Self::Number(v) => Value::Number(v),
            Self::Text(v) => Value::Text(v),
            Self::Boolean(v) => Value::Boolean(v),
            Self::Null => Value::Null,
            Self::Function(name) => Value::Function(name),
            Self::List(values) => Value::List(values.into_iter().map(Self::into_value).collect()),
            Self::Object(values) => Value::Object(values.into_iter().map(|(key, value)| (key, value.into_value())).collect()),
        }
    }
}

struct AdamState {
    step: u64,
    m: Vec<f32>,
    v: Vec<f32>,
}

pub(crate) struct DebugMetadata {
    pub(crate) main: Vec<Option<SourceSpan>>,
    pub(crate) functions: HashMap<String, Vec<Option<SourceSpan>>>,
}

pub(crate) struct Compiler {
    break_targets: Vec<Vec<usize>>,
    known_functions: HashSet<String>,
    debug_enabled: bool,
    debug_input: Vec<SourceSpan>,
    debug_cursor: usize,
    debug_active: Vec<Option<SourceSpan>>,
    debug_main_spans: Vec<Option<SourceSpan>>,
    debug_functions: HashMap<String, Vec<Option<SourceSpan>>>,
}

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
    pub(crate) fn new() -> Self {
        Self {
            break_targets: Vec::new(),
            known_functions: HashSet::new(),
            debug_enabled: false,
            debug_input: Vec::new(),
            debug_cursor: 0,
            debug_active: Vec::new(),
            debug_main_spans: Vec::new(),
            debug_functions: HashMap::new(),
        }
    }

    pub(crate) fn compile_debug(
        &mut self,
        program: &[Stmt],
        spans: &[SourceSpan],
    ) -> Result<(IrProgram, DebugMetadata), String> {
        self.debug_enabled = true;
        self.debug_input = spans.to_vec();
        self.debug_cursor = 0;
        self.debug_active.clear();
        self.debug_main_spans.clear();
        self.debug_functions.clear();
        let result = self.compile(program);
        self.debug_enabled = false;
        self.debug_input.clear();
        let program = result?;
        Ok((program, DebugMetadata {
            main: self.debug_main_spans.clone(),
            functions: self.debug_functions.clone(),
        }))
    }

    pub(crate) fn compile(&mut self, program: &[Stmt]) -> Result<IrProgram, String> {
        let mut functions = HashMap::new();
        self.known_functions.clear();
        self.debug_main_spans.clear();
        for stmt in program {
            if let Stmt::Function(name, _, _) = stmt {
                self.known_functions.insert(name.clone());
            }
        }
        let mut code = Vec::new();

        for stmt in program {
            if let Stmt::Function(name, params, body) = stmt {
                let mut function_code = Vec::new();
                self.break_targets.clear();
                self.debug_active.clear();
                for stmt in body {
                    self.compile_stmt(stmt, &mut function_code)?;
                }
                if !matches!(function_code.last(), Some(IrInst::Return)) {
                    function_code.push(IrInst::Const(Value::Null));
                    function_code.push(IrInst::Return);
                }
                if self.debug_enabled {
                    self.debug_active.resize(function_code.len(), None);
                    self.debug_functions.insert(name.clone(), self.debug_active.clone());
                }
                functions.insert(name.clone(), IrFunction { params: params.clone(), code: function_code });
            }
        }

        self.debug_active.clear();
        for stmt in program {
            if !matches!(stmt, Stmt::Function(_, _, _)) {
                self.compile_stmt(stmt, &mut code)?;
            }
        }
        if self.debug_enabled {
            self.debug_active.resize(code.len(), None);
            self.debug_main_spans = self.debug_active.clone();
        }

        Ok(IrProgram { code, functions })
    }

    fn compile_stmt(&mut self, stmt: &Stmt, code: &mut Vec<IrInst>) -> Result<(), String> {
        let before = code.len();
        let span = if self.debug_enabled {
            let span = self.debug_input.get(self.debug_cursor).copied();
            self.debug_cursor = self.debug_cursor.saturating_add(1);
            span
        } else {
            None
        };
        let result = self.compile_stmt_inner(stmt, code);
        if result.is_ok() && self.debug_enabled {
            self.debug_active.resize(code.len(), None);
            if let Some(span) = span {
                for index in before..code.len() {
                    if self.debug_active[index].is_none() {
                        self.debug_active[index] = Some(span);
                    }
                }
            }
        }
        result
    }

    fn compile_stmt_inner(&mut self, stmt: &Stmt, code: &mut Vec<IrInst>) -> Result<(), String> {
        match stmt {
            Stmt::Assign(name, expr) => {
                self.compile_expr(expr, code)?;
                code.push(IrInst::Store(name.clone()));
            }
            Stmt::AssignIndex(target, index, value) => {
                let Expr::Var(name) = target else {
                    return Err("Nano: atribuição por índice exige uma variável de lista/objeto como alvo".into());
                };
                self.compile_expr(target, code)?;
                self.compile_expr(index, code)?;
                self.compile_expr(value, code)?;
                code.push(IrInst::SetIndex);
                code.push(IrInst::Store(name.clone()));
            }
            Stmt::AssignField(target, name, value) => {
                let Expr::Var(base) = target else {
                    return Err("Nano: atribuição por campo exige uma variável Object como alvo".into());
                };
                self.compile_expr(target, code)?;
                self.compile_expr(value, code)?;
                code.push(IrInst::SetField(name.clone()));
                code.push(IrInst::Store(base.clone()));
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
            Stmt::Break => {
                let Some(targets) = self.break_targets.last_mut() else {
                    return Err("Nano: break só pode ser usado dentro de while ou for".into());
                };
                let jump = code.len();
                code.push(IrInst::Jump(usize::MAX));
                targets.push(jump);
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
                self.break_targets.push(Vec::new());
                for stmt in body {
                    self.compile_stmt(stmt, code)?;
                }
                code.push(IrInst::Jump(start));
                let end = code.len();
                code[exit] = IrInst::JumpIfFalse(end);
                let breaks = self.break_targets.pop().unwrap_or_default();
                for jump in breaks {
                    code[jump] = IrInst::Jump(end);
                }
            }
            Stmt::For(name, iterable, body) => {
                self.compile_expr(iterable, code)?;
                code.push(IrInst::IterInit);
                let check = code.len();
                code.push(IrInst::IterNext(name.clone(), usize::MAX));
                self.break_targets.push(Vec::new());
                for stmt in body {
                    self.compile_stmt(stmt, code)?;
                }
                code.push(IrInst::Jump(check));
                let end = code.len();
                code[check] = IrInst::IterNext(name.clone(), end);
                let breaks = self.break_targets.pop().unwrap_or_default();
                for jump in breaks {
                    code[jump] = IrInst::Jump(end);
                }
            }
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &Expr, code: &mut Vec<IrInst>) -> Result<(), String> {
        match expr {
            Expr::Value(value) => code.push(IrInst::Const(value.clone())),
            Expr::Var(name) => {
                if self.known_functions.contains(name) {
                    code.push(IrInst::Const(Value::Function(name.clone())));
                } else {
                    code.push(IrInst::Load(name.clone()));
                }
            },
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
            Expr::Binary(left, Op::Add, right) => {
                if let Expr::Binary(a, Op::Mul, b) = &**left {
                    self.compile_expr(a, code)?;
                    self.compile_expr(b, code)?;
                    self.compile_expr(right, code)?;
                    code.push(IrInst::FusedMulAdd);
                } else if let Expr::Binary(a, Op::Mul, b) = &**right {
                    self.compile_expr(a, code)?;
                    self.compile_expr(b, code)?;
                    self.compile_expr(left, code)?;
                    code.push(IrInst::FusedMulAdd);
                } else {
                    self.compile_expr(left, code)?;
                    self.compile_expr(right, code)?;
                    code.push(IrInst::Binary(Op::Add));
                }
            }
            Expr::Binary(left, Op::And, right) => {
                self.compile_expr(left, code)?;
                let left_false = code.len();
                code.push(IrInst::JumpIfFalse(usize::MAX));

                self.compile_expr(right, code)?;
                let right_false = code.len();
                code.push(IrInst::JumpIfFalse(usize::MAX));

                code.push(IrInst::Const(Value::Boolean(true)));
                let end_jump = code.len();
                code.push(IrInst::Jump(usize::MAX));

                let false_label = code.len();
                code[left_false] = IrInst::JumpIfFalse(false_label);
                code[right_false] = IrInst::JumpIfFalse(false_label);
                code.push(IrInst::Const(Value::Boolean(false)));

                let end = code.len();
                code[end_jump] = IrInst::Jump(end);
            }
            Expr::Binary(left, Op::Or, right) => {
                self.compile_expr(left, code)?;
                let evaluate_right = code.len();
                code.push(IrInst::JumpIfFalse(usize::MAX));
                code.push(IrInst::Const(Value::Boolean(true)));
                let end_jump = code.len();
                code.push(IrInst::Jump(usize::MAX));

                let right_start = code.len();
                code[evaluate_right] = IrInst::JumpIfFalse(right_start);
                self.compile_expr(right, code)?;
                let right_false = code.len();
                code.push(IrInst::JumpIfFalse(usize::MAX));
                code.push(IrInst::Const(Value::Boolean(true)));
                let right_end_jump = code.len();
                code.push(IrInst::Jump(usize::MAX));

                let false_label = code.len();
                code[right_false] = IrInst::JumpIfFalse(false_label);
                code.push(IrInst::Const(Value::Boolean(false)));

                let end = code.len();
                code[end_jump] = IrInst::Jump(end);
                code[right_end_jump] = IrInst::Jump(end);
            }
            Expr::Binary(left, op, right) => {
                self.compile_expr(left, code)?;
                self.compile_expr(right, code)?;
                code.push(IrInst::Binary(*op));
            }
            Expr::Unary(op, expr) => {
                self.compile_expr(expr, code)?;
                code.push(IrInst::Unary(*op));
            }
            Expr::Call(target, args) => {
                if let Some(name) = qualified_name(target) {
                    if self.known_functions.contains(&name) || is_builtin_name(&name) {
                        for arg in args {
                            self.compile_expr(arg, code)?;
                        }
                        code.push(IrInst::Call(name, args.len()));
                        return Ok(());
                    }
                }

                self.compile_expr(target, code)?;
                for arg in args {
                    self.compile_expr(arg, code)?;
                }
                code.push(IrInst::CallValue(args.len()));
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
    dtype: DType,
    memory: MemoryPlanner,
    loaded_modules: HashSet<PathBuf>,
    module_stack: Vec<PathBuf>,
    next_handle: u64,
    tcp_streams: HashMap<u64, TcpStream>,
    tcp_listeners: HashMap<u64, TcpListener>,
    children: HashMap<u64, Child>,
    tasks: HashMap<u64, JoinHandle<Result<TaskValue, String>>>,
    ui_windows: HashMap<u64, ui::UiHandle>,
    channels: HashMap<u64, (Sender<Value>, Receiver<Value>)>,
    quiet: bool,
}

impl IrRuntime {
    fn sync_tensor_host(&self, tensor: &TensorRef) -> Result<(), String> {
        let (id, elements, device, valid) = {
            let t=tensor.borrow();
            (t.id,t.data_len(),t.device,t.host_valid)
        };
        if device!=backend::BackendKind::Gpu || valid { return Ok(()); }
        let data=self.backend.read_tensor(id,elements)
            .map_err(|e|format!("Nano: readback {}: {}",self.backend.kind().name(),e))?;
        tensor.borrow_mut().set_data_f32(data);
        Ok(())
    }

    pub(crate) fn new() -> Self {
        Self::with_backend_and_dtype(backend::BackendKind::Cpu, DType::F32)
            .expect("CPU backend must be available")
    }

    pub(crate) fn with_backend(kind: backend::BackendKind) -> Result<Self, String> {
        Self::with_backend_and_dtype(kind, DType::F32)
    }

    pub(crate) fn with_backend_and_dtype(kind: backend::BackendKind, dtype: DType) -> Result<Self, String> {
        let backend = backend::create(kind)
            .map_err(|e| format!("Nano: backend {}: {}", kind.name(), e))?;
        Ok(Self {
            vars: HashMap::new(),
            functions: HashMap::new(),
            adam: HashMap::new(),
            backend,
            dtype,
            memory: MemoryPlanner::new(),
            loaded_modules: HashSet::new(),
            module_stack: Vec::new(),
            next_handle: 1,
            tcp_streams: HashMap::new(),
            tcp_listeners: HashMap::new(),
            children: HashMap::new(),
            tasks: HashMap::new(),
            ui_windows: HashMap::new(),
            channels: HashMap::new(),
            quiet: false,
        })
    }

    pub(crate) fn run(&mut self, program: &IrProgram) -> Result<(), String> {
        self.functions.extend(program.functions.clone());

        if program.code.is_empty() {
            if let Some(main) = self.functions.get("main").cloned() {
                if !main.params.is_empty() {
                    return Err("Nano: função main() não pode receber parâmetros no entrypoint".into());
                }
                let _ = self.execute_code(&main.code)?;
            }
            return Ok(());
        }

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
                        .or_else(|| is_builtin_name(name).then(|| Value::Function(name.clone())))
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
                IrInst::Unary(op) => {
                    let value = stack.pop().ok_or_else(|| "Nano IR: stack vazia no operando unário".to_string())?;
                    stack.push(unary_value(value, *op)?);
                }
                IrInst::FusedMulAdd => {
                    let bias = stack.pop().ok_or_else(|| "Nano IR: stack vazia no bias do FMA".to_string())?;
                    let right = stack.pop().ok_or_else(|| "Nano IR: stack vazia no operando direito do FMA".to_string())?;
                    let left = stack.pop().ok_or_else(|| "Nano IR: stack vazia no operando esquerdo do FMA".to_string())?;
                    stack.push(self.fused_mul_add_value(left, right, bias)?);
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
                    stack.push(self.index_value(target, index)?);
                }
                IrInst::SetIndex => {
                    let value = stack.pop().ok_or_else(|| "Nano IR: stack vazia no valor de atribuição".to_string())?;
                    let index = stack.pop().ok_or_else(|| "Nano IR: stack vazia no índice de atribuição".to_string())?;
                    let target = stack.pop().ok_or_else(|| "Nano IR: stack vazia no alvo de atribuição".to_string())?;
                    let updated = match (target, index) {
                        (Value::List(mut values), Value::Number(n)) if n >= 0.0 && n.fract() == 0.0 => {
                            let index = n as usize;
                            if index >= values.len() {
                                return Err("Nano: índice de atribuição fora do limite".into());
                            }
                            values[index] = value;
                            Value::List(values)
                        }
                        (Value::Object(mut values), Value::Text(key)) => {
                            values.insert(key, value);
                            Value::Object(values)
                        }
                        _ => return Err("Nano: atribuição por índice requer List[número] ou Object[texto]".into()),
                    };
                    stack.push(updated);
                }
                IrInst::SetField(name) => {
                    let value = stack.pop().ok_or_else(|| "Nano IR: stack vazia no valor de campo".to_string())?;
                    let target = stack.pop().ok_or_else(|| "Nano IR: stack vazia no alvo de campo".to_string())?;
                    let mut values = match target {
                        Value::Object(values) => values,
                        _ => return Err("Nano: atribuição por campo requer Object".into()),
                    };
                    values.insert(name.clone(), value);
                    stack.push(Value::Object(values));
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
                IrInst::CallValue(count) => {
                    let mut args = pop_n(&mut stack, *count)?;
                    args.reverse();
                    let target = stack.pop().ok_or_else(|| "Nano IR: stack vazia no alvo da chamada".to_string())?;
                    let name = match target {
                        Value::Function(name) => name,
                        _ => return Err("Nano: alvo da chamada não é uma Function".into()),
                    };
                    let value = self.call(&name, args)?;
                    stack.push(value);
                }
                IrInst::Print => {
                    let value = stack.pop().ok_or_else(|| "Nano IR: stack vazia em Print".to_string())?;
                    if !self.quiet {
                        println!("{}", value.show());
                    }
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

    fn index_value(&self,target:Value,index:Value)->Result<Value,String>{
        match target{
            Value::Tensor(t)=>{self.sync_tensor_host(&t)?;index_value(Value::Tensor(t),index)}
            other=>index_value(other,index),
        }
    }

    fn call(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        let name = match name {
            "std.fs.read_text" => "fs_read_text",
            "std.fs.write_text" => "fs_write_text",
            "std.fs.append_text" => "fs_append_text",
            "std.fs.exists" => "fs_exists",
            "std.fs.list" => "fs_list",
            "std.fs.mkdir" => "fs_mkdir",
            "std.fs.remove" => "fs_remove",
            "std.fs.is_file" => "fs_is_file",
            "std.fs.is_dir" => "fs_is_dir",
            "std.fs.cwd" => "fs_cwd",
            "std.path.join" => "path_join",
            "std.path.basename" => "path_basename",
            "std.path.dirname" => "path_dirname",
            "std.path.extension" => "path_extension",
            "std.process.spawn" => "process_spawn",
            "std.process.wait" => "process_wait",
            "std.process.output" => "process_output",
            "std.os.cwd" => "os_cwd",
            "std.os.args" => "os_args",
            "std.time.now_ms" => "time_now_ms",
            "std.time.sleep_ms" => "time_sleep_ms",
            "std.net.tcp_connect" => "net_tcp_connect",
            "std.net.tcp_listen" => "net_tcp_listen",
            "std.net.tcp_accept" => "net_tcp_accept",
            "std.net.tcp_send" => "net_tcp_send",
            "std.net.tcp_recv" => "net_tcp_recv",
            "std.net.tcp_close" => "net_tcp_close",
            "std.net.http_get" => "net_http_get",
            "std.http.get" => "net_http_get",
            "std.env.get" => "env_get",
            "std.env.set" => "env_set",
            "std.json.encode" => "json_encode",
            "std.json.decode" => "json_decode",
            "std.sync.mutex_new" => "sync_mutex_new",
            "std.sync.mutex_get" => "sync_mutex_get",
            "std.sync.mutex_set" => "sync_mutex_set",
            "std.sync.mutex_swap" => "sync_mutex_swap",
            "std.sync.mutex_close" => "sync_mutex_close",
            "std.sync.semaphore_new" => "sync_semaphore_new",
            "std.sync.semaphore_acquire" => "sync_semaphore_acquire",
            "std.sync.semaphore_release" => "sync_semaphore_release",
            "std.sync.semaphore_close" => "sync_semaphore_close",
            "std.sync.channel" => "sync_channel_new",
            "std.sync.send" => "sync_channel_send",
            "std.sync.recv" => "sync_channel_recv",
            "std.sync.recv_timeout" => "sync_channel_recv_timeout",
            "std.sync.try_recv" => "sync_channel_try_recv",
            "std.sync.close_channel" => "sync_channel_close",
            "std.crypto.sha256" => "crypto_sha256",
            "std.crypto.hmac_sha256" => "crypto_hmac_sha256",
            "std.http.request" => "http_request",
            "std.http.post" => "http_post",
            "std.http.get_structured" => "http_get_structured",
            "std.http.get_secure" => "https_get",
            "std.math.abs" => "abs",
            "std.math.matmul_transposed" => "matmul_transposed",
            "std.math.sqrt" => "sqrt",
            "std.math.floor" => "floor",
            "std.math.ceil" => "ceil",
            "std.math.round" => "round",
            "std.math.sin" => "sin",
            "std.math.cos" => "cos",
            "std.math.tan" => "tan",
            "std.math.exp" => "exp",
            "std.math.log" => "log",
            "std.math.pow" => "pow",
            "std.math.min" => "min",
            "std.math.max" => "max",
            "std.ui.window" => "ui_window",
            "std.ui.set_title" => "ui_set_title",
            "std.ui.close" => "ui_close",
            "std.ui.poll_event" => "ui_poll_event",
            "std.ui.clear" => "ui_clear",
            "std.ui.rect" => "ui_rect",
            "std.ui.button" => "ui_button",
            "std.ui.text" => "ui_text",
            "std.ui.vbox" => "ui_vbox",
            "std.async.channel" => "channel",
            "std.async.send" => "send",
            "std.async.recv" => "recv",
            "std.async.close_channel" => "close_channel",
            "std.async.spawn" => "async_spawn",
            "std.async.join" => "async_join",
            "std.async.select" => "async_select",
            "std.async.all" => "async_all",
            "std.async.sleep_ms" => "async_sleep_ms",
            "std.async.yield" => "thread_yield",
            "std.backend.info" => "backend_info",
            other => other,
        };

        if name == "backend_info" {
            if !args.is_empty() { return Err("Nano: backend.info() não recebe argumentos".into()); }
            let caps = self.backend.capabilities();
            let mut info = HashMap::new();
            info.insert("backend".into(), Value::Text(self.backend.kind().name().into()));
            info.insert("matmul".into(), Value::Boolean(caps.matmul));
            info.insert("matmul_transposed".into(), Value::Boolean(caps.matmul_transposed));
            info.insert("elementwise".into(), Value::Boolean(caps.elementwise));
            info.insert("fused_mul_add".into(), Value::Boolean(caps.fused_mul_add));
            info.insert("reduce".into(), Value::Boolean(caps.reduce));
            info.insert("resident_async".into(), Value::Boolean(caps.resident_async));
            info.insert("readback".into(), Value::Boolean(caps.readback));
            return Ok(Value::Object(info));
        }

        if name == "fs_read_text" {
            if args.len() != 1 { return Err("Nano: fs_read_text() recebe caminho".into()); }
            let path = text_arg(&args[0], "caminho")?;
            return fs::read_to_string(&path)
                .map(Value::Text)
                .map_err(|e| format!("Nano: fs_read_text('{path}'): {e}"));
        }

        if name == "fs_write_text" || name == "fs_append_text" {
            if args.len() != 2 { return Err(format!("Nano: {name}() recebe caminho e texto")); }
            let path = text_arg(&args[0], "caminho")?;
            let content = text_arg(&args[1], "texto")?;
            let result = if name == "fs_write_text" {
                fs::write(&path, content)
            } else {
                let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)
                    .map_err(|e| format!("Nano: fs_append_text('{path}'): {e}"))?;
                file.write_all(content.as_bytes())
            };
            return result.map(|_| Value::Null).map_err(|e| format!("Nano: {name}('{path}'): {e}"));
        }

        if name == "fs_exists" {
            if args.len() != 1 { return Err("Nano: fs_exists() recebe caminho".into()); }
            let path = text_arg(&args[0], "caminho")?;
            return Ok(Value::Boolean(Path::new(&path).exists()));
        }

        if name == "fs_is_file" || name == "fs_is_dir" {
            if args.len() != 1 { return Err(format!("Nano: {name}() recebe caminho")); }
            let path = text_arg(&args[0], "caminho")?;
            let metadata = fs::metadata(&path).map_err(|e| format!("Nano: {name}('{path}'): {e}"))?;
            return Ok(Value::Boolean(if name == "fs_is_file" { metadata.is_file() } else { metadata.is_dir() }));
        }

        if name == "fs_cwd" || name == "os_cwd" {
            if !args.is_empty() { return Err(format!("Nano: {name}() não recebe argumentos")); }
            return std::env::current_dir()
                .map(|path| Value::Text(path.to_string_lossy().into_owned()))
                .map_err(|e| format!("Nano: {name}(): {e}"));
        }

        if name == "path_join" {
            if args.len() < 2 { return Err("Nano: path_join() recebe pelo menos 2 caminhos".into()); }
            let mut path = PathBuf::from(text_arg(&args[0], "caminho")?);
            for arg in &args[1..] { path.push(text_arg(arg, "caminho")?); }
            return Ok(Value::Text(path.to_string_lossy().into_owned()));
        }

        if matches!(name, "path_basename" | "path_dirname" | "path_extension") {
            if args.len() != 1 { return Err(format!("Nano: {name}() recebe 1 caminho")); }
            let value = text_arg(&args[0], "caminho")?;
            let path = Path::new(&value);
            let output = match name {
                "path_basename" => path.file_name().map(|v| v.to_string_lossy().into_owned()).unwrap_or_default(),
                "path_dirname" => path.parent().map(|v| v.to_string_lossy().into_owned()).unwrap_or_default(),
                _ => path.extension().map(|v| v.to_string_lossy().into_owned()).unwrap_or_default(),
            };
            return Ok(Value::Text(output));
        }

        if name == "fs_list" {
            if args.len() != 1 { return Err("Nano: fs_list() recebe diretório".into()); }
            let path = text_arg(&args[0], "diretório")?;
            let mut items = Vec::new();
            for entry in fs::read_dir(&path).map_err(|e| format!("Nano: fs_list('{path}'): {e}"))? {
                let entry = entry.map_err(|e| format!("Nano: fs_list('{path}'): {e}"))?;
                items.push(Value::Text(entry.file_name().to_string_lossy().into_owned()));
            }
            items.sort_by(|a,b| a.show().cmp(&b.show()));
            return Ok(Value::List(items));
        }

        if name == "fs_mkdir" {
            if args.len() != 1 { return Err("Nano: fs_mkdir() recebe diretório".into()); }
            let path = text_arg(&args[0], "diretório")?;
            return fs::create_dir_all(&path).map(|_| Value::Null)
                .map_err(|e| format!("Nano: fs_mkdir('{path}'): {e}"));
        }

        if name == "fs_remove" {
            if args.len() != 1 { return Err("Nano: fs_remove() recebe caminho".into()); }
            let path = text_arg(&args[0], "caminho")?;
            let metadata = fs::metadata(&path).map_err(|e| format!("Nano: fs_remove('{path}'): {e}"))?;
            return if metadata.is_dir() { fs::remove_dir_all(&path) } else { fs::remove_file(&path) }
                .map(|_| Value::Null)
                .map_err(|e| format!("Nano: fs_remove('{path}'): {e}"));
        }

        if name == "env_get" {
            if args.len() != 1 { return Err("Nano: env_get() recebe nome".into()); }
            let key = text_arg(&args[0], "nome")?;
            return Ok(match std::env::var(&key) { Ok(v) => Value::Text(v), Err(_) => Value::Null });
        }

        if name == "env_set" {
            if args.len() != 2 { return Err("Nano: env_set() recebe nome e valor".into()); }
            let key = text_arg(&args[0], "nome")?;
            let value = text_arg(&args[1], "valor")?;
            std::env::set_var(&key, &value);
            return Ok(Value::Null);
        }
        if name == "http_request" || name == "http_get_structured" || name == "http_post" {
            let (method, host, port, path, headers, body) = match name {
                "http_get_structured" => {
                    if args.len() != 3 { return Err("Nano: http.get_structured() recebe host, porta e caminho".into()); }
                    ("GET".into(), text_arg(&args[0], "host")?, integer_arg(&args[1], "porta")? as u16, text_arg(&args[2], "caminho")?, HashMap::new(), String::new())
                }
                "http_post" => {
                    if args.len() != 6 { return Err("Nano: http.post() recebe host, porta, caminho, headers, body e content_type".into()); }
                    let mut headers = object_text_map(&args[3], "headers")?;
                    let content_type = text_arg(&args[5], "content_type")?;
                    headers.entry("Content-Type".into()).or_insert(content_type);
                    ("POST".into(), text_arg(&args[0], "host")?, integer_arg(&args[1], "porta")? as u16, text_arg(&args[2], "caminho")?, headers, text_arg(&args[4], "body")?)
                }
                _ => {
                    if args.len() != 6 { return Err("Nano: http.request() recebe método, host, porta, caminho, headers e body".into()); }
                    (text_arg(&args[0], "método")?, text_arg(&args[1], "host")?, integer_arg(&args[2], "porta")? as u16, text_arg(&args[3], "caminho")?, object_text_map(&args[4], "headers")?, text_arg(&args[5], "body")?)
                }
            };
            return crate::http::request(&method, &host, port, &path, &headers, &body);
        }

        if name == "https_get" {
            if args.len() != 1 { return Err("Nano: https_get() recebe URL".into()); }
            let url = text_arg(&args[0], "URL")?;
            let rest = url.strip_prefix("https://")
                .ok_or_else(|| "Nano HTTPS: a URL deve começar por https://".to_string())?;
            let (authority, path) = rest.split_once('/').unwrap_or((rest, "/"));
            let (host, port) = authority.rsplit_once(':')
                .and_then(|(host, port)| port.parse::<u16>().ok().map(|port| (host, port)))
                .unwrap_or((authority, 443));
            return Ok(Value::Text(crate::tls::https_get(host, port, &format!("/{path}"))?));
        }

        if name == "sync_mutex_new" {
            if args.len() != 1 { return Err("Nano: sync.mutex_new() recebe valor".into()); }
            return sync_runtime::mutex_new(&args[0]).map(|id| Value::Number(id as f64));
        }
        if name == "sync_mutex_get" {
            if args.len() != 1 { return Err("Nano: sync.mutex_get() recebe handle".into()); }
            return sync_runtime::mutex_get(integer_arg(&args[0], "mutex")?);
        }
        if name == "sync_mutex_set" {
            if args.len() != 2 { return Err("Nano: sync.mutex_set() recebe handle e valor".into()); }
            sync_runtime::mutex_set(integer_arg(&args[0], "mutex")?, &args[1])?;
            return Ok(Value::Null);
        }
        if name == "sync_mutex_swap" {
            if args.len() != 2 { return Err("Nano: sync.mutex_swap() recebe handle e valor".into()); }
            return sync_runtime::mutex_swap(integer_arg(&args[0], "mutex")?, &args[1]);
        }
        if name == "sync_mutex_close" {
            if args.len() != 1 { return Err("Nano: sync.mutex_close() recebe handle".into()); }
            sync_runtime::sync_remove(integer_arg(&args[0], "mutex")?);
            return Ok(Value::Null);
        }
        if name == "sync_semaphore_new" {
            if args.len() != 1 { return Err("Nano: sync.semaphore_new() recebe contador".into()); }
            return Ok(Value::Number(sync_runtime::semaphore_new(integer_arg(&args[0], "contador")?) as f64));
        }
        if name == "sync_semaphore_acquire" {
            if args.len() != 1 && args.len() != 2 { return Err("Nano: sync.semaphore_acquire() recebe handle e timeout opcional".into()); }
            let timeout = args.get(1).map(|v| integer_arg(v, "timeout")).transpose()?;
            return sync_runtime::semaphore_acquire(integer_arg(&args[0], "semaphore")?, timeout).map(Value::Boolean);
        }
        if name == "sync_semaphore_release" {
            if args.len() != 1 { return Err("Nano: sync.semaphore_release() recebe handle".into()); }
            sync_runtime::semaphore_release(integer_arg(&args[0], "semaphore")?)?;
            return Ok(Value::Null);
        }
        if name == "sync_semaphore_close" {
            if args.len() != 1 { return Err("Nano: sync.semaphore_close() recebe handle".into()); }
            sync_runtime::semaphore_remove(integer_arg(&args[0], "semaphore")?);
            return Ok(Value::Null);
        }
        if name == "sync_channel_new" {
            if !args.is_empty() { return Err("Nano: sync.channel() não recebe argumentos".into()); }
            return Ok(Value::Number(sync_runtime::channel_new() as f64));
        }
        if name == "sync_channel_send" {
            if args.len() != 2 { return Err("Nano: sync.send() recebe canal e valor".into()); }
            sync_runtime::channel_send(integer_arg(&args[0], "canal")?, &args[1])?;
            return Ok(Value::Null);
        }
        if name == "sync_channel_recv" {
            if args.len() != 1 { return Err("Nano: sync.recv() recebe canal".into()); }
            return sync_runtime::channel_recv(integer_arg(&args[0], "canal")?, None);
        }
        if name == "sync_channel_recv_timeout" {
            if args.len() != 2 { return Err("Nano: sync.recv_timeout() recebe canal e timeout".into()); }
            return sync_runtime::channel_recv(integer_arg(&args[0], "canal")?, Some(integer_arg(&args[1], "timeout")?));
        }
        if name == "sync_channel_try_recv" {
            if args.len() != 1 { return Err("Nano: sync.try_recv() recebe canal".into()); }
            return Ok(match sync_runtime::channel_try_recv(integer_arg(&args[0], "canal")?)? {
                Some(value) => value,
                None => Value::Null,
            });
        }
        if name == "sync_channel_close" {
            if args.len() != 1 { return Err("Nano: sync.close_channel() recebe canal".into()); }
            sync_runtime::channel_close(integer_arg(&args[0], "canal")?);
            return Ok(Value::Null);
        }

        if name == "crypto_sha256" {
            if args.len() != 1 { return Err("Nano: crypto.sha256() recebe Text".into()); }
            return Ok(Value::Text(crate::crypto::sha256_hex(text_arg(&args[0], "texto")?.as_bytes())));
        }
        if name == "crypto_hmac_sha256" {
            if args.len() != 2 { return Err("Nano: crypto.hmac_sha256() recebe chave e mensagem".into()); }
            return Ok(Value::Text(crate::crypto::hmac_sha256_hex(
                text_arg(&args[0], "chave")?.as_bytes(),
                text_arg(&args[1], "mensagem")?.as_bytes(),
            )));
        }

        if name == "json_encode" {
            if args.len() != 1 { return Err("Nano: json_encode() recebe 1 valor".into()); }
            let json = value_to_json(&args[0])?;
            let text = serde_json::to_string(&json)
                .map_err(|e| format!("Nano: json_encode(): {e}"))?;
            return Ok(Value::Text(text));
        }

        if name == "json_decode" {
            if args.len() != 1 { return Err("Nano: json_decode() recebe 1 Text".into()); }
            let text = text_arg(&args[0], "JSON")?;
            let json = serde_json::from_str::<JsonValue>(&text)
                .map_err(|e| format!("Nano: json_decode(): {e}"))?;
            return json_to_value(json);
        }

        if name == "time_now_ms" {
            if !args.is_empty() { return Err("Nano: time_now_ms() não recebe argumentos".into()); }
            let now = SystemTime::now().duration_since(UNIX_EPOCH)
                .map_err(|e| format!("Nano: relógio do sistema: {e}"))?;
            return Ok(Value::Number(now.as_millis() as f64));
        }

        if name == "time_sleep_ms" || name == "thread_sleep_ms" {
            if args.len() != 1 { return Err(format!("Nano: {name}() recebe milissegundos")); }
            let ms = number_arg(&args[0], "milissegundos")?;
            if ms < 0.0 { return Err("Nano: duração não pode ser negativa".into()); }
            thread::sleep(Duration::from_millis(ms as u64));
            return Ok(Value::Null);
        }

        if name == "process_spawn" {
            if args.len() != 2 { return Err("Nano: process_spawn() recebe comando e lista de argumentos".into()); }
            let command = text_arg(&args[0], "comando")?;
            let argv = text_list_arg(&args[1], "argumentos")?;
            let child = Command::new(&command).args(argv).spawn()
                .map_err(|e| format!("Nano: não foi possível iniciar '{command}': {e}"))?;
            let handle = self.next_handle;
            self.next_handle += 1;
            self.children.insert(handle, child);
            return Ok(Value::Number(handle as f64));
        }

        if name == "thread_yield" {
            if !args.is_empty() { return Err("Nano: thread_yield() não recebe argumentos".into()); }
            thread::yield_now();
            return Ok(Value::Null);
        }

        if name == "thread_spawn" {
            if args.len() != 2 { return Err("Nano: thread_spawn() recebe comando e lista de argumentos".into()); }
            let command = text_arg(&args[0], "comando")?;
            let argv = text_list_arg(&args[1], "argumentos")?;
            let handle = self.next_handle;
            self.next_handle += 1;
            let join = thread::spawn(move || {
                Command::new(&command)
                    .args(argv)
                    .status()
                    .map(|status| TaskValue::Number(status.code().unwrap_or(-1) as f64))
                    .map_err(|e| format!("Nano: thread_spawn('{command}'): {e}"))
            });
            self.tasks.insert(handle, join);
            return Ok(Value::Number(handle as f64));
        }

        if name == "task_spawn" || name == "async_spawn" {
            if args.len() != 2 { return Err("Nano: task_spawn() recebe nome da função e lista de argumentos".into()); }
            let function_name = text_arg(&args[0], "função")?;
            let argv = match &args[1] {
                Value::List(values) => values.clone(),
                _ => return Err("Nano: task_spawn() requer List de argumentos".into()),
            };
            fn is_sendable(value: &Value) -> bool {
                match value {
                    Value::Number(_) | Value::Text(_) | Value::Boolean(_) | Value::Function(_) | Value::Null => true,
                    Value::List(items) => items.iter().all(is_sendable),
                    Value::Object(items) => items.values().all(is_sendable),
                    Value::Tensor(_) => false,
                }
            }
            if !argv.iter().all(is_sendable) {
                return Err("Nano: task_spawn() não aceita Tensor ou valores não thread-safe".into());
            }

            let function = self.functions.get(&function_name)
                .ok_or_else(|| format!("Nano: função '{function_name}' não definida"))?;
            if function.params.len() != argv.len() {
                return Err(format!(
                    "Nano: task_spawn('{function_name}') esperava {} argumentos",
                    function.params.len()
                ));
            }

            let mut task_functions = HashMap::new();
            for (name, function) in &self.functions {
                task_functions.insert(name.clone(), TaskFunction::from_ir(function)?);
            }
            let function_name_for_thread = function_name.clone();
            let backend_kind = self.backend.kind();
            let dtype = self.dtype;
            let task_args = argv.into_iter()
                .map(TaskValue::from_value)
                .collect::<Result<Vec<_>, _>>()?;

            let handle = self.next_handle;
            self.next_handle += 1;
            let join = thread::spawn(move || {
                let mut runtime = IrRuntime::with_backend_and_dtype(backend_kind, dtype)?;
                for (name, function) in task_functions {
                    runtime.functions.insert(name, function.into_ir());
                }
                let function = runtime.functions.get(&function_name_for_thread)
                    .cloned()
                    .ok_or_else(|| format!("Nano: tarefa não encontrou função '{function_name_for_thread}'"))?;
                for (param, value) in function.params.iter().zip(task_args) {
                    runtime.vars.insert(param.clone(), value.into_value());
                }
                runtime.execute_code(&function.code)
                    .and_then(|value| TaskValue::from_value(value.unwrap_or(Value::Null)))
            });
            self.tasks.insert(handle, join);
            return Ok(Value::Number(handle as f64));
        }

        if name == "async_sleep_ms" {
            if args.len() != 1 { return Err("Nano: async.sleep_ms() recebe milissegundos".into()); }
            let ms = number_arg(&args[0], "milissegundos")?;
            if ms < 0.0 { return Err("Nano: duração não pode ser negativa".into()); }
            let handle = self.next_handle;
            self.next_handle += 1;
            let join = thread::spawn(move || {
                thread::sleep(Duration::from_millis(ms as u64));
                Ok(TaskValue::Null)
            });
            self.tasks.insert(handle, join);
            return Ok(Value::Number(handle as f64));
        }

        if name == "async_select" {
            if args.len() != 1 { return Err("Nano: async.select() recebe uma lista de handles".into()); }
            let handles = match &args[0] {
                Value::List(values) => values.iter().map(|value| integer_arg(value, "handle")).collect::<Result<Vec<_>, _>>()?,
                _ => return Err("Nano: async.select() requer List".into()),
            };
            if handles.is_empty() { return Ok(Value::Null); }
            loop {
                for handle in &handles {
                    if let Some(task) = self.tasks.get(handle) {
                        if task.is_finished() {
                            return Ok(Value::Number(*handle as f64));
                        }
                    }
                }
                thread::sleep(Duration::from_millis(1));
            }
        }

        if name == "async_all" {
            if args.len() != 1 { return Err("Nano: async.all() recebe uma lista de handles".into()); }
            let handles = match &args[0] {
                Value::List(values) => values.iter().map(|value| integer_arg(value, "handle")).collect::<Result<Vec<_>, _>>()?,
                _ => return Err("Nano: async.all() requer List".into()),
            };
            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                let task = self.tasks.remove(&handle)
                    .ok_or_else(|| format!("Nano: tarefa {handle} não encontrada"))?;
                let value = task.join()
                    .map_err(|_| format!("Nano: async.all(): tarefa {handle} entrou em pânico"))?
                    .map_err(|error| format!("Nano: async.all(): {error}"))?;
                results.push(value.into_value());
            }
            return Ok(Value::List(results));
        }

        if name == "async_join" {
            if args.len() != 1 { return Err("Nano: async.join() recebe handle".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let task = self.tasks.remove(&handle)
                .ok_or_else(|| format!("Nano: tarefa {handle} não encontrada"))?;
            return task.join()
                .map_err(|_| format!("Nano: async.join(): tarefa {handle} entrou em pânico"))?
                .map(TaskValue::into_value);
        }

        if name == "thread_join" || name == "task_join" {
            if args.len() != 1 { return Err(format!("Nano: {name}() recebe handle")); }
            let handle = integer_arg(&args[0], "handle")?;
            let task = self.tasks.remove(&handle)
                .ok_or_else(|| format!("Nano: tarefa {handle} não encontrada"))?;
            return task.join()
                .map_err(|_| format!("Nano: {name}(): tarefa {handle} entrou em pânico"))?
                .map(TaskValue::into_value);
        }

        if name == "process_wait" {
            if args.len() != 1 { return Err("Nano: process_wait() recebe handle".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let mut child = self.children.remove(&handle).ok_or_else(|| format!("Nano: processo {handle} não encontrado"))?;
            let status = child.wait().map_err(|e| format!("Nano: process_wait(): {e}"))?;
            return Ok(Value::Number(status.code().unwrap_or(-1) as f64));
        }

        if name == "process_output" {
            if args.len() != 2 { return Err("Nano: process_output() recebe comando e lista de argumentos".into()); }
            let command = text_arg(&args[0], "comando")?;
            let argv = text_list_arg(&args[1], "argumentos")?;
            let output = Command::new(&command).args(argv).output()
                .map_err(|e| format!("Nano: process_output('{command}'): {e}"))?;
            let mut result = HashMap::new();
            result.insert("code".into(), Value::Number(output.status.code().unwrap_or(-1) as f64));
            result.insert("stdout".into(), Value::Text(String::from_utf8_lossy(&output.stdout).into_owned()));
            result.insert("stderr".into(), Value::Text(String::from_utf8_lossy(&output.stderr).into_owned()));
            return Ok(Value::Object(result));
        }

        if name == "os_args" {
            if !args.is_empty() { return Err("Nano: os_args() não recebe argumentos".into()); }
            return Ok(Value::List(std::env::args().map(Value::Text).collect()));
        }

        if name == "net_http_get" {
            if args.len() != 3 { return Err("Nano: net_http_get() recebe host, porta e caminho".into()); }
            let host = text_arg(&args[0], "host")?;
            let port = integer_arg(&args[1], "porta")?;
            let path = text_arg(&args[2], "caminho")?;
            let mut stream = TcpStream::connect((host.as_str(), port as u16))
                .map_err(|e| format!("Nano: HTTP connect: {e}"))?;
            let request = format!(
                "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: Nano/0.1\r\n\r\n"
            );
            stream.write_all(request.as_bytes())
                .map_err(|e| format!("Nano: HTTP send: {e}"))?;
            let mut response = String::new();
            stream.read_to_string(&mut response)
                .map_err(|e| format!("Nano: HTTP recv: {e}"))?;
            if let Some((headers, body)) = response.split_once("\r\n\r\n") {
                let status_ok = headers.lines().next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|code| code.parse::<u16>().ok())
                    .map(|code| (200..300).contains(&code))
                    .unwrap_or(false);
                if !status_ok {
                    let status = headers.lines().next().unwrap_or("HTTP");
                    return Err(format!("Nano: HTTP GET falhou: {status}"));
                }
                return Ok(Value::Text(body.to_string()));
            }
            return Err("Nano: resposta HTTP inválida".into());
        }

        if name == "net_tcp_connect" {
            if args.len() != 2 { return Err("Nano: net_tcp_connect() recebe host e porta".into()); }
            let host = text_arg(&args[0], "host")?;
            let port = integer_arg(&args[1], "porta")?;
            let stream = TcpStream::connect((host.as_str(), port as u16))
                .map_err(|e| format!("Nano: conexão TCP: {e}"))?;
            let handle = self.next_handle;
            self.next_handle += 1;
            self.tcp_streams.insert(handle, stream);
            return Ok(Value::Number(handle as f64));
        }

        if name == "net_tcp_listen" {
            if args.len() != 2 { return Err("Nano: net_tcp_listen() recebe host e porta".into()); }
            let host = text_arg(&args[0], "host")?;
            let port = integer_arg(&args[1], "porta")?;
            let listener = TcpListener::bind((host.as_str(), port as u16))
                .map_err(|e| format!("Nano: listener TCP: {e}"))?;
            let handle = self.next_handle;
            self.next_handle += 1;
            self.tcp_listeners.insert(handle, listener);
            return Ok(Value::Number(handle as f64));
        }

        if name == "net_tcp_accept" {
            if args.len() != 1 { return Err("Nano: net_tcp_accept() recebe listener".into()); }
            let handle = integer_arg(&args[0], "listener")?;
            let listener = self.tcp_listeners.get(&handle)
                .ok_or_else(|| format!("Nano: listener {handle} não encontrado"))?;
            let (stream, _) = listener.accept().map_err(|e| format!("Nano: accept(): {e}"))?;
            let stream_handle = self.next_handle;
            self.next_handle += 1;
            self.tcp_streams.insert(stream_handle, stream);
            return Ok(Value::Number(stream_handle as f64));
        }

        if name == "net_tcp_send" {
            if args.len() != 2 { return Err("Nano: net_tcp_send() recebe stream e texto".into()); }
            let handle = integer_arg(&args[0], "stream")?;
            let payload = text_arg(&args[1], "texto")?;
            let stream = self.tcp_streams.get_mut(&handle)
                .ok_or_else(|| format!("Nano: stream {handle} não encontrado"))?;
            let bytes = stream.write(payload.as_bytes()).map_err(|e| format!("Nano: send(): {e}"))?;
            return Ok(Value::Number(bytes as f64));
        }

        if name == "net_tcp_recv" {
            if args.len() != 2 { return Err("Nano: net_tcp_recv() recebe stream e máximo de bytes".into()); }
            let handle = integer_arg(&args[0], "stream")?;
            let max = integer_arg(&args[1], "máximo")?.max(1) as usize;
            let stream = self.tcp_streams.get_mut(&handle)
                .ok_or_else(|| format!("Nano: stream {handle} não encontrado"))?;
            let mut buf = vec![0u8; max];
            let bytes = stream.read(&mut buf).map_err(|e| format!("Nano: recv(): {e}"))?;
            buf.truncate(bytes);
            return Ok(Value::Text(String::from_utf8_lossy(&buf).into_owned()));
        }

        if name == "net_tcp_close" {
            if args.len() != 1 { return Err("Nano: net_tcp_close() recebe handle".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            self.tcp_streams.remove(&handle);
            self.tcp_listeners.remove(&handle);
            return Ok(Value::Null);
        }

        if name == "abs" || name == "sqrt" || name == "floor" || name == "ceil" || name == "round"
            || name == "sin" || name == "cos" || name == "tan" || name == "exp" || name == "log" {
            if args.len() != 1 { return Err(format!("Nano: {name}() recebe 1 Number")); }
            let value = number_arg(&args[0], "número")?;
            let output = match name {
                "abs" => value.abs(),
                "sqrt" => value.sqrt(),
                "floor" => value.floor(),
                "ceil" => value.ceil(),
                "round" => value.round(),
                "sin" => value.sin(),
                "cos" => value.cos(),
                "tan" => value.tan(),
                "exp" => value.exp(),
                _ => value.ln(),
            };
            return Ok(Value::Number(output));
        }

        if name == "pow" {
            if args.len() != 2 { return Err("Nano: pow() recebe base e expoente".into()); }
            let base = number_arg(&args[0], "base")?;
            let exponent = number_arg(&args[1], "expoente")?;
            return Ok(Value::Number(base.powf(exponent)));
        }

        if name == "min" || name == "max" {
            if args.len() != 2 { return Err(format!("Nano: {name}() recebe 2 Numbers")); }
            let a = number_arg(&args[0], "a")?;
            let b = number_arg(&args[1], "b")?;
            return Ok(Value::Number(if name == "min" { a.min(b) } else { a.max(b) }));
        }

        if name == "to_text" {
            if args.len() != 1 { return Err("Nano: to_text() recebe 1 argumento".into()); }
            return Ok(Value::Text(args[0].show()));
        }

        if name == "to_number" {
            if args.len() != 1 { return Err("Nano: to_number() recebe 1 Text".into()); }
            let value = text_arg(&args[0], "texto")?;
            let number = value.parse::<f64>().map_err(|_| format!("Nano: to_number(): '{value}' não é Number"))?;
            return Ok(Value::Number(number));
        }

        if matches!(name, "upper" | "lower" | "trim") {
            if args.len() != 1 { return Err(format!("Nano: {name}() recebe 1 Text")); }
            let value = text_arg(&args[0], "texto")?;
            let output = match name {
                "upper" => value.to_uppercase(),
                "lower" => value.to_lowercase(),
                _ => value.trim().to_string(),
            };
            return Ok(Value::Text(output));
        }

        if matches!(name, "contains" | "starts_with" | "ends_with") {
            if args.len() != 2 { return Err(format!("Nano: {name}() recebe 2 Text")); }
            let value = text_arg(&args[0], "texto")?;
            let needle = text_arg(&args[1], "texto")?;
            let result = match name {
                "contains" => value.contains(&needle),
                "starts_with" => value.starts_with(&needle),
                _ => value.ends_with(&needle),
            };
            return Ok(Value::Boolean(result));
        }

        if name == "replace" {
            if args.len() != 3 { return Err("Nano: replace() recebe texto, antigo e novo".into()); }
            let value = text_arg(&args[0], "texto")?;
            let from = text_arg(&args[1], "antigo")?;
            let to = text_arg(&args[2], "novo")?;
            return Ok(Value::Text(value.replace(&from, &to)));
        }

        if name == "substring" {
            if args.len() != 3 { return Err("Nano: substring() recebe texto, início e fim".into()); }
            let value = text_arg(&args[0], "texto")?;
            let start = integer_arg(&args[1], "início")? as usize;
            let end = integer_arg(&args[2], "fim")? as usize;
            let chars: Vec<char> = value.chars().collect();
            if start > end || end > chars.len() {
                return Err("Nano: substring() possui limites fora do texto".into());
            }
            return Ok(Value::Text(chars[start..end].iter().collect()));
        }

        if name == "char_at" {
            if args.len() != 2 { return Err("Nano: char_at() recebe texto e índice".into()); }
            let value = text_arg(&args[0], "texto")?;
            let index = integer_arg(&args[1], "índice")? as usize;
            let ch = value.chars().nth(index)
                .ok_or_else(|| "Nano: char_at() índice fora do limite".to_string())?;
            return Ok(Value::Text(ch.to_string()));
        }

        if name == "split" {
            if args.len() != 2 { return Err("Nano: split() recebe texto e separador".into()); }
            let value = text_arg(&args[0], "texto")?;
            let separator = text_arg(&args[1], "separador")?;
            return Ok(Value::List(value.split(&separator).map(|v| Value::Text(v.to_string())).collect()));
        }

        if name == "join" {
            if args.len() != 2 { return Err("Nano: join() recebe lista e separador".into()); }
            let items = match &args[0] {
                Value::List(items) => items,
                _ => return Err("Nano: join() requer List".into()),
            };
            let separator = text_arg(&args[1], "separador")?;
            return Ok(Value::Text(items.iter().map(Value::show).collect::<Vec<_>>().join(&separator)));
        }

        if name == "append" {
            if args.len() != 2 { return Err("Nano: append() recebe lista e valor".into()); }
            let mut items = match &args[0] {
                Value::List(items) => items.clone(),
                _ => return Err("Nano: append() requer List".into()),
            };
            items.push(args[1].clone());
            return Ok(Value::List(items));
        }

        if name == "ui_window" {
            if args.len() != 3 { return Err("Nano: ui_window() recebe título, largura e altura".into()); }
            let title = text_arg(&args[0], "título")?;
            let width = number_arg(&args[1], "largura")?;
            let height = number_arg(&args[2], "altura")?;
            let handle = self.next_handle;
            self.next_handle += 1;
            let window = ui::spawn(title, width, height)?;
            self.ui_windows.insert(handle, window);
            return Ok(Value::Number(handle as f64));
        }

        if name == "ui_clear" {
            if args.len() != 4 { return Err("Nano: ui.clear() recebe r,g,b,a".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let color = [number_arg(&args[1], "r")? as f32, number_arg(&args[2], "g")? as f32, number_arg(&args[3], "b")? as f32];
            let window = self.ui_windows.get(&handle).ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            window.command.send(ui::UiCommand::Clear([color[0], color[1], color[2], 1.0]))
                .map_err(|_| "Nano: thread da UI não está disponível".to_string())?;
            return Ok(Value::Null);
        }
        if name == "ui_rect" {
            if args.len() != 8 { return Err("Nano: ui.rect() recebe handle,x,y,width,height,r,g,b".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let rect = ui::UiCommand::Rect {
                x: number_arg(&args[1], "x")? as f32,
                y: number_arg(&args[2], "y")? as f32,
                width: number_arg(&args[3], "width")? as f32,
                height: number_arg(&args[4], "height")? as f32,
                color: [
                    number_arg(&args[5], "r")? as f32,
                    number_arg(&args[6], "g")? as f32,
                    number_arg(&args[7], "b")? as f32,
                    1.0,
                ],
            };
            let window = self.ui_windows.get(&handle).ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            window.command.send(rect).map_err(|_| "Nano: thread da UI não está disponível".to_string())?;
            return Ok(Value::Null);
        }
        if name == "ui_button" {
            if args.len() != 9 { return Err("Nano: ui.button() recebe handle,id,x,y,width,height,r,g,b".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let id = text_arg(&args[1], "id")?;
            let button = ui::UiCommand::Button {
                id,
                x: number_arg(&args[2], "x")? as f32,
                y: number_arg(&args[3], "y")? as f32,
                width: number_arg(&args[4], "width")? as f32,
                height: number_arg(&args[5], "height")? as f32,
                color: [
                    number_arg(&args[6], "r")? as f32,
                    number_arg(&args[7], "g")? as f32,
                    number_arg(&args[8], "b")? as f32,
                    1.0,
                ],
            };
            let window = self.ui_windows.get(&handle).ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            window.command.send(button).map_err(|_| "Nano: thread da UI não está disponível".to_string())?;
            return Ok(Value::Null);
        }

        if name == "ui_text" {
            if args.len() != 8 { return Err("Nano: ui.text() recebe handle,text,x,y,size,r,g,b".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let command = ui::UiCommand::Text {
                text: text_arg(&args[1], "texto")?,
                x: number_arg(&args[2], "x")? as f32,
                y: number_arg(&args[3], "y")? as f32,
                size: number_arg(&args[4], "tamanho")?.max(1.0) as f32,
                color: [
                    number_arg(&args[5], "r")? as f32,
                    number_arg(&args[6], "g")? as f32,
                    number_arg(&args[7], "b")? as f32,
                    1.0,
                ],
            };
            let window = self.ui_windows.get(&handle)
                .ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            window.command.send(command)
                .map_err(|_| "Nano: thread da UI não está disponível".to_string())?;
            return Ok(Value::Null);
        }

        if name == "ui_vbox" {
            if args.len() != 7 { return Err("Nano: ui.vbox() recebe handle,x,y,width,row_height,gap,children".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let x = number_arg(&args[1], "x")? as f32;
            let y = number_arg(&args[2], "y")? as f32;
            let width = number_arg(&args[3], "width")? as f32;
            let row_height = number_arg(&args[4], "row_height")?.max(1.0) as f32;
            let gap = number_arg(&args[5], "gap")?.max(0.0) as f32;
            let children = match &args[6] {
                Value::List(values) => {
                    let mut specs = Vec::with_capacity(values.len());
                    for value in values {
                        let Value::Object(object) = value else {
                            return Err("Nano: ui.vbox() requer uma lista de Objects".into());
                        };
                        let kind = object.get("kind")
                            .and_then(|value| match value { Value::Text(text) => Some(text.as_str()), _ => None })
                            .ok_or_else(|| "Nano: ui.vbox() exige campo 'kind'".to_string())?;
                        let number = |name: &str, default: f64| -> f32 {
                            match object.get(name) {
                                Some(Value::Number(value)) => *value as f32,
                                _ => default as f32,
                            }
                        };
                        let color = [number("r", 0.2), number("g", 0.3), number("b", 0.5), 1.0];
                        match kind {
                            "text" => {
                                let text = object.get("text")
                                    .and_then(|value| match value { Value::Text(text) => Some(text.clone()), _ => None })
                                    .unwrap_or_default();
                                specs.push(ui::UiWidgetSpec::Text {
                                    text,
                                    size: number("size", 16.0).max(1.0),
                                    color,
                                });
                            }
                            "button" => {
                                let id = object.get("id")
                                    .and_then(|value| match value { Value::Text(text) => Some(text.clone()), _ => None })
                                    .unwrap_or_else(|| "button".into());
                                let label = object.get("label")
                                    .and_then(|value| match value { Value::Text(text) => Some(text.clone()), _ => None })
                                    .unwrap_or_else(|| id.clone());
                                specs.push(ui::UiWidgetSpec::Button { id, label, color });
                            }
                            "rect" => specs.push(ui::UiWidgetSpec::Rect { color }),
                            _ => return Err(format!("Nano: ui.vbox() widget desconhecido '{kind}'")),
                        }
                    }
                    specs
                }
                _ => return Err("Nano: ui.vbox() requer uma lista".into()),
            };
            let window = self.ui_windows.get(&handle)
                .ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            window.command.send(ui::UiCommand::VBox { x, y, width, row_height, gap, children })
                .map_err(|_| "Nano: thread da UI não está disponível".to_string())?;
            return Ok(Value::Null);
        }

        if name == "ui_set_title" {
            if args.len() != 2 { return Err("Nano: ui_set_title() recebe handle e título".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let title = text_arg(&args[1], "título")?;
            let window = self.ui_windows.get(&handle)
                .ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            window.command.send(ui::UiCommand::SetTitle(title))
                .map_err(|_| "Nano: thread da UI não está disponível".to_string())?;
            return Ok(Value::Null);
        }

        if name == "ui_close" {
            if args.len() != 1 { return Err("Nano: ui_close() recebe handle".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            if let Some(window) = self.ui_windows.remove(&handle) {
                let _ = window.command.send(ui::UiCommand::Close);
            }
            return Ok(Value::Null);
        }

        if name == "ui_wait_event" {
            if args.len() != 1 { return Err("Nano: ui_wait_event() recebe handle".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let window = self.ui_windows.get(&handle)
                .ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            return window.events.recv()
                .map(Value::Text)
                .map_err(|_| "Nano: thread da UI não está disponível".to_string());
        }

        if name == "ui_poll_event" {
            if args.len() != 1 { return Err("Nano: ui_poll_event() recebe handle".into()); }
            let handle = integer_arg(&args[0], "handle")?;
            let window = self.ui_windows.get(&handle)
                .ok_or_else(|| format!("Nano: janela {handle} não encontrada"))?;
            return Ok(match window.events.try_recv() {
                Ok(event) => Value::Text(event),
                Err(_) => Value::Text(String::new()),
            });
        }

        if name == "ok" {
            if args.len() != 1 { return Err("Nano: ok() recebe 1 valor".into()); }
            let mut object = HashMap::new();
            object.insert("ok".into(), Value::Boolean(true));
            object.insert("value".into(), args[0].clone());
            return Ok(Value::Object(object));
        }

        if name == "err" {
            if args.len() != 1 { return Err("Nano: err() recebe 1 mensagem".into()); }
            let message = text_arg(&args[0], "mensagem")?;
            let mut object = HashMap::new();
            object.insert("ok".into(), Value::Boolean(false));
            object.insert("error".into(), Value::Text(message));
            return Ok(Value::Object(object));
        }

        if name == "is_ok" {
            if args.len() != 1 { return Err("Nano: is_ok() recebe 1 Result".into()); }
            return Ok(Value::Boolean(matches!(
                &args[0],
                Value::Object(object) if matches!(object.get("ok"), Some(Value::Boolean(true)))
            )));
        }

        if name == "unwrap" {
            if args.len() != 1 { return Err("Nano: unwrap() recebe 1 Result".into()); }
            let object = match &args[0] {
                Value::Object(object) => object,
                _ => return Err("Nano: unwrap() requer Result".into()),
            };
            return match object.get("ok") {
                Some(Value::Boolean(true)) => object.get("value").cloned().ok_or_else(|| "Nano: Result ok sem value".into()),
                Some(Value::Boolean(false)) => {
                    let message = object.get("error").map(Value::show).unwrap_or_else(|| "erro Nano desconhecido".into());
                    Err(format!("Nano: unwrap(): {message}"))
                }
                _ => Err("Nano: objeto não é um Result válido".into()),
            };
        }

        if name == "error" {
            if args.len() != 1 { return Err("Nano: error() recebe 1 Result".into()); }
            let object = match &args[0] {
                Value::Object(object) => object,
                _ => return Err("Nano: error() requer Result".into()),
            };
            return Ok(Value::Text(object.get("error").map(Value::show).unwrap_or_default()));
        }

        if name == "assert" {
            if args.len() != 1 && args.len() != 2 {
                return Err("Nano: assert() recebe condição e, opcionalmente, mensagem".into());
            }
            if !args[0].truthy() {
                let detail = if args.len() == 2 { format!(": {}", args[1].show()) } else { String::new() };
                return Err(format!("Nano assertion failed{detail}"));
            }
            return Ok(Value::Null);
        }

        if name == "channel" || name == "std.async.channel" {
            if !args.is_empty() { return Err("Nano: channel() não recebe argumentos".into()); }
            let (tx, rx) = mpsc::channel();
            let handle = self.next_handle;
            self.next_handle += 1;
            self.channels.insert(handle, (tx, rx));
            return Ok(Value::Number(handle as f64));
        }

        if name == "send" || name == "std.async.send" {
            if args.len() != 2 { return Err("Nano: send() recebe canal e valor".into()); }
            let handle = integer_arg(&args[0], "canal")?;
            let (tx, _) = self.channels.get(&handle)
                .ok_or_else(|| format!("Nano: canal {handle} não encontrado"))?;
            tx.send(args[1].clone()).map_err(|_| format!("Nano: canal {handle} foi fechado"))?;
            return Ok(Value::Null);
        }

        if name == "recv" || name == "std.async.recv" {
            if args.len() != 1 { return Err("Nano: recv() recebe canal".into()); }
            let handle = integer_arg(&args[0], "canal")?;
            let (_, rx) = self.channels.get_mut(&handle)
                .ok_or_else(|| format!("Nano: canal {handle} não encontrado"))?;
            return rx.recv().map_err(|_| format!("Nano: canal {handle} foi fechado"));
        }

        if name == "close_channel" || name == "std.async.close_channel" {
            if args.len() != 1 { return Err("Nano: close_channel() recebe canal".into()); }
            let handle = integer_arg(&args[0], "canal")?;
            self.channels.remove(&handle);
            return Ok(Value::Null);
        }

        if name == "len" {
            if args.len() != 1 {
                return Err("Nano: len() recebe 1 argumento".into());
            }
            return match &args[0] {
                Value::Text(v) => Ok(Value::Number(v.chars().count() as f64)),
                Value::List(v) => Ok(Value::Number(v.len() as f64)),
                Value::Object(v) => Ok(Value::Number(v.len() as f64)),
                Value::Tensor(v) => Ok(Value::Number(v.borrow().data_len() as f64)),
                _ => Err("Nano: len() requer texto, lista, objeto ou tensor".into()),
            };
        }

        if name == "dtype" {
            if args.len() != 1 { return Err("Nano: dtype() recebe 1 tensor".into()); }
            return match &args[0] {
                Value::Tensor(t) => Ok(Value::Text(t.borrow().dtype.name().into())),
                _ => Err("Nano: dtype() requer Tensor".into()),
            };
        }

        if name == "memory_bytes" {
            if args.len() != 1 { return Err("Nano: memory_bytes() recebe 1 tensor".into()); }
            return match &args[0] {
                Value::Tensor(t) => Ok(Value::Number(t.borrow().memory_bytes() as f64)),
                _ => Err("Nano: memory_bytes() requer Tensor".into()),
            };
        }

        if name == "cast" {
            if args.len() != 2 { return Err("Nano: cast() recebe tensor e dtype".into()); }
            let tensor = match &args[0] {
                Value::Tensor(t) => std::rc::Rc::clone(t),
                _ => return Err("Nano: cast() requer Tensor".into()),
            };
            let dtype = match &args[1] {
                Value::Text(value) => DType::parse(value)?,
                _ => return Err("Nano: cast() requer dtype Text".into()),
            };
            self.sync_tensor_host(&tensor)?;
            let source = tensor.borrow();
            let data = source.data_f32();
            let shape = source.shape.clone();
            let requires_grad = source.requires_grad;
            let device = source.device;
            let op = source.op.clone();
            drop(source);
            let output=super::Tensor::derived_dtype_on(data.clone(),shape,requires_grad,device,dtype,op)?;
            let id=output.borrow().id;
            self.backend.sync_tensor(id,&data).map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
            return Ok(Value::Tensor(output));
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
            let tensor=super::Tensor::new_with_dtype_on(data.clone(),shape,false,self.backend.kind(),self.dtype)?;
            let id=tensor.borrow().id;
            self.backend.sync_tensor(id,&data).map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
            return Ok(Value::Tensor(tensor));
        }

        if name == "parameter" {
            if args.len() != 2 {
                return Err("Nano: parameter() recebe dados e shape".into());
            }
            let data = list_numbers(&args[0], "dados")?;
            let shape = list_shape(&args[1])?;
            let tensor=super::Tensor::new_with_dtype_on(data.clone(),shape,true,self.backend.kind(),self.dtype)?;
            let id=tensor.borrow().id;
            self.backend.sync_tensor(id,&data).map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
            return Ok(Value::Tensor(tensor));
        }

        if name == "zeros" {
            if args.len() != 1 {
                return Err("Nano: zeros() recebe shape".into());
            }
            let shape = list_shape(&args[0])?;
            let size = shape.iter().copied().product::<usize>();
            let data=vec![0.0;size];
            let tensor=super::Tensor::new_with_dtype_on(data.clone(),shape,false,self.backend.kind(),self.dtype)?;
            let id=tensor.borrow().id;
            self.backend.sync_tensor(id,&data).map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
            return Ok(Value::Tensor(tensor));
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

        if name == "matmul_transposed" {
            if args.len() != 4 {
                return Err("Nano: matmul_transposed() recebe A, B, transposeA, transposeB".into());
            }
            let (left, right) = match (&args[0], &args[1]) {
                (Value::Tensor(a), Value::Tensor(b)) => (a, b),
                _ => return Err("Nano: matmul_transposed() requer Tensor, Tensor, Boolean, Boolean".into()),
            };
            let left_transpose = match args[2] { Value::Boolean(v) => v, _ => return Err("Nano: transposeA deve ser Boolean".into()) };
            let right_transpose = match args[3] { Value::Boolean(v) => v, _ => return Err("Nano: transposeB deve ser Boolean".into()) };
            let (lshape, rshape, requires_grad, dtype) = {
                let l = left.borrow();
                let r = right.borrow();
                if l.device != self.backend.kind() || r.device != self.backend.kind() {
                    return Err("Nano: matmul_transposed requer tensors no backend ativo".into());
                }
                (l.shape.clone(), r.shape.clone(), l.requires_grad || r.requires_grad, l.dtype)
            };
            let data = self.sync_tensor_host(left).and_then(|_| self.sync_tensor_host(right))?;
            let _ = data;
            let left_data = left.borrow().data_f32();
            let right_data = right.borrow().data_f32();
            let output = self.backend.matmul_transposed(&left_data, &lshape, &right_data, &rshape, left_transpose, right_transpose)
                .map_err(|e| format!("Nano: backend {}: {e}", self.backend.kind().name()))?;
            let (m, _k) = if left_transpose { (lshape[1], lshape[0]) } else { (lshape[0], lshape[1]) };
            let (_k2, n) = if right_transpose { (rshape[1], rshape[0]) } else { (rshape[0], rshape[1]) };
            let tensor = super::Tensor::derived_dtype_on(output, vec![m, n], requires_grad, self.backend.kind(), dtype, TensorOp::MatmulTransposed(left.clone(), right.clone(), left_transpose, right_transpose))?;
            return Ok(Value::Tensor(tensor));
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
            return gradient_value(&args[0], &args[1], self.backend.as_ref());
        }

        if name == "step" {
            if args.len() != 3 {
                return Err("Nano: step() recebe parâmetro, gradiente e taxa".into());
            }
            return step_value(&args[0], &args[1], &args[2], self.backend.as_ref());
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
            if p.device != g.device {
                return Err("Nano: parâmetro e gradiente precisam estar no mesmo dispositivo".into());
            }
            (p.id, p.data_len())
        };

        if self.backend.kind()==backend::BackendKind::Gpu {
            let state=self.adam.entry(id).or_insert_with(||AdamState{step:0,m:Vec::new(),v:Vec::new()});
            state.step+=1;
            self.backend.adam_resident_async(id,grad.borrow().id,len,lr,state.step as u32)
                .map_err(|e|format!("Nano: GPU Adam: {e}"))?;
            param.borrow_mut().mark_host_stale();
            return Ok(Value::Tensor(param));
        }

        self.sync_tensor_host(&param)?;
        self.sync_tensor_host(&grad)?;
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

        let mut data = param.borrow().data_f32();
        let gradient = grad.borrow().data_f32();
        for i in 0..data.len() {
            state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * gradient[i];
            state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * gradient[i] * gradient[i];
            let m_hat = state.m[i] / (1.0 - beta1.powf(t));
            let v_hat = state.v[i] / (1.0 - beta2.powf(t));
            data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        let id=param.borrow().id;
        param.borrow_mut().set_data_f32(data.clone());
        self.backend.sync_tensor(id,&data)
            .map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;

        Ok(Value::Tensor(std::rc::Rc::clone(&param)))
    }

    fn fused_mul_add_value(&mut self, left: Value, right: Value, bias: Value) -> Result<Value, String> {
        let (left_ref, right_ref, bias_ref) = match (left, right, bias) {
            (Value::Tensor(a), Value::Tensor(b), Value::Tensor(c)) => (a, b, c),
            _ => return Err("Nano: FMA requer Tensor, Tensor, Tensor".into()),
        };

        let a = left_ref.borrow();
        let b = right_ref.borrow();
        let c = bias_ref.borrow();
        if a.shape != b.shape || a.shape != c.shape {
            return Err("Nano: FMA requer tensors com shapes iguais".into());
        }
        if a.device != self.backend.kind() || b.device != self.backend.kind() || c.device != self.backend.kind() {
            return Err("Nano: FMA requer tensors no dispositivo do backend ativo".into());
        }

        let shape=a.shape.clone();
        let dtype=DType::promote(DType::promote(a.dtype,b.dtype),c.dtype);
        let requires_grad=a.requires_grad||b.requires_grad||c.requires_grad;
        let output_id=super::next_tensor_id();
        let op=TensorOp::FusedMulAdd(std::rc::Rc::clone(&left_ref),std::rc::Rc::clone(&right_ref),std::rc::Rc::clone(&bias_ref));
        if self.backend.kind()==backend::BackendKind::Gpu {
            self.backend.fused_mul_add_resident_async(a.id,b.id,c.id,&shape,output_id)
                .map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
            let out=super::Tensor::remote_with_id(output_id,shape.clone(),requires_grad,self.backend.kind(),dtype,op)?;
            out.borrow_mut().mark_host_stale();
            return Ok(Value::Tensor(out));
        }
        let data=self.backend.fused_mul_add(&a.data_f32(),&b.data_f32(),&c.data_f32(),&shape)
            .map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
        drop(a);drop(b);drop(c);
        Ok(Value::Tensor(super::Tensor::derived_dtype_on_with_id(output_id,data,shape,requires_grad,self.backend.kind(),dtype,op)?))
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
                let source=tensor.borrow();
                if source.device==backend::BackendKind::Gpu {
                    let output_id=super::next_tensor_id();
                    let shape=source.shape.clone();
                    let dtype=source.dtype;
                    let requires_grad=source.requires_grad;
                    let elements=source.data_len();
                    let input_id=source.id;
                    self.backend.scale_resident_async(input_id,elements,*n as f32,output_id)
                        .map_err(|e|format!("Nano: backend {}: {}",self.backend.kind().name(),e))?;
                    let out=super::Tensor::remote_with_id(
                        output_id,shape,requires_grad,self.backend.kind(),dtype,TensorOp::Leaf
                    )?;
                    return Ok(Value::Tensor(out));
                }
                let data=source.data_f32().into_iter().map(|v|v*(*n as f32)).collect();
                let out=super::Tensor::derived_dtype_on_with_id(
                    super::next_tensor_id(),data,source.shape.clone(),source.requires_grad,
                    self.backend.kind(),source.dtype,TensorOp::Leaf
                )?;
                Ok(Value::Tensor(out))
            }
            _ => binary(a, op, b),
        }
    }

    fn load_module(&mut self, path: &str) -> Result<(), String> {
        if path.starts_with("std.") {
            self.loaded_modules.insert(PathBuf::from(path));
            return Ok(());
        }
        let requested = Path::new(path);
        let resolved = if requested.is_absolute() {
            requested.to_path_buf()
        } else if let Some(parent) = self.module_stack.last() {
            parent.parent().unwrap_or_else(|| Path::new(".")).join(requested)
        } else {
            requested.to_path_buf()
        };

        let canonical = fs::canonicalize(&resolved)
            .map_err(|e| format!("Nano: não foi possível localizar módulo '{path}': {e}"))?;

        if self.loaded_modules.contains(&canonical) {
            return Ok(());
        }

        if let Some(start) = self.module_stack.iter().position(|item| item == &canonical) {
            let mut chain = self.module_stack[start..]
                .iter()
                .map(|item| item.display().to_string())
                .collect::<Vec<_>>();
            chain.push(canonical.display().to_string());
            return Err(format!("Nano: ciclo de módulos detectado: {}", chain.join(" -> ")));
        }

        self.module_stack.push(canonical.clone());

        let result = (|| {
            let source = fs::read_to_string(&canonical)
                .map_err(|e| format!("Nano: não foi possível carregar módulo '{}': {e}", canonical.display()))?;
            let tokens = Lexer::new(&source).lex()?;
            let program = Parser::new(tokens).program()?;

            let mut semantic = Semantic::new();
            semantic.check(&program)?;

            let mut compiler = Compiler::new();
            let module = compiler.compile(&program)?;
            self.functions.extend(module.functions);

            let _ = self.execute_code(&module.code)?;
            Ok(())
        })();

        self.module_stack.pop();

        if result.is_ok() {
            self.loaded_modules.insert(canonical);
        }

        result
    }
}


fn value_to_json(value: &Value) -> Result<JsonValue, String> {
    Ok(match value {
        Value::Number(v) => JsonValue::from(*v),
        Value::Text(v) => JsonValue::from(v.clone()),
        Value::Boolean(v) => JsonValue::from(*v),
        Value::Null => JsonValue::Null,
        Value::List(values) => JsonValue::Array(
            values.iter().map(value_to_json).collect::<Result<Vec<_>, _>>()?
        ),
        Value::Object(values) => {
            let mut object = serde_json::Map::new();
            for (key, item) in values {
                object.insert(key.clone(), value_to_json(item)?);
            }
            JsonValue::Object(object)
        }
        Value::Function(_) | Value::Tensor(_) => {
            return Err("Nano: json_encode() não aceita Function ou Tensor".into());
        }
    })
}

fn json_to_value(value: JsonValue) -> Result<Value, String> {
    Ok(match value {
        JsonValue::Null => Value::Null,
        JsonValue::Bool(v) => Value::Boolean(v),
        JsonValue::Number(v) => Value::Number(
            v.as_f64().ok_or_else(|| "Nano: número JSON fora do formato Number".to_string())?
        ),
        JsonValue::String(v) => Value::Text(v),
        JsonValue::Array(values) => Value::List(
            values.into_iter().map(json_to_value).collect::<Result<Vec<_>, _>>()?
        ),
        JsonValue::Object(values) => {
            let mut object = HashMap::new();
            for (key, item) in values {
                object.insert(key, json_to_value(item)?);
            }
            Value::Object(object)
        }
    })
}
fn object_text_map(value: &Value, label: &str) -> Result<HashMap<String, String>, String> {
    let object = match value {
        Value::Object(values) => values,
        _ => return Err(format!("Nano: {label} requer Object")),
    };
    let mut result = HashMap::new();
    for (key, value) in object {
        result.insert(key.clone(), text_arg(value, "header")?);
    }
    Ok(result)
}

fn text_arg(value: &Value, label: &str) -> Result<String, String> {
    match value { Value::Text(v) => Ok(v.clone()), _ => Err(format!("Nano: {label} requer Text")) }
}

fn number_arg(value: &Value, label: &str) -> Result<f64, String> {
    match value { Value::Number(v) => Ok(*v), _ => Err(format!("Nano: {label} requer Number")) }
}

fn integer_arg(value: &Value, label: &str) -> Result<u64, String> {
    let n = number_arg(value, label)?;
    if n < 0.0 || !n.is_finite() || n.fract() != 0.0 { return Err(format!("Nano: {label} requer inteiro não negativo")); }
    Ok(n as u64)
}

fn text_list_arg(value: &Value, label: &str) -> Result<Vec<String>, String> {
    match value {
        Value::List(items) => items.iter().map(|item| text_arg(item, label)).collect(),
        _ => Err(format!("Nano: {label} requer List de Text")),
    }
}

fn is_builtin_name(name: &str) -> bool {
    matches!(name,
        "print" | "assert" | "ok" | "err" | "is_ok" | "unwrap" | "error"
        | "len" | "range" | "to_text" | "to_number" | "upper" | "lower" | "trim"
        | "contains" | "starts_with" | "ends_with" | "replace" | "substring" | "char_at"
        | "split" | "join" | "append"
        | "abs" | "sqrt" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan" | "exp"
        | "log" | "pow" | "min" | "max"
        | "tensor" | "parameter" | "zeros" | "shape" | "matmul" | "sum" | "mean" | "grad"
        | "step" | "adam" | "cast" | "dtype" | "device" | "memory_bytes" | "backend"
        | "fs_read_text" | "fs_write_text" | "fs_append_text" | "fs_exists" | "fs_list"
        | "fs_mkdir" | "fs_remove"
        | "env_get" | "env_set" | "process_spawn" | "process_wait"
        | "time_now_ms" | "time_sleep_ms" | "thread_sleep_ms" | "thread_spawn" | "thread_join"
        | "task_spawn" | "task_join" | "async_spawn" | "async_join" | "async_select" | "async_all" | "async_sleep_ms"
        | "channel" | "send" | "recv" | "close_channel"
        | "net_tcp_connect" | "net_tcp_listen" | "net_tcp_accept" | "net_tcp_send"
        | "net_tcp_recv" | "net_tcp_close" | "net_http_get"
        | "ui_window" | "ui_set_title" | "ui_close" | "ui_poll_event" | "ui_text" | "ui_vbox"
        | "std.fs.read_text" | "std.fs.write_text" | "std.fs.append_text" | "std.fs.exists"
        | "std.fs.list" | "std.fs.mkdir" | "std.fs.remove"
        | "std.process.spawn" | "std.process.wait"
        | "std.time.now_ms" | "std.time.sleep_ms"
        | "std.net.tcp_connect" | "std.net.tcp_listen" | "std.net.tcp_accept"
        | "std.net.tcp_send" | "std.net.tcp_recv" | "std.net.tcp_close"
        | "std.net.http_get" | "std.http.get"
        | "std.env.get" | "std.env.set"
        | "std.math.abs" | "std.math.sqrt" | "std.math.floor" | "std.math.ceil" | "std.math.round"
        | "std.math.sin" | "std.math.cos" | "std.math.tan" | "std.math.exp" | "std.math.log"
        | "std.math.pow" | "std.math.min" | "std.math.max"
        | "std.ui.window" | "std.ui.set_title" | "std.ui.close" | "std.ui.poll_event"
        | "std.async.channel" | "std.async.send" | "std.async.recv" | "std.async.close_channel"
    )
}

fn pop_n(stack: &mut Vec<Value>, count: usize) -> Result<Vec<Value>, String> {
    if stack.len() < count {
        return Err("Nano IR: stack insuficiente".into());
    }
    let start = stack.len() - count;
    let mut values: Vec<Value> = stack.drain(start..).collect();
    values.reverse();
    Ok(values)
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
        Op::Mod => num(a, b, |x, y| x % y),
        Op::Eq => Ok(Value::Boolean(a == b)),
        Op::Ne => Ok(Value::Boolean(a != b)),
        Op::Gt => cmp(a, b, |x, y| x > y),
        Op::Ge => cmp(a, b, |x, y| x >= y),
        Op::Lt => cmp(a, b, |x, y| x < y),
        Op::Le => cmp(a, b, |x, y| x <= y),
        Op::And => Ok(Value::Boolean(a.truthy() && b.truthy())),
        Op::Or => Ok(Value::Boolean(a.truthy() || b.truthy())),
    }
}

fn unary_value(value: Value, op: crate::UnaryOp) -> Result<Value, String> {
    match op {
        crate::UnaryOp::Neg => match value {
            Value::Number(v) => Ok(Value::Number(-v)),
            _ => Err("Nano: menos unário requer Number".into()),
        },
        crate::UnaryOp::Not => Ok(Value::Boolean(!value.truthy())),
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
        (Value::Tensor(tensor), Value::Number(n)) => {
            if n < 0.0 || n.fract() != 0.0 {
                return Err("Nano: índice de Tensor deve ser um número inteiro".into());
            }
            let tensor = tensor.borrow();
            tensor.data_f32().get(n as usize)
                .map(|value| Value::Number(*value as f64))
                .ok_or_else(|| "Nano: índice de Tensor fora do limite".into())
        }
        (Value::Object(values), Value::Text(key)) => values.get(&key).cloned()
            .ok_or_else(|| format!("Nano: chave '{key}' não existe")),
        _ => Err("Nano: indexação requer lista[número], Tensor[número] ou objeto[texto]".into()),
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

    let (lshape, rshape, requires_grad, ldtype, rdtype) = {
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
        (l.shape.clone(), r.shape.clone(), l.requires_grad || r.requires_grad, l.dtype, r.dtype)
    };

    if lshape.len() != 2 || rshape.len() != 2 {
        return Err("Nano: matmul() nesta versão requer tensores 2D".into());
    }

    let (m, k) = (lshape[0], lshape[1]);
    let (k2, n) = (rshape[0], rshape[1]);
    if k != k2 {
        return Err(format!("Nano: matmul() incompatível: {}x{} com {}x{}", m, k, k2, n));
    }

    let output_id=super::next_tensor_id();
    let op=TensorOp::Matmul(std::rc::Rc::clone(left),std::rc::Rc::clone(right));
    let dtype=DType::promote(ldtype,rdtype);
    if backend.kind()==backend::BackendKind::Gpu {
        backend.matmul_resident_async(left.borrow().id,&lshape,right.borrow().id,&rshape,output_id)
            .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
        let out=super::Tensor::remote_with_id(output_id,vec![m,n],requires_grad,backend.kind(),dtype,op)?;
        out.borrow_mut().mark_host_stale();
        return Ok(Value::Tensor(out));
    }
    let (left_data,right_data)={let l=left.borrow();let r=right.borrow();(l.data_f32(),r.data_f32())};
    let out=backend.matmul(&left_data,&lshape,&right_data,&rshape)
        .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
    Ok(Value::Tensor(super::Tensor::derived_dtype_on_with_id(output_id,out,vec![m,n],requires_grad,backend.kind(),dtype,op)?))
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
    let left_id=left.id;
    let right_id=right.id;
    let shape=left.shape.clone();
    let requires_grad=left.requires_grad||right.requires_grad;
    let dtype=DType::promote(left.dtype,right.dtype);
    let output_id=super::next_tensor_id();
    let op_node=TensorOp::Elementwise(op,std::rc::Rc::clone(a),std::rc::Rc::clone(b));
    if backend.kind()==backend::BackendKind::Gpu {
        backend.elementwise_resident_async(left_id,right_id,&shape,backend_op,output_id)
            .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
        drop(left);drop(right);
        let _elements=shape.iter().copied().product::<usize>();
        let out=super::Tensor::remote_with_id(output_id,shape.clone(),requires_grad,backend.kind(),dtype,op_node)?;
        out.borrow_mut().mark_host_stale();
        return Ok(out);
    }
    let data=backend.elementwise(&left.data_f32(),&right.data_f32(),&shape,backend_op)
        .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
    drop(left);drop(right);
    Ok(super::Tensor::derived_dtype_on_with_id(output_id,data,shape,requires_grad,backend.kind(),dtype,op_node)?)
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
    let op = if mean { TensorOp::Mean(std::rc::Rc::clone(&tensor)) } else { TensorOp::Sum(std::rc::Rc::clone(&tensor)) };
    let requires_grad=borrowed.requires_grad;
    let dtype=borrowed.dtype;
    let input_id=borrowed.id;
    let elements=borrowed.data_len();
    if backend.kind()==backend::BackendKind::Gpu {
        let output_id=super::next_tensor_id();
        backend.reduce_resident_async(input_id,elements,mean,output_id)
            .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
        drop(borrowed);
        let out=super::Tensor::remote_with_id(output_id,vec![1],requires_grad,backend.kind(),dtype,op)?;
        out.borrow_mut().mark_host_stale();
        return Ok(Value::Tensor(out));
    }
    let result=backend.reduce(&borrowed.data_f32(),mean)
        .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
    drop(borrowed);
    Ok(Value::Tensor(super::Tensor::derived_dtype_on(vec![result],vec![1],requires_grad,backend.kind(),dtype,op)?))
}

fn sync_graph_host(
    node: &TensorRef,
    backend: &dyn TensorBackend,
    seen: &mut HashSet<u64>,
) -> Result<(), String> {
    let (id, valid, elements, op) = {
        let t=node.borrow();
        (t.id,t.host_valid,t.data_len(),t.op.clone())
    };
    if !seen.insert(id) { return Ok(()); }

    if backend.kind()==backend::BackendKind::Gpu && !valid {
        let data=backend.read_tensor(id,elements)
            .map_err(|e|format!("Nano: readback {}: {}",backend.kind().name(),e))?;
        node.borrow_mut().set_data_f32(data);
    }

    match op {
        TensorOp::Leaf => {}
        TensorOp::Elementwise(_,left,right) => {
            sync_graph_host(&left,backend,seen)?;
            sync_graph_host(&right,backend,seen)?;
        }
        TensorOp::Matmul(left,right) => {
            sync_graph_host(&left,backend,seen)?;
            sync_graph_host(&right,backend,seen)?;
        }
        TensorOp::MatmulTransposed(left,right,_,_) => {
            sync_graph_host(&left,backend,seen)?;
            sync_graph_host(&right,backend,seen)?;
        }
        TensorOp::FusedMulAdd(left,right,bias) => {
            sync_graph_host(&left,backend,seen)?;
            sync_graph_host(&right,backend,seen)?;
            sync_graph_host(&bias,backend,seen)?;
        }
        TensorOp::Sum(input) | TensorOp::Mean(input) => {
            sync_graph_host(&input,backend,seen)?;
        }
    }
    Ok(())
}

fn gpu_remote_tensor(
    id: u64,
    source: &TensorRef,
    shape: Vec<usize>,
    backend: &dyn TensorBackend,
) -> Result<TensorRef, String> {
    let (dtype, requires_grad) = {
        let t=source.borrow();
        (t.dtype,false)
    };
    super::Tensor::remote_with_id(id,shape,requires_grad,backend.kind(),dtype,TensorOp::Leaf)
}

fn gpu_gradient_accumulate(
    node: &TensorRef,
    upstream: &TensorRef,
    grads: &mut HashMap<u64, TensorRef>,
    backend: &dyn TensorBackend,
) -> Result<(), String> {
    let (id, shape, dtype) = {
        let t=node.borrow();
        (t.id,t.shape.clone(),t.dtype)
    };
    if let Some(existing)=grads.get(&id).cloned() {
        let output_id=super::next_tensor_id();
        backend.elementwise_resident_async(
            existing.borrow().id,
            upstream.borrow().id,
            &shape,
            ElementwiseOp::Add,
            output_id,
        ).map_err(|e|format!("Nano: GPU grad accumulate: {e}"))?;
        let sum=super::Tensor::remote_with_id(output_id,shape,false,backend.kind(),dtype,TensorOp::Leaf)?;
        grads.insert(id,sum);
    } else {
        grads.insert(id,std::rc::Rc::clone(upstream));
    }
    Ok(())
}

fn backward_gpu_matmul_transposed(
    node: &TensorRef,
    upstream: TensorRef,
    left: TensorRef,
    right: TensorRef,
    left_transpose: bool,
    right_transpose: bool,
    grads: &mut HashMap<u64, TensorRef>,
    backend: &dyn TensorBackend,
) -> Result<(), String> {
    let out_shape = node.borrow().shape.clone();
    let left_shape = left.borrow().shape.clone();
    let right_shape = right.borrow().shape.clone();

    let left_grad_id = super::next_tensor_id();
    if left_transpose {
        backend.matmul_transposed_resident_async(
            right.borrow().id,
            &right_shape,
            upstream.borrow().id,
            &out_shape,
            right_transpose,
            true,
            left_grad_id,
            &left_shape,
        ).map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;
    } else {
        backend.matmul_transposed_resident_async(
            upstream.borrow().id,
            &out_shape,
            right.borrow().id,
            &right_shape,
            false,
            !right_transpose,
            left_grad_id,
            &left_shape,
        ).map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;
    }
    let left_grad = super::Tensor::remote_with_id(
        left_grad_id,
        left_shape.clone(),
        false,
        backend.kind(),
        left.borrow().dtype,
        TensorOp::Leaf,
    ).map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;

    let right_grad_id = super::next_tensor_id();
    if right_transpose {
        backend.matmul_transposed_resident_async(
            upstream.borrow().id,
            &out_shape,
            left.borrow().id,
            &left_shape,
            true,
            left_transpose,
            right_grad_id,
            &right_shape,
        ).map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;
    } else {
        backend.matmul_transposed_resident_async(
            left.borrow().id,
            &left_shape,
            upstream.borrow().id,
            &out_shape,
            !left_transpose,
            false,
            right_grad_id,
            &right_shape,
        ).map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;
    }
    let right_grad = super::Tensor::remote_with_id(
        right_grad_id,
        right_shape.clone(),
        false,
        backend.kind(),
        right.borrow().dtype,
        TensorOp::Leaf,
    ).map_err(|e| format!("Nano: backend {}: {}", backend.kind().name(), e))?;

    backward_gpu(&left, left_grad, grads, backend)?;
    backward_gpu(&right, right_grad, grads, backend)?;
    Ok(())
}

fn backward_gpu(
    node: &TensorRef,
    upstream: TensorRef,
    grads: &mut HashMap<u64, TensorRef>,
    backend: &dyn TensorBackend,
) -> Result<(), String> {
    let op=node.borrow().op.clone();
    match op {
        TensorOp::Leaf => {
            gpu_gradient_accumulate(node,&upstream,grads,backend)
        }
        TensorOp::Elementwise(kind,left,right) => {
            let shape=node.borrow().shape.clone();
            match kind {
                TensorOpKind::Add => {
                    backward_gpu(&left,std::rc::Rc::clone(&upstream),grads,backend)?;
                    backward_gpu(&right,upstream,grads,backend)?;
                }
                TensorOpKind::Sub => {
                    backward_gpu(&left,std::rc::Rc::clone(&upstream),grads,backend)?;
                    let output_id=super::next_tensor_id();
                    backend.scale_resident_async(upstream.borrow().id,upstream.borrow().data_len(),-1.0,output_id)
                        .map_err(|e|format!("Nano: GPU grad sub: {e}"))?;
                    let neg=super::Tensor::remote_with_id(output_id,shape,false,backend.kind(),right.borrow().dtype,TensorOp::Leaf)?;
                    backward_gpu(&right,neg,grads,backend)?;
                }
                TensorOpKind::Mul => {
                    let lout=super::next_tensor_id();
                    backend.elementwise_resident_async(upstream.borrow().id,right.borrow().id,&shape,ElementwiseOp::Mul,lout)
                        .map_err(|e|format!("Nano: GPU grad mul: {e}"))?;
                    let lg=super::Tensor::remote_with_id(lout,left.borrow().shape.clone(),false,backend.kind(),left.borrow().dtype,TensorOp::Leaf)?;
                    let rout=super::next_tensor_id();
                    backend.elementwise_resident_async(upstream.borrow().id,left.borrow().id,&shape,ElementwiseOp::Mul,rout)
                        .map_err(|e|format!("Nano: GPU grad mul: {e}"))?;
                    let rg=super::Tensor::remote_with_id(rout,right.borrow().shape.clone(),false,backend.kind(),right.borrow().dtype,TensorOp::Leaf)?;
                    backward_gpu(&left,lg,grads,backend)?;
                    backward_gpu(&right,rg,grads,backend)?;
                }
                TensorOpKind::Div => {
                    let lgid=super::next_tensor_id();
                    backend.elementwise_resident_async(upstream.borrow().id,right.borrow().id,&shape,ElementwiseOp::Div,lgid)
                        .map_err(|e|format!("Nano: GPU grad div: {e}"))?;
                    let lg=super::Tensor::remote_with_id(lgid,left.borrow().shape.clone(),false,backend.kind(),left.borrow().dtype,TensorOp::Leaf)?;

                    let sqid=super::next_tensor_id();
                    backend.elementwise_resident_async(right.borrow().id,right.borrow().id,&shape,ElementwiseOp::Mul,sqid)
                        .map_err(|e|format!("Nano: GPU grad div: {e}"))?;
                    let numid=super::next_tensor_id();
                    backend.elementwise_resident_async(upstream.borrow().id,left.borrow().id,&shape,ElementwiseOp::Mul,numid)
                        .map_err(|e|format!("Nano: GPU grad div: {e}"))?;
                    let posid=super::next_tensor_id();
                    backend.elementwise_resident_async(numid,sqid,&shape,ElementwiseOp::Div,posid)
                        .map_err(|e|format!("Nano: GPU grad div: {e}"))?;
                    let negid=super::next_tensor_id();
                    backend.scale_resident_async(posid,shape.iter().copied().product::<usize>(),-1.0,negid)
                        .map_err(|e|format!("Nano: GPU grad div: {e}"))?;
                    let rg=super::Tensor::remote_with_id(negid,right.borrow().shape.clone(),false,backend.kind(),right.borrow().dtype,TensorOp::Leaf)?;
                    backward_gpu(&left,lg,grads,backend)?;
                    backward_gpu(&right,rg,grads,backend)?;
                }
            }
            Ok(())
        }
        TensorOp::Sum(input) => {
            let elements=input.borrow().data_len();
            let outid=super::next_tensor_id();
            backend.broadcast_resident_async(upstream.borrow().id,elements,1.0,outid)
                .map_err(|e|format!("Nano: GPU grad sum: {e}"))?;
            let grad=super::Tensor::remote_with_id(outid,input.borrow().shape.clone(),false,backend.kind(),input.borrow().dtype,TensorOp::Leaf)?;
            backward_gpu(&input,grad,grads,backend)
        }
        TensorOp::Mean(input) => {
            let elements=input.borrow().data_len().max(1);
            let outid=super::next_tensor_id();
            backend.broadcast_resident_async(upstream.borrow().id,elements,1.0/(elements as f32),outid)
                .map_err(|e|format!("Nano: GPU grad mean: {e}"))?;
            let grad=super::Tensor::remote_with_id(outid,input.borrow().shape.clone(),false,backend.kind(),input.borrow().dtype,TensorOp::Leaf)?;
            backward_gpu(&input,grad,grads,backend)
        }
        TensorOp::FusedMulAdd(left,right,bias) => {
            let shape=node.borrow().shape.clone();
            let lid=super::next_tensor_id();
            backend.elementwise_resident_async(upstream.borrow().id,right.borrow().id,&shape,ElementwiseOp::Mul,lid)
                .map_err(|e|format!("Nano: GPU grad FMA: {e}"))?;
            let lg=super::Tensor::remote_with_id(lid,left.borrow().shape.clone(),false,backend.kind(),left.borrow().dtype,TensorOp::Leaf)?;
            let rid=super::next_tensor_id();
            backend.elementwise_resident_async(upstream.borrow().id,left.borrow().id,&shape,ElementwiseOp::Mul,rid)
                .map_err(|e|format!("Nano: GPU grad FMA: {e}"))?;
            let rg=super::Tensor::remote_with_id(rid,right.borrow().shape.clone(),false,backend.kind(),right.borrow().dtype,TensorOp::Leaf)?;
            backward_gpu(&left,lg,grads,backend)?;
            backward_gpu(&right,rg,grads,backend)?;
            backward_gpu(&bias,upstream,grads,backend)
        }
        TensorOp::Matmul(left,right) => {
            let out_shape=node.borrow().shape.clone();
            let left_shape=left.borrow().shape.clone();
            let right_shape=right.borrow().shape.clone();
            let lid=super::next_tensor_id();
            backend.matmul_transposed_resident_async(
                upstream.borrow().id,&out_shape,
                right.borrow().id,&right_shape,
                false,true,lid,&left_shape
            ).map_err(|e|format!("Nano: GPU grad matmul: {e}"))?;
            let lg=super::Tensor::remote_with_id(lid,left_shape.clone(),false,backend.kind(),left.borrow().dtype,TensorOp::Leaf)?;
            let rid=super::next_tensor_id();
            backend.matmul_transposed_resident_async(
                left.borrow().id,&left_shape,
                upstream.borrow().id,&out_shape,
                true,false,rid,&right_shape
            ).map_err(|e|format!("Nano: GPU grad matmul: {e}"))?;
            let rg=super::Tensor::remote_with_id(rid,right_shape.clone(),false,backend.kind(),right.borrow().dtype,TensorOp::Leaf)?;
            backward_gpu(&left,lg,grads,backend)?;
            backward_gpu(&right,rg,grads,backend)
        }
        TensorOp::MatmulTransposed(left,right,left_transpose,right_transpose) => {
            backward_gpu_matmul_transposed(
                node,
                upstream,
                left,
                right,
                left_transpose,
                right_transpose,
                grads,
                backend,
            )
        }
    }
}

fn gradient_value(loss: &Value, parameter: &Value, backend: &dyn TensorBackend) -> Result<Value, String> {
    let loss_ref = match loss {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: grad() requer Tensor como loss".into()),
    };
    let param_ref = match parameter {
        Value::Tensor(t) => std::rc::Rc::clone(t),
        _ => return Err("Nano: grad() requer Tensor como parâmetro".into()),
    };

    let output_shape=param_ref.borrow().shape.clone();
    if backend.kind()==backend::BackendKind::Gpu {
        let loss_shape=loss_ref.borrow().shape.clone();
        let loss_dtype=loss_ref.borrow().dtype;
        let upstream_id=super::next_tensor_id();
        backend.fill_resident_async(loss_ref.borrow().data_len(),1.0,upstream_id)
            .map_err(|e|format!("Nano: GPU grad seed: {e}"))?;
        let upstream=super::Tensor::remote_with_id(upstream_id,loss_shape,false,backend.kind(),loss_dtype,TensorOp::Leaf)?;
        let mut grads:HashMap<u64,TensorRef>=HashMap::new();
        backward_gpu(&loss_ref,upstream,&mut grads,backend)?;
        let param_id=param_ref.borrow().id;
        if let Some(grad)=grads.remove(&param_id) {
            return Ok(Value::Tensor(grad));
        }
        let shape=param_ref.borrow().shape.clone();
        let dtype=param_ref.borrow().dtype;
        let output_id=super::next_tensor_id();
        backend.fill_resident_async(param_ref.borrow().data_len(),0.0,output_id)
            .map_err(|e|format!("Nano: GPU grad zero: {e}"))?;
        return Ok(Value::Tensor(super::Tensor::remote_with_id(output_id,shape,false,backend.kind(),dtype,TensorOp::Leaf)?));
    }
    let mut seen=HashSet::new();
    sync_graph_host(&loss_ref,backend,&mut seen)?;
    let mut grads: HashMap<u64,Vec<f32>>=HashMap::new();
    let upstream=vec![1.0_f32;loss_ref.borrow().data_len()];
    backward(&loss_ref,upstream,&mut grads)?;

    let id = param_ref.borrow().id;
    let data = grads.get(&id).cloned().unwrap_or_else(|| vec![0.0; param_ref.borrow().data_len()]);
    let source = param_ref.borrow();
    let device = source.device;
    let dtype = source.dtype;
    drop(source);
    let tensor=super::Tensor::new_with_dtype_on(data.clone(),output_shape,false,device,dtype)?;
    let id=tensor.borrow().id;
    backend.sync_tensor(id,&data).map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
    Ok(Value::Tensor(tensor))
}

fn step_value(parameter: &Value, gradient: &Value, rate: &Value, backend: &dyn TensorBackend) -> Result<Value, String> {
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

    {
        let p=param.borrow();
        let g=grad.borrow();
        if p.shape!=g.shape {
            return Err("Nano: parâmetro e gradiente precisam ter o mesmo shape".into());
        }
        if p.device!=g.device {
            return Err("Nano: parâmetro e gradiente precisam estar no mesmo dispositivo".into());
        }
        if backend.kind()==backend::BackendKind::Gpu {
            backend.step_resident_async(p.id,g.id,p.data_len(),lr)
                .map_err(|e|format!("Nano: GPU step: {e}"))?;
            drop(p);
            drop(g);
            param.borrow_mut().mark_host_stale();
            return Ok(Value::Tensor(param));
        }
    }

    let (mut data, gradient)={
        let p=param.borrow();
        let g=grad.borrow();
        (p.data_f32(),g.data_f32())
    };
    for (value,delta) in data.iter_mut().zip(&gradient) {
        *value-=lr*delta;
    }
    let id=param.borrow().id;
    param.borrow_mut().set_data_f32(data.clone());
    backend.sync_tensor(id,&data)
        .map_err(|e|format!("Nano: backend {}: {}",backend.kind().name(),e))?;
    Ok(Value::Tensor(param))
}

fn backward_cpu_matmul_transposed(
    node: &TensorRef,
    upstream: Vec<f32>,
    left: TensorRef,
    right: TensorRef,
    left_transpose: bool,
    right_transpose: bool,
    grads: &mut HashMap<u64, Vec<f32>>,
) -> Result<(), String> {
    let out_shape = node.borrow().shape.clone();
    if out_shape.len() != 2 {
        return Err("Nano: grad matmul_transposed requer saída 2D".into());
    }
    let (ar, ac, br, bc) = {
        let l = left.borrow();
        let r = right.borrow();
        (l.shape[0], l.shape[1], r.shape[0], r.shape[1])
    };
    let (m, k) = if left_transpose { (ac, ar) } else { (ar, ac) };
    let (k2, n) = if right_transpose { (bc, br) } else { (br, bc) };
    if k != k2 || out_shape != vec![m, n] || upstream.len() != m * n {
        return Err("Nano: shapes incompatíveis no grad matmul_transposed".into());
    }

    let (ldata, rdata) = {
        let l = left.borrow();
        let r = right.borrow();
        (l.data_f32(), r.data_f32())
    };

    let a_log = |row: usize, col: usize| -> f32 {
        if left_transpose { ldata[col * ac + row] } else { ldata[row * ac + col] }
    };
    let b_log = |row: usize, col: usize| -> f32 {
        if right_transpose { rdata[col * bc + row] } else { rdata[row * bc + col] }
    };

    let mut left_grad = vec![0.0f32; ar * ac];
    for i in 0..ar {
        for j in 0..ac {
            let (lr, lc) = if left_transpose { (j, i) } else { (i, j) };
            let mut sum = 0.0;
            for q in 0..n {
                sum += upstream[lr * n + q] * b_log(lc, q);
            }
            left_grad[i * ac + j] = sum;
        }
    }

    let mut right_grad = vec![0.0f32; br * bc];
    for i in 0..br {
        for j in 0..bc {
            let (lr, lc) = if right_transpose { (j, i) } else { (i, j) };
            let mut sum = 0.0;
            for p in 0..m {
                sum += a_log(p, lr) * upstream[p * n + lc];
            }
            right_grad[i * bc + j] = sum;
        }
    }

    backward(&left, left_grad, grads)?;
    backward(&right, right_grad, grads)
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
                (l.data_f32(), r.data_f32())
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
            let size = input.borrow().data_len();
            backward(&input, vec![scalar; size], grads)
        }
        TensorOp::Mean(input) => {
            let scalar = upstream.first().copied().unwrap_or(0.0);
            let size = input.borrow().data_len().max(1);
            backward(&input, vec![scalar / size as f32; size], grads)
        }
        TensorOp::FusedMulAdd(left, right, bias) => {
            let (ldata, rdata) = {
                let l = left.borrow();
                let r = right.borrow();
                (l.data_f32(), r.data_f32())
            };
            let lg = upstream.iter().zip(&rdata).map(|(u, r)| u * r).collect();
            let rg = upstream.iter().zip(&ldata).map(|(u, l)| u * l).collect();
            backward(&left, lg, grads)?;
            backward(&right, rg, grads)?;
            backward(&bias, upstream, grads)
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
                (l.data_f32(), l.shape.clone(), r.data_f32(), r.shape.clone())
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
        TensorOp::MatmulTransposed(left, right, left_transpose, right_transpose) => {
            backward_cpu_matmul_transposed(
                node,
                upstream,
                left,
                right,
                left_transpose,
                right_transpose,
                grads,
            )
        }
    }
}


#[cfg(test)]
mod ir_tests {
    use std::collections::HashMap;
    use super::{backward, index_value, IrRuntime, TensorOp, Value};
    use crate::{backend::BackendKind, Tensor};

    #[test]
    fn tensor_index_reads_flattened_storage() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        let value = index_value(Value::Tensor(tensor), Value::Number(2.0)).unwrap();
        assert_eq!(value, Value::Number(3.0));
    }

    #[test]
    fn runtime_starts_on_cpu() {
        let runtime = IrRuntime::new();
        assert_eq!(runtime.backend.kind(), BackendKind::Cpu);
    }

    #[test]
    fn transposed_matmul_cpu_backward_maps_gradients_to_raw_shapes() {
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
        let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true).unwrap();
        let output = Tensor::derived_on(
            vec![26.0, 30.0, 38.0, 44.0],
            vec![2, 2],
            true,
            BackendKind::Cpu,
            TensorOp::MatmulTransposed(left.clone(), right.clone(), true, false),
        ).unwrap();
        let mut grads = HashMap::new();
        backward(&output, vec![1.0; 4], &mut grads).unwrap();

        assert_eq!(grads.get(&left.borrow().id).unwrap(), &vec![11.0, 11.0, 15.0, 15.0]);
        assert_eq!(grads.get(&right.borrow().id).unwrap(), &vec![3.0, 3.0, 7.0, 7.0]);
    }


    #[test]
    fn standard_library_strings_and_math_work() {
        let mut runtime = IrRuntime::new();

        assert_eq!(
            runtime.call("upper", vec![Value::Text("nano".into())]).unwrap(),
            Value::Text("NANO".into())
        );
        assert_eq!(
            runtime.call("substring", vec![
                Value::Text("Nanolang".into()),
                Value::Number(0.0),
                Value::Number(4.0),
            ]).unwrap(),
            Value::Text("Nano".into())
        );
        assert_eq!(
            runtime.call("split", vec![
                Value::Text("a,b,c".into()),
                Value::Text(",".into()),
            ]).unwrap(),
            Value::List(vec![
                Value::Text("a".into()),
                Value::Text("b".into()),
                Value::Text("c".into()),
            ])
        );
        assert_eq!(
            runtime.call("pow", vec![
                Value::Number(2.0),
                Value::Number(8.0),
            ]).unwrap(),
            Value::Number(256.0)
        );
        assert_eq!(
            runtime.call("min", vec![
                Value::Number(4.0),
                Value::Number(2.0),
            ]).unwrap(),
            Value::Number(2.0)
        );
    }

    #[test]
    fn standard_library_filesystem_round_trip() {
        let mut runtime = IrRuntime::new();
        let path = std::env::temp_dir().join(format!(
            "nano-test-{}.txt",
            std::process::id()
        ));
        let path_text = path.to_string_lossy().into_owned();

        runtime.call("fs_write_text", vec![
            Value::Text(path_text.clone()),
            Value::Text("hello nano".into()),
        ]).unwrap();

        let read = runtime.call("fs_read_text", vec![
            Value::Text(path_text.clone()),
        ]).unwrap();

        assert_eq!(read, Value::Text("hello nano".into()));

        runtime.call("fs_remove", vec![
            Value::Text(path_text.clone()),
        ]).unwrap();

        assert!(!path.exists());
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DebugStop {
    Step,
    Breakpoint,
    Exited,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DebugStepMode {
    Instruction,
    Next,
    StepOut,
}

#[derive(Debug, Clone)]
struct DebugFrame {
    id: usize,
    name: String,
    code: Vec<IrInst>,
    spans: Vec<Option<SourceSpan>>,
    ip: usize,
    stack: Vec<Value>,
    iterators: Vec<IterState>,
    locals: HashMap<String, Value>,
}

pub(crate) struct DebugSession {
    runtime: IrRuntime,
    debug_functions: HashMap<String, Vec<Option<SourceSpan>>>,
    frames: Vec<DebugFrame>,
    breakpoints: HashSet<usize>,
    last_breakpoint: Option<(usize, usize)>,
    next_frame_id: usize,
}

impl DebugSession {
    pub(crate) fn new(program: &IrProgram, debug: &DebugMetadata) -> Result<Self, String> {
        let mut runtime = IrRuntime::new();
        runtime.quiet = true;
        runtime.functions.extend(program.functions.clone());

        Ok(Self {
            runtime,
            debug_functions: debug.functions.clone(),
            frames: vec![DebugFrame {
                id: 1,
                name: "_main".into(),
                code: program.code.clone(),
                spans: debug.main.clone(),
                ip: 0,
                stack: Vec::new(),
                iterators: Vec::new(),
                locals: HashMap::new(),
            }],
            breakpoints: HashSet::new(),
            last_breakpoint: None,
            next_frame_id: 2,
        })
    }

    pub(crate) fn set_breakpoints(&mut self, lines: &[usize]) {
        self.breakpoints = lines.iter().copied().filter(|line| *line > 0).collect();
        self.last_breakpoint = None;
    }

    pub(crate) fn exited(&self) -> bool {
        self.frames.is_empty()
    }

    fn current_frame(&self) -> Option<&DebugFrame> {
        self.frames.last()
    }

    fn current_frame_mut(&mut self) -> Option<&mut DebugFrame> {
        self.frames.last_mut()
    }

    fn current_span(&self) -> Option<SourceSpan> {
        let frame = self.current_frame()?;
        frame.spans.get(frame.ip).and_then(|span| *span)
    }

    pub(crate) fn current_line(&self) -> usize {
        self.current_span().map(|span| span.start_line).unwrap_or(1)
    }

    pub(crate) fn current_column(&self) -> usize {
        self.current_span().map(|span| span.start_col).unwrap_or(1)
    }

    pub(crate) fn stack_frames(&self) -> Vec<(usize, String, usize, usize)> {
        self.frames.iter().rev().map(|frame| {
            let span = frame.spans.get(frame.ip).and_then(|span| *span);
            (
                frame.id,
                frame.name.clone(),
                span.map(|value| value.start_line).unwrap_or(1),
                span.map(|value| value.start_col).unwrap_or(1),
            )
        }).collect()
    }

    pub(crate) fn variables_for_frame(&self, frame_id: usize) -> HashMap<String, Value> {
        self.frames.iter()
            .find(|frame| frame.id == frame_id)
            .map(|frame| {
                let mut vars = self.runtime.vars.clone();
                vars.extend(frame.locals.clone());
                vars
            })
            .unwrap_or_default()
    }

    pub(crate) fn evaluate_for_frame(&self, frame_id: usize, expression: &str) -> Option<Value> {
        let trimmed = expression.trim();
        self.frames.iter()
            .find(|frame| frame.id == frame_id)
            .and_then(|frame| frame.locals.get(trimmed).cloned().or_else(|| self.runtime.vars.get(trimmed).cloned()))
            .or_else(|| match trimmed {
                "true" => Some(Value::Boolean(true)),
                "false" => Some(Value::Boolean(false)),
                "null" => Some(Value::Null),
                value if value.starts_with('"') && value.ends_with('"') => Some(Value::Text(value[1..value.len()-1].to_string())),
                value => value.parse::<f64>().ok().map(Value::Number),
            })
    }

    fn breakpoint_here(&self) -> bool {
        self.current_span()
            .map(|span| self.breakpoints.iter().any(|line| span.contains_line(*line)))
            .unwrap_or(false)
    }

    fn push_user_frame(&mut self, name: &str, args: Vec<Value>) -> Result<bool, String> {
        let Some(function) = self.runtime.functions.get(name).cloned() else {
            return Ok(false);
        };
        if function.params.len() != args.len() {
            return Err(format!(
                "Nano debug: função '{name}' esperava {} argumentos, recebeu {}",
                function.params.len(), args.len()
            ));
        }

        let mut locals = HashMap::new();
        for (param, value) in function.params.iter().zip(args) {
            locals.insert(param.clone(), value);
        }

        if let Some(caller) = self.current_frame_mut() {
            caller.ip = caller.ip.saturating_add(1);
        }

        let frame = DebugFrame {
            id: self.next_frame_id,
            name: name.to_string(),
            code: function.code,
            spans: self.debug_functions.get(name).cloned().unwrap_or_default(),
            ip: 0,
            stack: Vec::new(),
            iterators: Vec::new(),
            locals,
        };
        self.next_frame_id += 1;
        self.frames.push(frame);
        Ok(true)
    }

    fn exec_one(&mut self) -> Result<(), String> {
        if self.exited() {
            return Ok(());
        }

        let instruction = {
            let frame = self.current_frame().unwrap();
            if frame.ip >= frame.code.len() {
                return Ok(());
            }
            frame.code[frame.ip].clone()
        };

        let mut jumped = false;
        match instruction {
            IrInst::Const(value) => self.current_frame_mut().unwrap().stack.push(value),

            IrInst::Load(name) => {
                let value = {
                    let frame = self.current_frame().unwrap();
                    frame.locals.get(&name).cloned()
                        .or_else(|| self.runtime.vars.get(&name).cloned())
                        .or_else(|| is_builtin_name(&name).then(|| Value::Function(name.clone())))
                        .ok_or_else(|| format!("Nano debug: variável '{name}' não definida"))?
                };
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::Store(name) => {
                let value = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia em Store".to_string())?;
                self.current_frame_mut().unwrap().locals.insert(name, value);
            }

            IrInst::SetIndex => {
                let frame = self.current_frame_mut().unwrap();
                let value = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em SetIndex value".to_string())?;
                let index = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em SetIndex index".to_string())?;
                let target = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em SetIndex target".to_string())?;
                let updated = match (target, index) {
                    (Value::List(mut values), Value::Number(index)) if index >= 0.0 && index.fract() == 0.0 => {
                        let index = index as usize;
                        if index >= values.len() {
                            return Err("Nano debug: índice fora do limite".into());
                        }
                        values[index] = value;
                        Value::List(values)
                    }
                    (Value::Object(mut values), Value::Text(key)) => {
                        values.insert(key, value);
                        Value::Object(values)
                    }
                    _ => return Err("Nano debug: SetIndex inválido".into()),
                };
                frame.stack.push(updated);
            }

            IrInst::SetField(name) => {
                let frame = self.current_frame_mut().unwrap();
                let value = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em SetField value".to_string())?;
                let target = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em SetField target".to_string())?;
                let mut values = match target {
                    Value::Object(values) => values,
                    _ => return Err("Nano debug: SetField requer Object".into()),
                };
                values.insert(name, value);
                frame.stack.push(Value::Object(values));
            }

            IrInst::Binary(op) => {
                let right = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia no operando direito".to_string())?;
                let left = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia no operando esquerdo".to_string())?;
                let value = self.runtime.binary_value(left, op, right)?;
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::Unary(op) => {
                let value = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia em Unary".to_string())?;
                self.current_frame_mut().unwrap().stack.push(unary_value(value, op)?);
            }

            IrInst::FusedMulAdd => {
                let (left, right, bias) = {
                    let frame = self.current_frame_mut().unwrap();
                    let bias = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em FMA bias".to_string())?;
                    let right = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em FMA right".to_string())?;
                    let left = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia em FMA left".to_string())?;
                    (left, right, bias)
                };
                let value = self.runtime.fused_mul_add_value(left, right, bias)?;
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::MakeList(count) => {
                let mut values = pop_n(&mut self.current_frame_mut().unwrap().stack, count)?;
                values.reverse();
                self.current_frame_mut().unwrap().stack.push(Value::List(values));
            }

            IrInst::MakeObject(keys) => {
                let mut values = pop_n(&mut self.current_frame_mut().unwrap().stack, keys.len())?;
                values.reverse();
                let mut object = HashMap::new();
                for (key, value) in keys.into_iter().zip(values) {
                    object.insert(key, value);
                }
                self.current_frame_mut().unwrap().stack.push(Value::Object(object));
            }

            IrInst::Index => {
                let (target, index) = {
                    let frame = self.current_frame_mut().unwrap();
                    let index = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia no índice".to_string())?;
                    let target = frame.stack.pop().ok_or_else(|| "Nano debug: stack vazia no alvo".to_string())?;
                    (target, index)
                };
                let value = self.runtime.index_value(target, index)?;
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::Field(name) => {
                let target = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia no field".to_string())?;
                let value = match target {
                    Value::Object(values) => values.get(&name).cloned()
                        .ok_or_else(|| format!("Nano debug: campo '{name}' não existe"))?,
                    _ => return Err("Nano debug: Field requer Object".into()),
                };
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::Call(name, count) => {
                let mut args = pop_n(&mut self.current_frame_mut().unwrap().stack, count)?;
                args.reverse();
                if self.push_user_frame(&name, args.clone())? {
                    return Ok(());
                }
                let value = self.runtime.call(&name, args)?;
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::CallValue(count) => {
                let mut args = pop_n(&mut self.current_frame_mut().unwrap().stack, count)?;
                args.reverse();
                let target = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia no alvo da chamada".to_string())?;
                let name = match target {
                    Value::Function(name) => name,
                    _ => return Err("Nano debug: alvo não é Function".into()),
                };
                if self.push_user_frame(&name, args.clone())? {
                    return Ok(());
                }
                let value = self.runtime.call(&name, args)?;
                self.current_frame_mut().unwrap().stack.push(value);
            }

            IrInst::Print => {
                let _ = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia em Print".to_string())?;
            }

            IrInst::Pop => {
                let _ = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia em Pop".to_string())?;
            }

            IrInst::JumpIfFalse(target) => {
                let value = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia em JumpIfFalse".to_string())?;
                if !value.truthy() {
                    self.current_frame_mut().unwrap().ip = target;
                    jumped = true;
                }
            }

            IrInst::Jump(target) => {
                self.current_frame_mut().unwrap().ip = target;
                jumped = true;
            }

            IrInst::IterInit => {
                let iterable = self.current_frame_mut().unwrap().stack.pop()
                    .ok_or_else(|| "Nano debug: stack vazia em IterInit".to_string())?;
                match iterable {
                    Value::List(values) => self.current_frame_mut().unwrap().iterators.push(IterState { values, index: 0 }),
                    _ => return Err("Nano debug: for requer List".into()),
                }
            }

            IrInst::IterNext(name, target) => {
                let next = {
                    let frame = self.current_frame_mut().unwrap();
                    let state = frame.iterators.last_mut()
                        .ok_or_else(|| "Nano debug: iterador ausente".to_string())?;
                    if state.index < state.values.len() {
                        let value = state.values[state.index].clone();
                        state.index += 1;
                        Some(value)
                    } else {
                        None
                    }
                };
                match next {
                    Some(value) => {
                        self.current_frame_mut().unwrap().locals.insert(name, value);
                    }
                    None => {
                        self.current_frame_mut().unwrap().iterators.pop();
                        self.current_frame_mut().unwrap().ip = target;
                        jumped = true;
                    }
                }
            }

            IrInst::Return => {
                let value = self.current_frame_mut().unwrap().stack.pop().unwrap_or(Value::Null);
                if self.frames.len() == 1 {
                    let len = self.current_frame().unwrap().code.len();
                    self.current_frame_mut().unwrap().ip = len;
                    return Ok(());
                }
                self.frames.pop();
                self.current_frame_mut().unwrap().stack.push(value);
                return Ok(());
            }

            IrInst::Use(path) => self.runtime.load_module(&path)?,
        }

        if let Some(frame) = self.current_frame_mut() {
            if !jumped {
                frame.ip += 1;
            }
        }
        Ok(())
    }

    pub(crate) fn step(&mut self, mode: DebugStepMode) -> Result<DebugStop, String> {
        if self.exited() {
            return Ok(DebugStop::Exited);
        }
        let start_depth = self.frames.len();
        let start_frame_id = self.frames.last().unwrap().id;
        let start_span = self.current_span();

        loop {
            if self.exited() {
                return Ok(DebugStop::Exited);
            }

            if self.breakpoint_here() {
                let key = (self.frames.last().unwrap().id, self.frames.last().unwrap().ip);
                if self.last_breakpoint != Some(key) {
                    self.last_breakpoint = Some(key);
                    return Ok(DebugStop::Breakpoint);
                }
            }

            self.exec_one()?;

            if self.exited() {
                return Ok(DebugStop::Exited);
            }

            let depth = self.frames.len();
            let span = self.current_span();
            let frame_changed = depth != start_depth || self.frames.last().map(|f| f.id).unwrap_or(0) != start_frame_id;
            let source_changed = span != start_span;

            match mode {
                DebugStepMode::Instruction => return Ok(DebugStop::Step),
                DebugStepMode::Next if depth <= start_depth && source_changed => return Ok(DebugStop::Step),
                DebugStepMode::StepOut if depth < start_depth => return Ok(DebugStop::Step),
                _ if frame_changed && mode == DebugStepMode::Instruction => return Ok(DebugStop::Step),
                _ => {}
            }
        }
    }

    pub(crate) fn continue_run(&mut self) -> Result<DebugStop, String> {
        self.last_breakpoint = None;
        loop {
            if self.exited() {
                return Ok(DebugStop::Exited);
            }
            if self.breakpoint_here() {
                let key = (self.frames.last().unwrap().id, self.frames.last().unwrap().ip);
                if self.last_breakpoint != Some(key) {
                    self.last_breakpoint = Some(key);
                    return Ok(DebugStop::Breakpoint);
                }
            }
            self.exec_one()?;
        }
    }
}
