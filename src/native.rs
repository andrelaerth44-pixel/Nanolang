use std::{
    collections::HashMap,
    fs,
    path::Path,
    process::Command,
};

use crate::{ir::{IrInst, IrProgram}, Op, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Number,
    Boolean,
    Text,
}

pub(crate) fn build(ir: &IrProgram, output: &Path) -> Result<(), String> {
    if !cfg!(target_arch = "x86_64") || !cfg!(target_os = "linux") {
        return Err("Nano: o backend --native atual requer x86_64 Linux.".into());
    }

    let mut module = NativeModule::new();
    let mut function_returns = HashMap::<String, Kind>::new();
    for (name, function) in &ir.functions {
        let locals = initial_locals(&function.params, &function.code)?;
        let (_, _, _, return_kind) = analyze_stack(
            &function.code,
            name,
            &locals,
            &function.params,
            &function_returns,
        )?;
        function_returns.insert(name.clone(), return_kind.unwrap_or(Kind::Number));
    }

    module.compile_function("_main", &ir.code, &[], &function_returns)?;
    for (name, function) in &ir.functions {
        module.compile_function(name, &function.code, &function.params, &function_returns)?;
    }

    let assembly = module.render();
    let asm_path = output.with_extension("s");
    fs::write(&asm_path, &assembly)
        .map_err(|e| format!("Nano: não foi possível escrever '{}': {e}", asm_path.display()))?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .arg(&asm_path)
        .arg("-o")
        .arg(output)
        .arg("-lm")
        .status()
        .map_err(|e| format!("Nano: não foi possível executar o linker '{cc}': {e}"))?;

    if !status.success() {
        return Err(format!("Nano: linker '{cc}' terminou com código {:?}", status.code()));
    }

    if std::env::var_os("NANO_KEEP_ASM").is_none() {
        let _ = fs::remove_file(&asm_path);
    }
    Ok(())
}

struct NativeModule {
    text: String,
    float_constants: Vec<f64>,
    text_constants: Vec<Vec<u8>>,
}

impl NativeModule {
    fn new() -> Self {
        Self {
            text: String::new(),
            float_constants: Vec::new(),
            text_constants: Vec::new(),
        }
    }

    fn compile_function(
        &mut self,
        name: &str,
        code: &[IrInst],
        params: &[String],
        function_returns: &HashMap<String, Kind>,
    ) -> Result<(), String> {
        let symbol = format!("nano_fn_{}", sanitize(name));
        let locals = initial_locals(params, code)?;
        let (entry_states, max_stack, _local_kinds, _return_kind) =
            analyze_stack(code, name, &locals, params, function_returns)?;

        let frame = (((4096 + locals.len().max(1) * 8 + max_stack * 8) + 15) / 16) * 16;
        self.text.push_str(&format!(
            "\n    .text\n    .globl {symbol}\n{symbol}:\n    pushq %rbp\n    movq %rsp, %rbp\n    subq "
        ));
        self.text.push_str("$");
        self.text.push_str(&frame.to_string());
        self.text.push_str(", %rsp\n");

        for index in 0..params.len() {
            self.text.push_str(&format!(
                "    movsd %xmm{index}, {}(%rbp)\n",
                local_offset(index)
            ));
        }

        let labels: HashMap<usize, String> = (0..code.len())
            .map(|i| (i, format!(".L{symbol}_{i}")))
            .collect();

        for (ip, inst) in code.iter().enumerate() {
            let Some(entry) = entry_states[ip].as_ref() else {
                // Código após return/jump pode existir no IR e não precisa ser emitido.
                continue;
            };
            let depth = entry.len();

            self.text.push_str(&format!("{}:\n", labels[&ip]));

            match inst {
                IrInst::Const(Value::Number(value)) => {
                    let label = self.add_float(*value);
                    self.text.push_str(&format!(
                        "    movsd {label}(%rip), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(depth)
                    ));
                }
                IrInst::Const(Value::Boolean(value)) => {
                    let label = self.add_float(if *value { 1.0 } else { 0.0 });
                    self.text.push_str(&format!(
                        "    movsd {label}(%rip), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(depth)
                    ));
                }
                IrInst::Const(Value::Text(value)) => {
                    let label = self.add_text(value);
                    self.text.push_str(&format!(
                        "    leaq {label}(%rip), %rax\n    movq %rax, {}(%rbp)\n",
                        stack_offset(depth)
                    ));
                }
                IrInst::Load(var) => {
                    let index = *locals.get(var)
                        .ok_or_else(|| format!("Nano native: variável '{var}' não é conhecida em '{name}'"))?;
                    self.text.push_str(&format!(
                        "    movsd {}(%rbp), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        local_offset(index),
                        stack_offset(depth)
                    ));
                }
                IrInst::Store(var) => {
                    let slot = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: Store sem valor em '{name}'"))?;
                    let index = *locals.get(var).unwrap();
                    self.text.push_str(&format!(
                        "    movsd {}(%rbp), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(slot),
                        local_offset(index)
                    ));
                }
                IrInst::Binary(op) => {
                    let right = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: binary sem operando direito em '{name}'"))?;
                    let left = depth.checked_sub(2)
                        .ok_or_else(|| format!("Nano native: binary sem operando esquerdo em '{name}'"))?;
                    self.load_stack(right, "%xmm1");
                    self.load_stack(left, "%xmm0");

                    match op {
                        Op::Add => self.text.push_str("    addsd %xmm1, %xmm0\n"),
                        Op::Sub => self.text.push_str("    subsd %xmm1, %xmm0\n"),
                        Op::Mul => self.text.push_str("    mulsd %xmm1, %xmm0\n"),
                        Op::Div => self.text.push_str("    divsd %xmm1, %xmm0\n"),
                        Op::Mod => {
                            self.text.push_str("    call fmod@PLT\n");
                        }
                        Op::And | Op::Or => {
                            let zero = self.add_float(0.0);
                            self.text.push_str(&format!(
                                "    ucomisd {zero}(%rip), %xmm0\n    setne %al\n    movzbl %al, %eax\n"
                            ));
                            self.text.push_str(&format!(
                                "    ucomisd {zero}(%rip), %xmm1\n    setne %cl\n    movzbl %cl, %ecx\n"
                            ));
                            match op {
                                Op::And => self.text.push_str("    andl %ecx, %eax\n"),
                                Op::Or => self.text.push_str("    orl %ecx, %eax\n"),
                                _ => unreachable!(),
                            }
                            self.text.push_str("    cvtsi2sd %eax, %xmm0\n");
                        }
                        Op::Eq | Op::Ne | Op::Gt | Op::Ge | Op::Lt | Op::Le => {
                            self.text.push_str("    xorl %eax, %eax\n    ucomisd %xmm1, %xmm0\n");
                            let set = match op {
                                Op::Eq => "sete",
                                Op::Ne => "setne",
                                Op::Gt => "seta",
                                Op::Ge => "setae",
                                Op::Lt => "setb",
                                Op::Le => "setbe",
                                _ => unreachable!(),
                            };
                            self.text.push_str(&format!(
                                "    {set} %al\n    movzbl %al, %eax\n    cvtsi2sd %eax, %xmm0\n"
                            ));
                        }
                    }

                    self.text.push_str(&format!(
                        "    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(left)
                    ));
                }
                IrInst::Unary(op) => {
                    let slot = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: unary sem valor em '{name}'"))?;
                    self.load_stack(slot, "%xmm0");
                    match op {
                        crate::UnaryOp::Neg => {
                            self.text.push_str("    xorpd %xmm1, %xmm1\n    subsd %xmm0, %xmm1\n    movsd %xmm1, %xmm0\n");
                        }
                        crate::UnaryOp::Not => {
                            let zero = self.add_float(0.0);
                            self.text.push_str(&format!(
                                "    ucomisd {zero}(%rip), %xmm0\n    sete %al\n    movzbl %al, %eax\n    cvtsi2sd %eax, %xmm0\n"
                            ));
                        }
                    }
                    self.text.push_str(&format!(
                        "    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(slot)
                    ));
                }
                IrInst::FusedMulAdd => {
                    let bias = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: FMA sem bias em '{name}'"))?;
                    let right = depth.checked_sub(2)
                        .ok_or_else(|| format!("Nano native: FMA sem direito em '{name}'"))?;
                    let left = depth.checked_sub(3)
                        .ok_or_else(|| format!("Nano native: FMA sem esquerdo em '{name}'"))?;
                    self.load_stack(right, "%xmm1");
                    self.load_stack(left, "%xmm0");
                    self.text.push_str("    mulsd %xmm1, %xmm0\n");
                    self.load_stack(bias, "%xmm1");
                    self.text.push_str(&format!(
                        "    addsd %xmm1, %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(left)
                    ));
                }
                IrInst::Call(callee, count) => {
                    let start = depth.checked_sub(*count)
                        .ok_or_else(|| format!("Nano native: chamada '{callee}' sem argumentos suficientes"))?;
                    for (arg, reg) in (start..depth).zip(0..8) {
                        self.load_stack(arg, &format!("%xmm{reg}"));
                    }
                    self.text.push_str(&format!("    call nano_fn_{}\n", sanitize(callee)));
                    self.text.push_str(&format!(
                        "    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(start)
                    ));
                }
                IrInst::Print => {
                    let slot = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: print sem valor em '{name}'"))?;
                    match entry_states[ip].as_ref().unwrap().last().copied().unwrap() {
                        Kind::Number => {
                            self.text.push_str(&format!(
                                "    movsd {}(%rbp), %xmm0\n    leaq nano_fmt(%rip), %rdi\n    movl $1, %eax\n    call printf@PLT\n",
                                stack_offset(slot)
                            ));
                        }
                        Kind::Boolean => {
                            let zero = self.add_float(0.0);
                            let false_label = format!(".L{symbol}_bool_false_{ip}");
                            let end_label = format!(".L{symbol}_bool_end_{ip}");
                            self.text.push_str(&format!(
                                "    movsd {}(%rbp), %xmm0\n    ucomisd {zero}(%rip), %xmm0\n    je {false_label}\n    leaq nano_true(%rip), %rdi\n    jmp {end_label}\n{false_label}:\n    leaq nano_false(%rip), %rdi\n{end_label}:\n    call puts@PLT\n",
                                stack_offset(slot)
                            ));
                        }
                        Kind::Text => {
                            self.text.push_str(&format!(
                                "    movq {}(%rbp), %rdi\n    call puts@PLT\n",
                                stack_offset(slot)
                            ));
                        }
                    }
                }
                IrInst::Pop => {}
                IrInst::JumpIfFalse(target) => {
                    let slot = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: condição vazia em '{name}'"))?;
                    self.load_stack(slot, "%xmm0");
                    let zero = self.add_float(0.0);
                    self.text.push_str(&format!(
                        "    ucomisd {zero}(%rip), %xmm0\n    je {}\n",
                        labels.get(target)
                            .cloned()
                            .unwrap_or_else(|| format!(".L{symbol}_end"))
                    ));
                }
                IrInst::Jump(target) => {
                    self.text.push_str(&format!(
                        "    jmp {}\n",
                        labels.get(target)
                            .cloned()
                            .unwrap_or_else(|| format!(".L{symbol}_end"))
                    ));
                }
                IrInst::Return => {
                    let slot = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: retorno sem valor em '{name}'"))?;
                    self.load_stack(slot, "%xmm0");
                    self.text.push_str("    movq %rbp, %rsp\n    popq %rbp\n    ret\n");
                }
                _ => unreachable!("unsupported instruction rejected in native stack analysis"),
            }
        }

        self.text.push_str(&format!(
            ".L{symbol}_end:\n    xorpd %xmm0, %xmm0\n    movq %rbp, %rsp\n    popq %rbp\n    ret\n"
        ));
        Ok(())
    }

    fn load_stack(&mut self, slot: usize, reg: &str) {
        self.text.push_str(&format!(
            "    movsd {}(%rbp), {reg}\n",
            stack_offset(slot)
        ));
    }

    fn add_float(&mut self, value: f64) -> String {
        let id = self.float_constants.len();
        self.float_constants.push(value);
        format!(".LCF{id}")
    }

    fn add_text(&mut self, value: &str) -> String {
        let id = self.text_constants.len();
        self.text_constants.push(value.as_bytes().to_vec());
        format!(".LCT{id}")
    }

    fn render(self) -> String {
        let mut text = self.text;
        text.push_str("\n    .section .rodata\n");
        text.push_str("nano_fmt:\n    .byte 37,103,10,0\n");
        text.push_str("nano_true:\n    .byte 116,114,117,101,0\n");
        text.push_str("nano_false:\n    .byte 102,97,108,115,101,0\n");
        for (i, value) in self.float_constants.iter().enumerate() {
            text.push_str(&format!(".LCF{i}:\n    .double {value:.17e}\n"));
        }
        for (i, value) in self.text_constants.iter().enumerate() {
            text.push_str(&format!(".LCT{i}:\n    .byte "));
            for (index, byte) in value.iter().chain(std::iter::once(&0u8)).enumerate() {
                if index > 0 {
                    text.push_str(", ");
                }
                text.push_str(&byte.to_string());
            }
            text.push('\n');
        }
        text.push_str("\n    .section .text\n    .globl main\nmain:\n    subq $8, %rsp\n    call nano_fn__main\n    addq $8, %rsp\n    xorl %eax, %eax\n    ret\n");
        text.push_str("\n    .section .note.GNU-stack,\"\",@progbits\n");
        text
    }
}

fn initial_locals(params: &[String], code: &[IrInst]) -> Result<HashMap<String, usize>, String> {
    let mut locals = HashMap::<String, usize>::new();
    for (index, param) in params.iter().enumerate() {
        if index >= 8 {
            return Err(format!("Nano native: função tem mais de 8 parâmetros"));
        }
        locals.insert(param.clone(), index);
    }
    for inst in code {
        if let IrInst::Store(var) = inst {
            let index = locals.len();
            locals.entry(var.clone()).or_insert(index);
        }
    }
    Ok(locals)
}

fn analyze_stack(
    code: &[IrInst],
    name: &str,
    locals: &HashMap<String, usize>,
    params: &[String],
    function_returns: &HashMap<String, Kind>,
) -> Result<(Vec<Option<Vec<Kind>>>, usize, HashMap<String, Kind>, Option<Kind>), String> {
    use std::collections::VecDeque;

    let mut states: Vec<Option<Vec<Kind>>> = vec![None; code.len()];
    let mut local_kinds = HashMap::<String, Kind>::new();
    for param in params {
        local_kinds.insert(param.clone(), Kind::Number);
    }
    let mut work = VecDeque::new();
    if !code.is_empty() {
        states[0] = Some(Vec::new());
        work.push_back(0usize);
    }

    let mut max_stack = 0usize;
    let mut return_kind: Option<Kind> = None;

    while let Some(ip) = work.pop_front() {
        let state = states[ip]
            .clone()
            .ok_or_else(|| format!("Nano native: estado de stack ausente em '{name}'"))?;

        max_stack = max_stack.max(state.len());
        let mut next = state.clone();

        match &code[ip] {
            IrInst::Const(Value::Number(_)) => {
                next.push(Kind::Number);
            }
            IrInst::Const(Value::Boolean(_)) => {
                next.push(Kind::Boolean);
            }
            IrInst::Const(Value::Text(_)) => {
                next.push(Kind::Text);
            }
            IrInst::Const(_) => {
                return Err(format!("Nano native: constante não suportada em '{name}'"));
            }
            IrInst::Load(var) => {
                if !locals.contains_key(var) {
                    return Err(format!("Nano native: variável '{var}' não é conhecida em '{name}'"));
                }
                let kind = local_kinds.get(var).copied().unwrap_or(Kind::Number);
                next.push(kind);
            }
            IrInst::Store(var) => {
                let value = next.pop().ok_or_else(|| format!("Nano native: Store sem valor em '{name}'"))?;
                if !matches!(value, Kind::Number | Kind::Boolean) {
                    return Err(format!("Nano native: variável '{var}' precisa ser Number ou Boolean"));
                }
                match local_kinds.get(var).copied() {
                    None => { local_kinds.insert(var.clone(), value); }
                    Some(previous) if previous == value => {}
                    Some(previous) => {
                        return Err(format!(
                            "Nano native: variável '{var}' muda de tipo: {:?} -> {:?}",
                            previous, value
                        ));
                    }
                }
            }
            IrInst::Binary(op) => {
                let right = next.pop().ok_or_else(|| format!("Nano native: binary sem direito em '{name}'"))?;
                let left = next.pop().ok_or_else(|| format!("Nano native: binary sem esquerdo em '{name}'"))?;
                match op {
                    Op::Eq | Op::Ne => {
                        if left != right || !matches!(left, Kind::Number | Kind::Boolean) {
                            return Err(format!("Nano native: comparação {:?} exige valores compatíveis", op));
                        }
                        next.push(Kind::Boolean);
                    }
                    Op::Gt | Op::Ge | Op::Lt | Op::Le => {
                        if left != Kind::Number || right != Kind::Number {
                            return Err(format!("Nano native: comparação {:?} exige Numbers", op));
                        }
                        next.push(Kind::Boolean);
                    }
                    Op::And | Op::Or => {
                        if !matches!(left, Kind::Number | Kind::Boolean)
                            || !matches!(right, Kind::Number | Kind::Boolean) {
                            return Err(format!("Nano native: lógica {:?} exige valores booleanos ou numéricos", op));
                        }
                        next.push(Kind::Boolean);
                    }
                    _ => {
                        if left != Kind::Number || right != Kind::Number {
                            return Err(format!("Nano native: operação {:?} exige Numbers", op));
                        }
                        next.push(Kind::Number);
                    }
                }
            }
            IrInst::Unary(op) => {
                let value = next.pop().ok_or_else(|| format!("Nano native: unary sem valor em '{name}'"))?;
                match op {
                    crate::UnaryOp::Neg => {
                        if value != Kind::Number {
                            return Err(format!("Nano native: operador {:?} exige Number", op));
                        }
                        next.push(Kind::Number);
                    }
                    crate::UnaryOp::Not => {
                        if !matches!(value, Kind::Number | Kind::Boolean) {
                            return Err(format!("Nano native: operador {:?} exige valor lógico", op));
                        }
                        next.push(Kind::Boolean);
                    }
                }
            }
            IrInst::FusedMulAdd => {
                for _ in 0..3 {
                    let value = next.pop().ok_or_else(|| format!("Nano native: FMA inválido em '{name}'"))?;
                    if value != Kind::Number {
                        return Err(format!("Nano native: FMA exige Numbers em '{name}'"));
                    }
                }
                next.push(Kind::Number);
            }
            IrInst::Call(callee, count) => {
                if *count > 8 {
                    return Err(format!("Nano native: chamada '{callee}' tem mais de 8 argumentos"));
                }
                for _ in 0..*count {
                    let value = next.pop().ok_or_else(|| format!("Nano native: chamada '{callee}' sem argumentos suficientes"))?;
                    if !matches!(value, Kind::Number | Kind::Boolean) {
                        return Err(format!("Nano native: chamada '{callee}' exige argumentos escalares"));
                    }
                }
                next.push(function_returns.get(callee).copied().unwrap_or(Kind::Number));
            }
            IrInst::Print => {
                next.pop().ok_or_else(|| format!("Nano native: print sem valor em '{name}'"))?;
            }
            IrInst::Pop => {
                next.pop().ok_or_else(|| format!("Nano native: pop inválido em '{name}'"))?;
            }
            IrInst::JumpIfFalse(_) => {
                let condition = next.pop().ok_or_else(|| format!("Nano native: condição vazia em '{name}'"))?;
                if !matches!(condition, Kind::Number | Kind::Boolean) {
                    return Err(format!("Nano native: condição precisa ser Number ou Boolean em '{name}'"));
                }
            }
            IrInst::Jump(_) => {}
            IrInst::Return => {
                let value = next.pop().ok_or_else(|| format!("Nano native: retorno sem valor em '{name}'"))?;
                if !matches!(value, Kind::Number | Kind::Boolean) {
                    return Err(format!("Nano native: retorno de '{name}' deve ser Number ou Boolean"));
                }
                match return_kind {
                    None => return_kind = Some(value),
                    Some(previous) if previous == value => {}
                    Some(previous) => {
                        return Err(format!(
                            "Nano native: função '{name}' retorna tipos diferentes: {:?} e {:?}",
                            previous, value
                        ));
                    }
                }
            }
            IrInst::MakeList(_)
            | IrInst::MakeObject(_)
            | IrInst::Index
            | IrInst::Field(_)
            | IrInst::IterInit
            | IrInst::IterNext(_, _)
            | IrInst::Use(_) => {
                return Err(format!("Nano native: instrução não suportada em '{name}': {:?}", code[ip]));
            }
        }

        max_stack = max_stack.max(next.len());

        let successors: Vec<usize> = match &code[ip] {
            IrInst::Jump(target) => vec![*target],
            IrInst::JumpIfFalse(target) => {
                let mut v = Vec::with_capacity(2);
                if ip + 1 < code.len() { v.push(ip + 1); }
                v.push(*target);
                v
            }
            IrInst::Return => Vec::new(),
            _ => {
                if ip + 1 < code.len() { vec![ip + 1] } else { Vec::new() }
            }
        };

        for successor in successors {
            if successor >= code.len() {
                continue;
            }
            match &states[successor] {
                None => {
                    states[successor] = Some(next.clone());
                    work.push_back(successor);
                }
                Some(existing) if existing == &next => {}
                Some(existing) => {
                    return Err(format!(
                        "Nano native: stack divergente no merge em '{name}' no IP {successor}: {:?} vs {:?}",
                        existing, next
                    ));
                }
            }
        }
    }

    Ok((states, max_stack, local_kinds, return_kind))
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn local_offset(index: usize) -> isize {
    -64 - (index as isize * 8)
}

fn stack_offset(slot: usize) -> isize {
    -2048 - (slot as isize * 8)
}
