use std::{
    collections::HashMap,
    fs,
    path::Path,
    process::Command,
};

use crate::{ir::{IrInst, IrProgram}, Op, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Unknown,
    Number,
    Boolean,
    Text,
    Function,
}

pub(crate) fn build(ir: &IrProgram, output: &Path) -> Result<(), String> {
    if !cfg!(target_arch = "x86_64") || !cfg!(target_os = "linux") {
        return Err("Nano: o backend --native atual requer x86_64 Linux.".into());
    }

    let mut module = NativeModule::new();
    let mut function_params = HashMap::<String, Vec<Kind>>::new();
    let mut function_returns = HashMap::<String, Kind>::new();
    for (name, function) in &ir.functions {
        function_params.insert(name.clone(), vec![Kind::Unknown; function.params.len()]);
        function_returns.insert(name.clone(), Kind::Unknown);
    }

    for _ in 0..64 {
        let mut changed = false;

        for (name, function) in &ir.functions {
            let params = function_params.get(name).cloned().unwrap_or_default();
            let locals = initial_locals(&function.params, &function.code)?;
            let (_, _, _, return_kind, calls) = analyze_stack(
                &function.code,
                name,
                &locals,
                &function.params,
                &params,
                &function_returns,
            )?;

            if let Some(kind) = return_kind.filter(|kind| *kind != Kind::Unknown) {
                let previous = function_returns.get(name).copied().unwrap_or(Kind::Unknown);
                let merged = merge_kind(previous, kind).map_err(|_| {
                    format!("Nano native: função '{name}' possui retornos incompatíveis")
                })?;
                if merged != previous {
                    function_returns.insert(name.clone(), merged);
                    changed = true;
                }
            }

            for (callee, args) in calls {
                let Some(target) = function_params.get_mut(&callee) else { continue; };
                if target.len() != args.len() { continue; }
                for (index, kind) in args.into_iter().enumerate() {
                    let previous = target[index];
                    let merged = merge_kind(previous, kind).map_err(|_| {
                        format!("Nano native: argumento {} de '{callee}' recebe tipos incompatíveis", index + 1)
                    })?;
                    if merged != previous {
                        target[index] = merged;
                        changed = true;
                    }
                }
            }
        }

        let main_locals = initial_locals(&[], &ir.code)?;
        let (_, _, _, _, calls) = analyze_stack(
            &ir.code,
            "_main",
            &main_locals,
            &[],
            &[],
            &function_returns,
        )?;
        for (callee, args) in calls {
            let Some(target) = function_params.get_mut(&callee) else { continue; };
            if target.len() != args.len() { continue; }
            for (index, kind) in args.into_iter().enumerate() {
                let previous = target[index];
                let merged = merge_kind(previous, kind).map_err(|_| {
                    format!("Nano native: argumento {} de '{callee}' recebe tipos incompatíveis", index + 1)
                })?;
                if merged != previous {
                    target[index] = merged;
                    changed = true;
                }
            }
        }

        if !changed { break; }
    }

    for (name, params) in &function_params {
        if params.iter().any(|kind| *kind == Kind::Unknown) {
            return Err(format!("Nano native: não foi possível inferir os tipos dos parâmetros de '{name}'"));
        }
    }
    for (name, kind) in &function_returns {
        if *kind == Kind::Unknown {
            return Err(format!("Nano native: não foi possível inferir o tipo de retorno de '{name}'"));
        }
    }

    module.compile_function("_main", &ir.code, &[], &[], &function_returns)?;
    for (name, function) in &ir.functions {
        let params = function_params.get(name).cloned().unwrap_or_default();
        module.compile_function(name, &function.code, &function.params, &params, &function_returns)?;
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
        param_kinds: &[Kind],
        function_returns: &HashMap<String, Kind>,
    ) -> Result<(), String> {
        let symbol = format!("nano_fn_{}", sanitize(name));
        let locals = initial_locals(params, code)?;
        let (entry_states, max_stack, _local_kinds, _return_kind, _calls) =
            analyze_stack(code, name, &locals, params, param_kinds, function_returns)?;

        let frame = (((4096 + locals.len().max(1) * 8 + max_stack * 8) + 15) / 16) * 16;
        self.text.push_str(&format!(
            "\n    .text\n    .globl {symbol}\n{symbol}:\n    pushq %rbp\n    movq %rsp, %rbp\n    subq "
        ));
        self.text.push_str("$");
        self.text.push_str(&frame.to_string());
        self.text.push_str(", %rsp\n");

        if param_kinds.len() != params.len() {
            return Err(format!("Nano native: assinatura inconsistente de '{name}'"));
        }
        let float_regs = ["%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"];
        let int_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9", "%r10", "%r11"];
        let mut float_index = 0usize;
        let mut int_index = 0usize;
        for (index, kind) in param_kinds.iter().copied().enumerate() {
            match kind {
                Kind::Number | Kind::Boolean => {
                    self.text.push_str(&format!(
                        "    movsd {}, {}(%rbp)\n",
                        float_regs[float_index],
                        local_offset(index)
                    ));
                    float_index += 1;
                }
                Kind::Text | Kind::Function => {
                    self.text.push_str(&format!(
                        "    movq {}, {}(%rbp)\n",
                        int_regs[int_index],
                        local_offset(index)
                    ));
                    int_index += 1;
                }
                Kind::Unknown => {
                    return Err(format!("Nano native: tipo de parâmetro desconhecido em '{name}'"));
                }
            }
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
                IrInst::Const(Value::Function(function)) => {
                    self.text.push_str(&format!(
                        "    leaq nano_fn_{}(%rip), %rax\n    movq %rax, {}(%rbp)\n",
                        sanitize(function),
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
                    let kind = entry_states[ip].as_ref().unwrap().last().copied().unwrap();
                    match kind {
                        Kind::Text | Kind::Function => self.text.push_str(&format!(
                            "    movq {}(%rbp), %rax\n    movq %rax, {}(%rbp)\n",
                            local_offset(index),
                            stack_offset(depth)
                        )),
                        Kind::Number | Kind::Boolean => self.text.push_str(&format!(
                            "    movsd {}(%rbp), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                            local_offset(index),
                            stack_offset(depth)
                        )),
                        Kind::Unknown => {
                            return Err(format!("Nano native: tipo desconhecido ao carregar '{var}' em '{name}'"));
                        }
                    }
                }
                IrInst::Store(var) => {
                    let slot = depth.checked_sub(1)
                        .ok_or_else(|| format!("Nano native: Store sem valor em '{name}'"))?;
                    let index = *locals.get(var).unwrap();
                    let kind = entry_states[ip].as_ref().unwrap().last().copied().unwrap();
                    match kind {
                        Kind::Text | Kind::Function => self.text.push_str(&format!(
                            "    movq {}(%rbp), %rax\n    movq %rax, {}(%rbp)\n",
                            stack_offset(slot),
                            local_offset(index)
                        )),
                        Kind::Number | Kind::Boolean => self.text.push_str(&format!(
                            "    movsd {}(%rbp), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                            stack_offset(slot),
                            local_offset(index)
                        )),
                        Kind::Unknown => {
                            return Err(format!("Nano native: tipo desconhecido ao armazenar '{var}' em '{name}'"));
                        }
                    }
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
                    let float_regs = ["%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"];
                    let int_regs = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9", "%r10", "%r11"];
                    let arg_kinds = &entry_states[ip].as_ref().unwrap()[start..depth];
                    let mut float_index = 0usize;
                    let mut int_index = 0usize;
                    for (offset, kind) in arg_kinds.iter().copied().enumerate() {
                        let arg = start + offset;
                        match kind {
                            Kind::Number | Kind::Boolean => {
                                self.load_stack(arg, float_regs[float_index]);
                                float_index += 1;
                            }
                            Kind::Text | Kind::Function => {
                                self.text.push_str(&format!(
                                    "    movq {}(%rbp), {}\n",
                                    stack_offset(arg),
                                    int_regs[int_index]
                                ));
                                int_index += 1;
                            }
                            Kind::Unknown => {
                                return Err(format!("Nano native: chamada '{callee}' contém um argumento de tipo desconhecido"));
                            }
                        }
                    }
                    self.text.push_str(&format!("    call nano_fn_{}\n", sanitize(callee)));
                    match function_returns.get(callee).copied().unwrap_or(Kind::Number) {
                        Kind::Function => self.text.push_str(&format!(
                            "    movq %rax, {}(%rbp)\n",
                            stack_offset(start)
                        )),
                        Kind::Text => self.text.push_str(&format!(
                            "    movq %rax, {}(%rbp)\n",
                            stack_offset(start)
                        )),
                        _ => self.text.push_str(&format!(
                            "    movsd %xmm0, {}(%rbp)\n",
                            stack_offset(start)
                        )),
                    }
                }
                IrInst::CallValue(count) => {
                    let function_slot = depth.checked_sub(*count + 1)
                        .ok_or_else(|| format!("Nano native: chamada indireta sem alvo em '{name}'"))?;

                    let args_start = function_slot + 1;
                    for (arg, reg) in (args_start..depth).zip(0..8) {
                        let kind = entry_states[ip].as_ref().unwrap()[arg];
                        if !matches!(kind, Kind::Number | Kind::Boolean) {
                            return Err(format!("Nano native: chamada indireta aceita apenas argumentos escalares em '{name}'"));
                        }
                        self.load_stack(arg, &format!("%xmm{reg}"));
                    }

                    self.text.push_str(&format!(
                        "    movq {}(%rbp), %rax\n    call *%rax\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(function_slot),
                        stack_offset(function_slot)
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
                        Kind::Function => {
                            self.text.push_str("    leaq nano_function(%rip), %rdi\n    call puts@PLT\n");
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
                    let kind = entry_states[ip].as_ref().unwrap().last().copied().unwrap();
                    match kind {
                        Kind::Function | Kind::Text => self.text.push_str(&format!(
                            "    movq {}(%rbp), %rax\n",
                            stack_offset(slot)
                        )),
                        _ => self.load_stack(slot, "%xmm0"),
                    }
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
        text.push_str("nano_function:\n    .byte 60,102,117,110,99,116,105,111,110,62,0\n");
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
    param_kinds: &[Kind],
    function_returns: &HashMap<String, Kind>,
) -> Result<(Vec<Option<Vec<Kind>>>, usize, HashMap<String, Kind>, Option<Kind>, Vec<(String, Vec<Kind>)>), String> {
    use std::collections::VecDeque;

    let mut states: Vec<Option<Vec<Kind>>> = vec![None; code.len()];
    let mut local_kinds = HashMap::<String, Kind>::new();
    if param_kinds.len() != params.len() {
        return Err(format!("Nano native: assinatura inconsistente de '{name}'"));
    }
    for (param, kind) in params.iter().zip(param_kinds.iter().copied()) {
        local_kinds.insert(param.clone(), kind);
    }
    let mut calls = Vec::<(String, Vec<Kind>)>::new();
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
            IrInst::Const(Value::Function(_)) => {
                next.push(Kind::Function);
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
                if !matches!(value, Kind::Number | Kind::Boolean | Kind::Function | Kind::Text) {
                    return Err(format!("Nano native: variável '{var}' tem um tipo que o backend não suporta"));
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
                        if left == Kind::Unknown || right == Kind::Unknown {
                            next.push(Kind::Boolean);
                        } else if left != right || !matches!(left, Kind::Number | Kind::Boolean) {
                            return Err(format!("Nano native: comparação {:?} exige valores compatíveis", op));
                        } else {
                            next.push(Kind::Boolean);
                        }
                    }
                    Op::Gt | Op::Ge | Op::Lt | Op::Le => {
                        if (left != Kind::Unknown && left != Kind::Number)
                            || (right != Kind::Unknown && right != Kind::Number) {
                            return Err(format!("Nano native: comparação {:?} exige Numbers", op));
                        }
                        next.push(Kind::Boolean);
                    }
                    Op::And | Op::Or => {
                        if (!matches!(left, Kind::Unknown | Kind::Number | Kind::Boolean))
                            || (!matches!(right, Kind::Unknown | Kind::Number | Kind::Boolean)) {
                            return Err(format!("Nano native: lógica {:?} exige valores booleanos ou numéricos", op));
                        }
                        next.push(Kind::Boolean);
                    }
                    _ => {
                        if (left != Kind::Unknown && left != Kind::Number)
                            || (right != Kind::Unknown && right != Kind::Number) {
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
                        if value != Kind::Unknown && value != Kind::Number {
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
                    if value != Kind::Unknown && value != Kind::Number {
                        return Err(format!("Nano native: FMA exige Numbers em '{name}'"));
                    }
                }
                next.push(Kind::Number);
            }
            IrInst::Call(callee, count) => {
                if *count > 8 {
                    return Err(format!("Nano native: chamada '{callee}' tem mais de 8 argumentos"));
                }
                let start = next.len().checked_sub(*count)
                    .ok_or_else(|| format!("Nano native: chamada '{callee}' sem argumentos suficientes"))?;
                let arg_kinds = next[start..].to_vec();
                for _ in 0..*count {
                    next.pop().ok_or_else(|| format!("Nano native: chamada '{callee}' sem argumentos suficientes"))?;
                }
                calls.push((callee.clone(), arg_kinds));
                next.push(function_returns.get(callee).copied().unwrap_or(Kind::Number));
            }
            IrInst::CallValue(count) => {
                if *count > 8 {
                    return Err(format!("Nano native: chamada indireta tem mais de 8 argumentos em '{name}'"));
                }
                for _ in 0..*count {
                    let value = next.pop().ok_or_else(|| format!("Nano native: chamada indireta sem argumento em '{name}'"))?;
                    if !matches!(value, Kind::Number | Kind::Boolean) {
                        return Err(format!("Nano native: chamada indireta aceita apenas argumentos escalares em '{name}'"));
                    }
                }
                let target = next.pop().ok_or_else(|| format!("Nano native: chamada indireta sem alvo em '{name}'"))?;
                if target != Kind::Function {
                    return Err(format!("Nano native: alvo de chamada indireta deve ser Function em '{name}'"));
                }
                next.push(Kind::Number);
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
                if !matches!(value, Kind::Unknown | Kind::Number | Kind::Boolean | Kind::Text | Kind::Function) {
                    return Err(format!("Nano native: retorno de '{name}' tem um tipo que o backend não suporta"));
                }
                if value != Kind::Unknown {
                    match return_kind {
                        None => return_kind = Some(value),
                        Some(Kind::Unknown) => return_kind = Some(value),
                        Some(previous) if previous == value => {}
                        Some(previous) => {
                            return Err(format!(
                                "Nano native: função '{name}' retorna tipos diferentes: {:?} e {:?}",
                                previous, value
                            ));
                        }
                    }
                }
            }
            IrInst::MakeList(_)
            | IrInst::MakeObject(_)
            | IrInst::Index
            | IrInst::Field(_)
            | IrInst::SetIndex
            | IrInst::SetField(_)
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

    Ok((states, max_stack, local_kinds, return_kind, calls))
}

fn merge_kind(previous: Kind, next: Kind) -> Result<Kind, ()> {
    match (previous, next) {
        (Kind::Unknown, kind) => Ok(kind),
        (kind, Kind::Unknown) => Ok(kind),
        (a, b) if a == b => Ok(a),
        _ => Err(()),
    }
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
