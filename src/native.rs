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
    Text,
}

pub(crate) fn build(ir: &IrProgram, output: &Path) -> Result<(), String> {
    if !cfg!(target_arch = "x86_64") || !cfg!(target_os = "linux") {
        return Err("Nano: o backend --native atual requer x86_64 Linux.".into());
    }

    let mut module = NativeModule::new();
    module.compile_function("_main", &ir.code, &[])?;
    for (name, function) in &ir.functions {
        module.compile_function(name, &function.code, &function.params)?;
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

    let _ = fs::remove_file(&asm_path);
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
    ) -> Result<(), String> {
        let symbol = format!("nano_fn_{}", sanitize(name));
        let mut locals = HashMap::<String, usize>::new();
        for (index, param) in params.iter().enumerate() {
            if index >= 8 {
                return Err(format!("Nano native: função '{name}' tem mais de 8 parâmetros"));
            }
            locals.insert(param.clone(), index);
        }

        let mut stack: Vec<Kind> = Vec::new();
        let mut max_stack = 0usize;
        for inst in code {
            match inst {
                IrInst::Const(Value::Number(_)) => stack.push(Kind::Number),
                IrInst::Const(Value::Text(_)) => stack.push(Kind::Text),
                IrInst::Const(_) => {
                    return Err(format!("Nano native: constante não suportada em '{name}'"));
                },
                IrInst::Load(var) => {
                    let _ = locals.get(var)
                        .ok_or_else(|| format!("Nano native: variável '{var}' não é conhecida em '{name}'"))?;
                    stack.push(Kind::Number);
                }
                IrInst::Store(var) => {
                    let kind = stack.pop().ok_or_else(|| format!("Nano native: Store inválido em '{name}'"))?;
                    if kind != Kind::Number {
                        return Err(format!("Nano native: variável '{var}' precisa ser Number"));
                    }
                    let slot = locals.len();
                    locals.entry(var.clone()).or_insert(slot);
                }
                IrInst::Binary(op) => {
                    let b = stack.pop().ok_or_else(|| format!("Nano native: stack insuficiente em '{name}'"))?;
                    let a = stack.pop().ok_or_else(|| format!("Nano native: stack insuficiente em '{name}'"))?;
                    if a != Kind::Number || b != Kind::Number {
                        return Err(format!("Nano native: operação {:?} exige Numbers", op));
                    }
                    stack.push(Kind::Number);
                }
                IrInst::Unary(op) => {
                    let value = stack.pop().ok_or_else(|| format!("Nano native: stack insuficiente em '{name}'"))?;
                    if value != Kind::Number {
                        return Err(format!("Nano native: operador {:?} exige Number", op));
                    }
                    stack.push(Kind::Number);
                }
                IrInst::Unary(op) => {
                    let _value_kind = stack.last().copied().ok_or_else(|| format!("Nano native: unary sem valor em '{name}'"))?;
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
                    self.text.push_str(&format!("    movsd %xmm0, {}(%rbp)\n", stack_offset(slot)));
                }
                IrInst::FusedMulAdd => {
                    for _ in 0..3 {
                        if stack.pop().is_none() {
                            return Err(format!("Nano native: FMA inválido em '{name}'"));
                        }
                    }
                    stack.push(Kind::Number);
                }
                IrInst::Call(callee, count) => {
                    if *count > 8 {
                        return Err(format!("Nano native: chamada '{callee}' tem mais de 8 argumentos"));
                    }
                    for _ in 0..*count {
                        if stack.pop() != Some(Kind::Number) {
                            return Err(format!("Nano native: chamada '{callee}' requer argumentos Number"));
                        }
                    }
                    stack.push(Kind::Number);
                }
                IrInst::Print => {
                    let _ = stack.pop().ok_or_else(|| format!("Nano native: print sem valor em '{name}'"))?;
                }
                IrInst::Pop => {
                    let _ = stack.pop().ok_or_else(|| format!("Nano native: pop inválido em '{name}'"))?;
                }
                IrInst::JumpIfFalse(_) => {
                    if stack.pop() != Some(Kind::Number) {
                        return Err(format!("Nano native: condição deve ser Number em '{name}'"));
                    }
                }
                IrInst::Jump(_) => {}
                IrInst::Return => {
                    if stack.pop() != Some(Kind::Number) {
                        return Err(format!("Nano native: retorno de '{name}' deve ser Number"));
                    }
                }
                IrInst::MakeList(_)
                | IrInst::MakeObject(_)
                | IrInst::Index
                | IrInst::Field(_)
                | IrInst::IterInit
                | IrInst::IterNext(_, _)
                | IrInst::Use(_) => {
                    return Err(format!("Nano native: instrução não suportada em '{name}': {:?}", inst));
                }
            }
            max_stack = max_stack.max(stack.len());
        }

        let frame = (((4096 + locals.len().max(1) * 8 + max_stack * 8) + 15) / 16) * 16;
        self.text.push_str(&format!(
            "\n    .text\n    .globl {symbol}\n{symbol}:\n    pushq %rbp\n    movq %rsp, %rbp\n    subq "
        ));
        self.text.push_str(&frame.to_string());
        self.text.push_str(", %rsp\n");

        for index in 0..params.len() {
            self.text.push_str(&format!(
                "    movsd %xmm{index}, {}(%rbp)\n",
                local_offset(index)
            ));
        }

        let mut compile_stack: Vec<Kind> = Vec::new();
        let labels: HashMap<usize, String> = (0..code.len())
            .map(|i| (i, format!(".L{symbol}_{i}")))
            .collect();

        for (ip, inst) in code.iter().enumerate() {
            self.text.push_str(&format!("{}:\n", labels[&ip]));
            match inst {
                IrInst::Const(Value::Number(value)) => {
                    let label = self.add_float(*value);
                    let slot = compile_stack.len();
                    self.text.push_str(&format!(
                        "    movsd {label}(%rip), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(slot)
                    ));
                    compile_stack.push(Kind::Number);
                }
                IrInst::Const(Value::Text(value)) => {
                    let label = self.add_text(value);
                    let slot = compile_stack.len();
                    self.text.push_str(&format!(
                        "    leaq {label}(%rip), %rax\n    movq %rax, {}(%rbp)\n",
                        stack_offset(slot)
                    ));
                    compile_stack.push(Kind::Text);
                }
                IrInst::Load(var) => {
                    let index = *locals.get(var).unwrap();
                    let slot = compile_stack.len();
                    self.text.push_str(&format!(
                        "    movsd {}(%rbp), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        local_offset(index),
                        stack_offset(slot)
                    ));
                    compile_stack.push(Kind::Number);
                }
                IrInst::Store(var) => {
                    let slot = compile_stack.len().checked_sub(1)
                        .ok_or_else(|| "Nano native: Store sem valor".to_string())?;
                    let index = *locals.get(var).unwrap();
                    self.text.push_str(&format!(
                        "    movsd {}(%rbp), %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(slot),
                        local_offset(index)
                    ));
                    compile_stack.pop();
                }
                IrInst::Binary(op) => {
                    let right = compile_stack.len() - 1;
                    let left = compile_stack.len() - 2;
                    self.load_stack(right, "%xmm1");
                    self.load_stack(left, "%xmm0");
                    match op {
                        Op::Add => self.text.push_str("    addsd %xmm1, %xmm0\n"),
                        Op::Sub => self.text.push_str("    subsd %xmm1, %xmm0\n"),
                        Op::Mul => self.text.push_str("    mulsd %xmm1, %xmm0\n"),
                        Op::Div => self.text.push_str("    divsd %xmm1, %xmm0\n"),
                        Op::Mod => self.text.push_str("    call fmod@PLT\n"),
                        Op::And | Op::Or => {
                            let zero = self.add_float(0.0);
                            self.text.push_str(&format!("    ucomisd {zero}(%rip), %xmm0\n"));
                            self.text.push_str("    setne %al\n    movzbl %al, %eax\n");
                            self.text.push_str(&format!("    movsd {zero}(%rip), %xmm0\n    ucomisd {zero}(%rip), %xmm1\n"));
                            self.text.push_str("    setne %cl\n    movzbl %cl, %ecx\n");
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
                    compile_stack.truncate(left + 1);
                }
                IrInst::FusedMulAdd => {
                    let bias = compile_stack.len() - 1;
                    let right = compile_stack.len() - 2;
                    let left = compile_stack.len() - 3;
                    self.load_stack(right, "%xmm1");
                    self.load_stack(left, "%xmm0");
                    self.text.push_str("    mulsd %xmm1, %xmm0\n");
                    self.load_stack(bias, "%xmm1");
                    self.text.push_str(&format!(
                        "    addsd %xmm1, %xmm0\n    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(left)
                    ));
                    compile_stack.truncate(left + 1);
                }
                IrInst::Call(callee, count) => {
                    let start = compile_stack.len() - count;
                    for (arg, reg) in (start..compile_stack.len()).zip(0..8) {
                        self.load_stack(arg, &format!("%xmm{reg}"));
                    }
                    self.text.push_str(&format!(
                        "    call nano_fn_{}\n",
                        sanitize(callee)
                    ));
                    self.text.push_str(&format!(
                        "    movsd %xmm0, {}(%rbp)\n",
                        stack_offset(start)
                    ));
                    compile_stack.truncate(start);
                    compile_stack.push(Kind::Number);
                }
                IrInst::Print => {
                    let slot = compile_stack.len().checked_sub(1)
                        .ok_or_else(|| "Nano native: print sem valor".to_string())?;
                    let kind = compile_stack.pop().unwrap();
                    match kind {
                        Kind::Number => {
                            self.text.push_str(&format!(
                                "    movsd {}(%rbp), %xmm0\n    leaq nano_fmt(%rip), %rdi\n    movl $1, %eax\n    call printf@PLT\n",
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
                IrInst::Pop => {
                    compile_stack.pop().ok_or_else(|| "Nano native: pop inválido".to_string())?;
                }
                IrInst::JumpIfFalse(target) => {
                    let slot = compile_stack.len().checked_sub(1)
                        .ok_or_else(|| "Nano native: condição vazia".to_string())?;
                    compile_stack.pop();
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
                    let slot = compile_stack.len().checked_sub(1)
                        .ok_or_else(|| "Nano native: retorno sem valor".to_string())?;
                    self.load_stack(slot, "%xmm0");
                    self.text.push_str("    movq %rbp, %rsp\n    popq %rbp\n    ret\n");
                }
                _ => unreachable!("unsupported instruction rejected in validation"),
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
        text.push_str("\n    .section .text\n    .globl main\nmain:\n    call nano_fn__main\n    xorl %eax, %eax\n    ret\n");
        text.push_str("\n    .section .note.GNU-stack,\"\",@progbits\n");
        text
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
