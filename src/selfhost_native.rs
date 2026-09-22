use std::collections::HashMap;

use crate::{ir::{IrFunction, IrInst, IrProgram}, Op, UnaryOp, Value};

pub(crate) fn parse(source: &str) -> Result<IrProgram, String> {
    let mut functions = HashMap::new();
    let mut main_code = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_params = Vec::new();
    let mut current_code = Vec::new();
    let mut labels = HashMap::<String, usize>::new();
    let mut unresolved = Vec::<(usize, String, u8, String)>::new();

    let finish = |name: &Option<String>,
                  params: &mut Vec<String>,
                  code: &mut Vec<IrInst>,
                  labels: &mut HashMap<String, usize>,
                  unresolved: &mut Vec<(usize, String, u8, String)>,
                  functions: &mut HashMap<String, IrFunction>,
                  main_code: &mut Vec<IrInst>| -> Result<(), String> {
        let Some(function_name) = name else {
            return Ok(());
        };

        for (index, label, kind, name) in unresolved.drain(..) {
            let target = labels.get(&label)
                .copied()
                .ok_or_else(|| format!("Nano selfhost IR: label '{label}' não existe"))?;
            code[index] = match kind {
                0 => IrInst::Jump(target),
                1 => IrInst::JumpIfFalse(target),
                2 => IrInst::IterNext(name, target),
                _ => return Err("Nano selfhost IR: referência de label inválida".into()),
            };
        }

        if function_name == "_main" {
            *main_code = std::mem::take(code);
        } else {
            functions.insert(function_name.clone(), IrFunction {
                params: std::mem::take(params),
                code: std::mem::take(code),
            });
        }
        labels.clear();
        Ok(())
    };

    for raw in source.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(signature) = line.strip_prefix("function ") {
            if !line.ends_with(')') {
                return Err(format!("Nano selfhost IR: assinatura inválida '{line}'"));
            }
            finish(&current_name, &mut current_params, &mut current_code, &mut labels, &mut unresolved, &mut functions, &mut main_code)?;

            let open = signature.find('(').ok_or_else(|| format!("Nano selfhost IR: assinatura inválida '{line}'"))?;
            let name = signature[..open].trim();
            let params_text = &signature[open + 1..signature.len() - 1];
            current_name = Some(name.to_string());
            current_params = if params_text.trim().is_empty() {
                Vec::new()
            } else {
                params_text.split(',').map(|p| p.trim().to_string()).collect()
            };
            current_code.clear();
            labels.clear();
            unresolved.clear();
            continue;
        }

        if line == "end_function" {
            finish(&current_name, &mut current_params, &mut current_code, &mut labels, &mut unresolved, &mut functions, &mut main_code)?;
            current_name = None;
            continue;
        }

        if current_name.is_none() {
            return Err(format!("Nano selfhost IR: instrução fora de função: '{line}'"));
        }

        if let Some(label) = line.strip_prefix("label ") {
            labels.insert(label.trim().to_string(), current_code.len());
            continue;
        }

        let mut parts = line.splitn(2, ' ');
        let opcode = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim();

        let inst = match opcode {
            "const" => IrInst::Const(parse_const(rest)?),
            "load" => IrInst::Load(rest.to_string()),
            "store" => IrInst::Store(rest.to_string()),
            "set_index" => IrInst::SetIndex,
            "set_field" => IrInst::SetField(rest.to_string()),
            "binary" => IrInst::Binary(parse_op(rest)?),
            "unary" => IrInst::Unary(parse_unary(rest)?),
            "call" => {
                let mut it = rest.rsplitn(2, ' ');
                let count = it.next().ok_or_else(|| "Nano selfhost IR: call sem contagem".to_string())?
                    .parse::<usize>().map_err(|_| "Nano selfhost IR: contagem inválida".to_string())?;
                let name = it.next().unwrap_or("").trim();
                if name.is_empty() { return Err("Nano selfhost IR: call sem nome".into()); }
                IrInst::Call(name.to_string(), count)
            }
            "call_value" => IrInst::CallValue(rest.parse::<usize>().map_err(|_| "Nano selfhost IR: call_value inválido".to_string())?),
            "make_list" => IrInst::MakeList(rest.parse::<usize>().map_err(|_| "Nano selfhost IR: make_list inválido".to_string())?),
            "make_object" => {
                if rest.is_empty() {
                    IrInst::MakeObject(Vec::new())
                } else {
                    IrInst::MakeObject(rest.split(',').map(|v| v.trim().to_string()).collect())
                }
            }
            "index" => IrInst::Index,
            "field" => IrInst::Field(rest.to_string()),
            "print" => IrInst::Print,
            "pop" => IrInst::Pop,
            "return" => IrInst::Return,
            "iter_init" => IrInst::IterInit,
            "iter_next" => {
                let mut it = rest.split_whitespace();
                let name = it.next().ok_or_else(|| "Nano selfhost IR: iter_next sem nome".to_string())?;
                let label = it.next().ok_or_else(|| "Nano selfhost IR: iter_next sem destino".to_string())?;
                let target_index = current_code.len();
                unresolved.push((target_index, label.to_string(), 2, name.to_string()));
                IrInst::IterNext(name.to_string(), usize::MAX)
            }
            "jump" => {
                let index = current_code.len();
                unresolved.push((index, rest.to_string(), 0, String::new()));
                IrInst::Jump(usize::MAX)
            }
            "jump_if_false" => {
                let index = current_code.len();
                unresolved.push((index, rest.to_string(), 1, String::new()));
                IrInst::JumpIfFalse(usize::MAX)
            }
            "use" => IrInst::Use(rest.to_string()),
            "unsupported" | "unsupported_stmt" => {
                return Err(format!("Nano selfhost IR: instrução não suportada '{line}'"));
            }
            "break" => {
                return Err("Nano selfhost IR: break deveria ter sido resolvido pelo compilador".into());
            }
            other => return Err(format!("Nano selfhost IR: opcode desconhecido '{other}'")),
        };
        current_code.push(inst);
    }

    finish(&current_name, &mut current_params, &mut current_code, &mut labels, &mut unresolved, &mut functions, &mut main_code)?;
    Ok(IrProgram { code: main_code, functions })
}

fn parse_const(value: &str) -> Result<Value, String> {
    match value {
        "true" => return Ok(Value::Boolean(true)),
        "false" => return Ok(Value::Boolean(false)),
        "null" => return Ok(Value::Null),
        _ => {}
    }
    if value.starts_with('"') {
        return serde_json::from_str::<String>(value)
            .map(Value::Text)
            .map_err(|e| format!("Nano selfhost IR: string inválida: {e}"));
    }
    value.parse::<f64>()
        .map(Value::Number)
        .map_err(|_| format!("Nano selfhost IR: constante inválida '{value}'"))
}

fn parse_op(value: &str) -> Result<Op, String> {
    match value {
        "+" => Ok(Op::Add), "-" => Ok(Op::Sub), "*" => Ok(Op::Mul), "/" => Ok(Op::Div), "%" => Ok(Op::Mod),
        "==" => Ok(Op::Eq), "!=" => Ok(Op::Ne), ">" => Ok(Op::Gt), ">=" => Ok(Op::Ge),
        "<" => Ok(Op::Lt), "<=" => Ok(Op::Le), "&&" => Ok(Op::And), "||" => Ok(Op::Or),
        other => Err(format!("Nano selfhost IR: operador desconhecido '{other}'")),
    }
}

fn parse_unary(value: &str) -> Result<UnaryOp, String> {
    match value {
        "-" => Ok(UnaryOp::Neg),
        "!" => Ok(UnaryOp::Not),
        other => Err(format!("Nano selfhost IR: unário desconhecido '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::parse;
    
    #[test]
    fn parses_selfhost_function_and_labels() {
        let ir = parse(
            "function _main()\nconst 1\nstore x\nload x\nprint\nend_function\n"
        ).unwrap();
        assert!(ir.code.len() == 4);
        assert!(ir.functions.is_empty());
    }
}
