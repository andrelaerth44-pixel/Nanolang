use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

use crate::{ir::{self, DebugSession, DebugStop}, Lexer, Parser, Semantic};

pub(crate) fn serve() -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let mut out = io::stdout();
    let mut session: Option<DebugSession> = None;
    let mut source_path = String::new();
    let mut source_text = String::new();

    loop {
        let Some(request) = read_message(&mut reader)? else { break };
        let command = request.get("command").and_then(Value::as_str).unwrap_or("");
        let seq = request.get("seq").and_then(Value::as_u64).unwrap_or(0);

        match command {
            "initialize" => respond(&mut out, seq, command, json!({
                "supportsConfigurationDoneRequest": true,
                "supportsFunctionBreakpoints": false,
                "supportsConditionalBreakpoints": false,
                "supportsEvaluateForHovers": true,
                "supportsStepBack": false,
                "supportsStepInRequest": true,
                "supportsNextRequest": true,
                "supportsStepOutRequest": true,
                "supportsTerminateRequest": true,
                "supportsLoadedSourcesRequest": true,
                "supportsBreakpointLocationsRequest": true,
                "supportsSetVariable": false,
                "supportsExceptionInfoRequest": true,
                "supportsCompletionsRequest": false,
                "supportsSourceRequest": true
            }))?,
            "launch" => {
                let program = request.pointer("/arguments/program")
                    .or_else(|| request.pointer("/arguments/programPath"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Nano DAP: launch requer arguments.program".to_string())?;

                let text = fs::read_to_string(program)
                    .map_err(|e| format!("Nano DAP: não foi possível ler '{program}': {e}"))?;
                let tokens = Lexer::new(&text).lex()?;
                let parsed = Parser::new(tokens).program()?;
                let mut semantic = Semantic::new();
                semantic.check(&parsed)?;
                let mut compiler = ir::Compiler::new();
                let ir_program = compiler.compile(&parsed)?;
                let mut optimizer = ir::Optimizer::new();
                let ir_program = optimizer.optimize_program(ir_program);

                source_path = program.to_string();
                source_text = text;
                session = Some(DebugSession::new(&ir_program)?);

                respond(&mut out, seq, command, Value::Null)?;
                send_event(&mut out, "initialized", Value::Null)?;
            }
            "setBreakpoints" => {
                let lines = request.pointer("/arguments/breakpoints")
                    .and_then(Value::as_array)
                    .map(|items| items.iter().filter_map(|item| item.get("line").and_then(Value::as_u64).map(|v| v as usize)).collect::<Vec<_>>())
                    .unwrap_or_default();

                if let Some(active) = session.as_mut() {
                    active.set_breakpoints(&lines);
                }

                let result = lines.into_iter().map(|line| json!({
                    "verified": session.is_some(),
                    "line": line,
                    "source": {"path": source_path}
                })).collect::<Vec<_>>();
                respond(&mut out, seq, command, json!({"breakpoints": result}))?;
            }
            "configurationDone" => respond(&mut out, seq, command, Value::Null)?,
            "threads" => respond(&mut out, seq, command, json!({"threads":[{"id":1,"name":"nano-main"}]}))?,
            "stackTrace" => {
                let frame = if let Some(active) = session.as_ref() {
                    json!({
                        "id": 1,
                        "name": "_main",
                        "line": active.current_line(),
                        "column": 1,
                        "source": {"name": Path::new(&source_path).file_name().and_then(|v| v.to_str()).unwrap_or("main.nano"), "path": source_path}
                    })
                } else {
                    json!({
                        "id": 1,
                        "name": "_main",
                        "line": 1,
                        "column": 1
                    })
                };
                respond(&mut out, seq, command, json!({"stackFrames":[frame],"totalFrames":1}))?;
            }
            "scopes" => respond(&mut out, seq, command, json!({
                "scopes": [
                    {"name":"Locals","presentationHint":"locals","variablesReference":1,"expensive":false}
                ]
            }))?,
            "variables" => {
                let mut variables = Vec::new();
                if request.pointer("/arguments/variablesReference").and_then(Value::as_u64) == Some(1) {
                    if let Some(active) = session.as_ref() {
                        for (name, value) in active.variables() {
                            variables.push(json!({
                                "name": name,
                                "value": value.show(),
                                "type": value_type(&value),
                                "variablesReference": 0
                            }));
                        }
                    }
                }
                respond(&mut out, seq, command, json!({"variables":variables}))?;
            }
            "evaluate" => {
                let expression = request.pointer("/arguments/expression").and_then(Value::as_str).unwrap_or("");
                let result = session.as_ref()
                    .and_then(|active| active.evaluate(expression))
                    .map(|value| json!({"result":value.show(),"type":value_type(&value),"variablesReference":0}))
                    .unwrap_or_else(|| json!({"result":"<unavailable>","variablesReference":0}));
                respond(&mut out, seq, command, result)?;
            }
            "continue" => {
                respond(&mut out, seq, command, json!({"allThreadsContinued":true}))?;
                run_until_stop(&mut out, session.as_mut())?;
            }
            "next" | "stepIn" | "stepOut" => {
                respond(&mut out, seq, command, Value::Null)?;
                run_one_step(&mut out, session.as_mut())?;
            }
            "pause" => {
                respond(&mut out, seq, command, Value::Null)?;
                send_event(&mut out, "stopped", json!({"reason":"pause","threadId":1,"allThreadsStopped":true}))?;
            }
            "breakpointLocations" => {
                let max = session.as_ref().map(|s| s.current_line().max(1)).unwrap_or(1);
                let locations = (1..=max).map(|line| json!({"line":line})).collect::<Vec<_>>();
                respond(&mut out, seq, command, json!({"breakpoints":locations}))?;
            }
            "loadedSources" => {
                respond(&mut out, seq, command, json!({"sources":[{
                    "name":Path::new(&source_path).file_name().and_then(|v| v.to_str()).unwrap_or("main.nano"),
                    "path":source_path,
                    "sourceReference":1
                }]}))?;
            }
            "source" => respond(&mut out, seq, command, json!({"content":source_text,"mimeType":"text/x-nano"}))?,
            "exceptionInfo" => respond(&mut out, seq, command, json!({
                "exceptionId":"nano",
                "description":"Nano runtime exception information is exposed through stderr/diagnostics."
            }))?,
            "disconnect" | "terminate" => {
                respond(&mut out, seq, command, Value::Null)?;
                send_event(&mut out, "terminated", Value::Null)?;
                break;
            }
            _ => respond_error(&mut out, seq, command, &format!("Nano DAP: comando '{command}' não implementado"))?,
        }
    }

    Ok(())
}

fn run_one_step(out: &mut impl Write, session: Option<&mut DebugSession>) -> Result<(), String> {
    let Some(session) = session else {
        return send_event(out, "terminated", Value::Null);
    };
    match session.step()? {
        DebugStop::Exited => send_event(out, "terminated", Value::Null),
        DebugStop::Breakpoint => send_event(out, "stopped", json!({"reason":"breakpoint","threadId":1,"allThreadsStopped":true})),
        DebugStop::Step => send_event(out, "stopped", json!({"reason":"step","threadId":1,"allThreadsStopped":true})),
    }
}

fn run_until_stop(out: &mut impl Write, session: Option<&mut DebugSession>) -> Result<(), String> {
    let Some(session) = session else {
        return send_event(out, "terminated", Value::Null);
    };
    loop {
        match session.step()? {
            DebugStop::Exited => return send_event(out, "terminated", Value::Null),
            DebugStop::Breakpoint => return send_event(out, "stopped", json!({"reason":"breakpoint","threadId":1,"allThreadsStopped":true})),
            DebugStop::Step => continue,
        }
    }
}

fn value_type(value: &crate::Value) -> &'static str {
    match value {
        crate::Value::Number(_) => "Number",
        crate::Value::Text(_) => "Text",
        crate::Value::Boolean(_) => "Boolean",
        crate::Value::Function(_) => "Function",
        crate::Value::List(_) => "List",
        crate::Value::Object(_) => "Object",
        crate::Value::Tensor(_) => "Tensor",
        crate::Value::Null => "Null",
    }
}

fn read_message(reader: &mut impl BufRead) -> Result<Option<Value>, String> {
    let mut content_length = None;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).map_err(|e| format!("Nano DAP: {e}"))? == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['','
']);
        if trimmed.is_empty() { break; }
        if let Some((key,value)) = trimmed.split_once(':') {
            if key.eq_ignore_ascii_case("Content-Length") {
                content_length = value.trim().parse::<usize>().ok();
            }
        }
    }
    let Some(length) = content_length else { return Ok(None); };
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).map_err(|e| format!("Nano DAP: {e}"))?;
    serde_json::from_slice(&body).map_err(|e| format!("Nano DAP: JSON: {e}")).map(Some)
}

fn write_message(out: &mut impl Write, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| format!("Nano DAP: JSON encode: {e}"))?;
    write!(out, "Content-Length: {}

", bytes.len()).map_err(|e| format!("Nano DAP: {e}"))?;
    out.write_all(&bytes).map_err(|e| format!("Nano DAP: {e}"))?;
    out.flush().map_err(|e| format!("Nano DAP: {e}"))
}

fn respond(out: &mut impl Write, request_seq: u64, command: &str, body: Value) -> Result<(), String> {
    write_message(out, &json!({
        "seq":0,
        "type":"response",
        "request_seq":request_seq,
        "success":true,
        "command":command,
        "body":body
    }))
}

fn respond_error(out: &mut impl Write, request_seq: u64, command: &str, message: &str) -> Result<(), String> {
    write_message(out, &json!({
        "seq":0,
        "type":"response",
        "request_seq":request_seq,
        "success":false,
        "command":command,
        "message":message
    }))
}

fn send_event(out: &mut impl Write, event: &str, body: Value) -> Result<(), String> {
    write_message(out, &json!({
        "seq":0,
        "type":"event",
        "event":event,
        "body":body
    }))
}
