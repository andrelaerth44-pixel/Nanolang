use std::{collections::HashMap, io::{self, BufRead}, io::Write};

fn main() {
    let stdin = io::stdin();
    let mut tensors: HashMap<u64, Vec<f32>> = HashMap::new();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(request) = serde_json::from_str::<serde_json::Value>(&line) else {
            respond(serde_json::json!({"ok": false, "error": "invalid json"}));
            continue;
        };

        let op = request.get("op").and_then(|v| v.as_str()).unwrap_or("");
        let response = match op {
            "handshake" => serde_json::json!({"ok": true, "abi": 1, "provider": "nano-reference-npu",
                "ops": ["matmul","elementwise","fused_mul_add","reduce","upload","read","release","transfer"]}),
            "transfer" => {
                let data = floats(&request["data"]);
                serde_json::json!({"ok": true, "data": data})
            }
            "upload" => {
                let id = request["tensor_id"].as_u64().unwrap_or(0);
                tensors.insert(id, floats(&request["data"]));
                serde_json::json!({"ok": true})
            }
            "read" => {
                let id = request["tensor_id"].as_u64().unwrap_or(0);
                serde_json::json!({"ok": true, "data": tensors.get(&id).cloned().unwrap_or_default()})
            }
            "release" => {
                let id = request["tensor_id"].as_u64().unwrap_or(0);
                tensors.remove(&id);
                serde_json::json!({"ok": true})
            }
            "reduce" => {
                let data = floats(&request["data"]);
                let sum: f32 = data.iter().copied().sum();
                let mean = request.get("mean").and_then(|v| v.as_bool()).unwrap_or(false);
                let value = if mean && !data.is_empty() { sum / data.len() as f32 } else { sum };
                serde_json::json!({"ok": true, "value": value})
            }
            "elementwise" => {
                let a = floats(&request["left"]);
                let b = floats(&request["right"]);
                let op = request.get("operator").and_then(|v| v.as_str()).unwrap_or("");
                let data = a.into_iter().zip(b).map(|(x, y)| match op {
                    "add" => x + y,
                    "sub" => x - y,
                    "mul" => x * y,
                    "div" => x / y,
                    _ => f32::NAN,
                }).collect::<Vec<_>>();
                serde_json::json!({"ok": true, "data": data})
            }
            "fused_mul_add" => {
                let a = floats(&request["left"]);
                let b = floats(&request["right"]);
                let c = floats(&request["bias"]);
                let data = a.into_iter().zip(b).zip(c).map(|((x, y), z)| x * y + z).collect::<Vec<_>>();
                serde_json::json!({"ok": true, "data": data})
            }
            "matmul" => {
                let a = floats(&request["left"]);
                let b = floats(&request["right"]);
                let ashape = dims(&request["left_shape"]);
                let bshape = dims(&request["right_shape"]);
                match (ashape.as_slice(), bshape.as_slice()) {
                    ([m, k], [k2, n]) if k == k2 => {
                        let expected_a = m * k;
                        let expected_b = k2 * n;
                        if a.len() != expected_a || b.len() != expected_b {
                            serde_json::json!({
                                "ok": false,
                                "error": format!(
                                    "matmul input size mismatch: left {} (expected {}), right {} (expected {})",
                                    a.len(), expected_a, b.len(), expected_b
                                )
                            })
                        } else {
                        let mut out = vec![0.0f32; m * n];
                        for i in 0..*m {
                            for j in 0..*n {
                                let mut acc = 0.0f32;
                                for p in 0..*k {
                                    acc += a[i * *k + p] * b[p * *n + j];
                                }
                                out[i * *n + j] = acc;
                            }
                        }
                        serde_json::json!({"ok": true, "data": out})
                        }
                    }
                    _ => serde_json::json!({"ok": false, "error": "matmul requires rank-2 compatible matrices"}),
                }
            }
            _ => serde_json::json!({"ok": false, "error": format!("unsupported op: {op}")}),
        };
        respond(response);
    }
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value.as_array().map(|items| items.iter().filter_map(|v| v.as_f64().map(|n| n as f32)).collect()).unwrap_or_default()
}

fn dims(value: &serde_json::Value) -> Vec<usize> {
    value.as_array().map(|items| items.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()).unwrap_or_default()
}

fn respond(value: serde_json::Value) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{}", value);
    let _ = out.flush();
}
