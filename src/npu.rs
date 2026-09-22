use serde_json::{json, Value};
use std::{
    env,
    io::{BufRead, BufReader, Write},
    process::{ChildStdin, ChildStdout, Command, Stdio},
    sync::Mutex,
};

use crate::backend::{BackendError, ElementwiseOp, TensorBackend};

const ABI_VERSION: u32 = 1;

struct ProviderProcess {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    _child: std::process::Child,
}

pub(crate) struct NpuBackend {
    provider: Mutex<ProviderProcess>,
}

impl NpuBackend {
    pub(crate) fn new() -> Result<Self, BackendError> {
        let program = env::var("NANO_NPU_PROVIDER")
            .map_err(|_| BackendError(
                "NPU indisponível: defina NANO_NPU_PROVIDER com o caminho do provider NPU.".into()
            ))?;

        let mut child = Command::new(&program)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| BackendError(format!(
                "Nano: não foi possível iniciar provider NPU '{program}': {e}"
            )))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| BackendError("Nano: provider NPU não forneceu stdin".into()))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| BackendError("Nano: provider NPU não forneceu stdout".into()))?;

        let backend = Self {
            provider: Mutex::new(ProviderProcess {
                stdin,
                stdout: BufReader::new(stdout),
                _child: child,
            }),
        };

        {
            let mut process = backend.provider.lock()
                .map_err(|_| BackendError("Nano: lock do provider NPU envenenado".into()))?;
            let response = request_locked(&mut process, json!({
                "id": 0,
                "op": "handshake",
                "abi": ABI_VERSION
            }))?;
            if response.get("abi").and_then(Value::as_u64) != Some(ABI_VERSION as u64) {
                return Err(BackendError(
                    "Nano: provider NPU rejeitou a ABI v1 ou respondeu com ABI incompatível.".into()
                ));
            }
            let required = ["matmul", "elementwise", "fused_mul_add", "reduce", "upload", "read", "release", "transfer"];
            let ops = response.get("ops")
                .and_then(Value::as_array)
                .ok_or_else(|| BackendError("Nano: provider NPU não declarou operações suportadas.".into()))?;
            for op in required {
                if !ops.iter().any(|item| item.as_str() == Some(op)) {
                    return Err(BackendError(format!("Nano: provider NPU não suporta operação obrigatória '{op}'.")));
                }
            }
        }

        Ok(backend)
    }

    fn request(&self, op: &str, payload: Value) -> Result<Value, BackendError> {
        let mut process = self.provider.lock()
            .map_err(|_| BackendError("Nano: lock do provider NPU envenenado".into()))?;
        let mut request = json!({ "id": next_request_id(), "op": op });
        if let (Value::Object(dst), Value::Object(src)) = (&mut request, payload) {
            for (key, value) in src {
                dst.insert(key, value);
            }
        }
        request_locked(&mut process, request)
    }

    fn response_data(&self, op: &str, payload: Value) -> Result<Vec<f32>, BackendError> {
        let response = self.request(op, payload)?;
        let data = response.get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| BackendError(format!("Nano: provider NPU não devolveu data para {op}")))?;
        data.iter()
            .map(|value| value.as_f64().map(|v| v as f32)
                .ok_or_else(|| BackendError(format!("Nano: provider NPU devolveu data inválida em {op}"))))
            .collect()
    }
}

fn request_locked(process: &mut ProviderProcess, request: Value) -> Result<Value, BackendError> {
    let encoded = serde_json::to_string(&request)
        .map_err(|e| BackendError(format!("Nano: erro ao serializar pedido NPU: {e}")))?;
    process.stdin.write_all(encoded.as_bytes())
        .and_then(|_| process.stdin.write_all(b"\n"))
        .and_then(|_| process.stdin.flush())
        .map_err(|e| BackendError(format!("Nano: erro ao enviar pedido NPU: {e}")))?;

    let mut line = String::new();
    let read = process.stdout.read_line(&mut line)
        .map_err(|e| BackendError(format!("Nano: erro ao ler provider NPU: {e}")))?;
    if read == 0 {
        return Err(BackendError("Nano: provider NPU encerrou a conexão.".into()));
    }

    let response: Value = serde_json::from_str(line.trim())
        .map_err(|e| BackendError(format!("Nano: resposta NPU inválida: {e}")))?;

    if response.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(BackendError(
            response.get("error")
                .and_then(Value::as_str)
                .unwrap_or("provider NPU retornou erro sem mensagem")
                .into()
        ));
    }

    Ok(response)
}

fn next_request_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl TensorBackend for NpuBackend {
    fn kind(&self) -> crate::backend::BackendKind {
        crate::backend::BackendKind::Npu
    }

    fn matmul(
        &self,
        left: &[f32],
        left_shape: &[usize],
        right: &[f32],
        right_shape: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        self.response_data("matmul", json!({
            "left": left,
            "left_shape": left_shape,
            "right": right,
            "right_shape": right_shape
        }))
    }

    fn elementwise(
        &self,
        left: &[f32],
        right: &[f32],
        shape: &[usize],
        op: ElementwiseOp,
    ) -> Result<Vec<f32>, BackendError> {
        let op = match op {
            ElementwiseOp::Add => "add",
            ElementwiseOp::Sub => "sub",
            ElementwiseOp::Mul => "mul",
            ElementwiseOp::Div => "div",
        };
        self.response_data("elementwise", json!({
            "left": left,
            "right": right,
            "shape": shape,
            "operator": op
        }))
    }

    fn fused_mul_add(
        &self,
        left: &[f32],
        right: &[f32],
        bias: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        self.response_data("fused_mul_add", json!({
            "left": left,
            "right": right,
            "bias": bias,
            "shape": shape
        }))
    }

    fn reduce(&self, data: &[f32], mean: bool) -> Result<f32, BackendError> {
        let response = self.request("reduce", json!({
            "data": data,
            "mean": mean
        }))?;
        response.get("value").and_then(Value::as_f64)
            .map(|v| v as f32)
            .ok_or_else(|| BackendError("Nano: provider NPU não devolveu value em reduce".into()))
    }

    fn sync_tensor(&self, id: u64, data: &[f32]) -> Result<(), BackendError> {
        self.request("upload", json!({
            "tensor_id": id,
            "data": data
        })).map(|_| ())
    }

    fn release_tensor(&self, id: u64) -> Result<(), BackendError> {
        self.request("release", json!({ "tensor_id": id })).map(|_| ())
    }

    fn read_tensor(&self, id: u64, _elements: usize) -> Result<Vec<f32>, BackendError> {
        self.response_data("read", json!({ "tensor_id": id }))
    }

    fn transfer(&self, data: &[f32]) -> Result<Vec<f32>, BackendError> {
        self.response_data("transfer", json!({ "data": data }))
    }
}

#[cfg(test)]
mod tests {
    use super::ABI_VERSION;

    #[test]
    fn provider_abi_is_stable() {
        assert_eq!(ABI_VERSION, 1);
    }
}
