use std::io::{Read, Write};
use std::net::TcpStream;
use std::collections::HashMap;
use crate::Value;

pub(crate) fn request(
    method: &str,
    host: &str,
    port: u16,
    path: &str,
    headers: &HashMap<String, String>,
    body: &str,
) -> Result<Value, String> {
    let method = method.to_ascii_uppercase();
    let mut stream = TcpStream::connect((host, port))
        .map_err(|e| format!("Nano HTTP: conexão {host}:{port}: {e}"))?;

    let mut request = format!(
        "{method} {path} HTTP/1.1
Host: {host}
Connection: close
User-Agent: Nano/1.0
"
    );
    for (key, value) in headers {
        request.push_str(key);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("
");
    }
    if !body.is_empty() {
        request.push_str(&format!("Content-Length: {}
", body.as_bytes().len()));
    }
    request.push_str("
");
    request.push_str(body);

    stream.write_all(request.as_bytes())
        .map_err(|e| format!("Nano HTTP: envio: {e}"))?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)
        .map_err(|e| format!("Nano HTTP: leitura: {e}"))?;
    let response = String::from_utf8_lossy(&raw);
    parse_response(&response)
}

fn parse_response(response: &str) -> Result<Value, String> {
    let (header_text, body) = response.split_once("

")
        .ok_or_else(|| "Nano HTTP: resposta inválida".to_string())?;
    let mut lines = header_text.lines();
    let status_line = lines.next().ok_or_else(|| "Nano HTTP: status ausente".to_string())?;
    let mut status_parts = status_line.splitn(3, ' ');
    let _version = status_parts.next().unwrap_or("");
    let status = status_parts.next()
        .and_then(|code| code.parse::<f64>().ok())
        .ok_or_else(|| "Nano HTTP: código de status inválido".to_string())?;
    let reason = status_parts.next().unwrap_or("").to_string();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(key.trim().to_ascii_lowercase(), Value::Text(value.trim().to_string()));
        }
    }

    let mut result = HashMap::new();
    result.insert("status".into(), Value::Number(status));
    result.insert("reason".into(), Value::Text(reason));
    result.insert("headers".into(), Value::Object(headers));
    result.insert("body".into(), Value::Text(body.to_string()));
    Ok(Value::Object(result))
}

#[cfg(test)]
mod tests {
    use super::parse_response;

    #[test]
    fn parses_http_response() {
        let value = parse_response("HTTP/1.1 201 Created
Content-Type: text/plain

hello").unwrap();
        let Value::Object(object) = value else { panic!("not object") };
        assert_eq!(object.get("status"), Some(&Value::Number(201.0)));
        assert_eq!(object.get("body"), Some(&Value::Text("hello".into())));
    }
}
