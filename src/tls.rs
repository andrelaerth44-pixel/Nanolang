use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use rustls::{pki_types::ServerName, ClientConfig, ClientConnection, RootCertStore, StreamOwned};
use webpki_roots::TLS_SERVER_ROOTS;

pub(crate) fn https_get(host: &str, port: u16, path: &str) -> Result<String, String> {
    let roots = RootCertStore::from_iter(TLS_SERVER_ROOTS.iter().cloned());
    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let server_name = ServerName::try_from(host.to_owned())
        .map_err(|_| format!("Nano TLS: hostname inválido '{host}'"))?;
    let connection = ClientConnection::new(Arc::new(config), server_name)
        .map_err(|error| format!("Nano TLS: handshake inicial: {error}"))?;

    let tcp = TcpStream::connect((host, port))
        .map_err(|error| format!("Nano TLS: conexão {host}:{port}: {error}"))?;
    let mut stream = StreamOwned::new(connection, tcp);

    let path = if path.is_empty() { "/" } else { path };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: Nano/1.0\r\n\r\n"
    );

    stream.write_all(request.as_bytes())
        .map_err(|error| format!("Nano TLS: envio: {error}"))?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)
        .map_err(|error| format!("Nano TLS: leitura: {error}"))?;

    let response = String::from_utf8_lossy(&raw);
    let (headers, body) = response.split_once("\r\n\r\n")
        .ok_or_else(|| "Nano TLS: resposta HTTP inválida".to_string())?;
    let status = headers.lines().next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| "Nano TLS: status HTTP inválido".to_string())?;

    if !(200..300).contains(&status) {
        return Err(format!("Nano HTTPS: servidor devolveu HTTP {status}"));
    }

    Ok(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::https_get;

    #[test]
    fn tls_entrypoint_is_native() {
        let _ = https_get as fn(&str, u16, &str) -> Result<String, String>;
    }
}
