use std::{io::{self,Read,Write}};

fn main(){
    let mut input=Vec::new();
    io::stdin().read_to_end(&mut input).unwrap();
    let mut out=io::stdout();
    let mut pos=0usize;
    while pos<input.len(){
        let Some(offset)=input[pos..].windows(4).position(|w|w==b"\r\n\r\n") else {break};
        let header_end=pos+offset;
        let header=String::from_utf8_lossy(&input[pos..header_end]);
        let len=header.lines().find_map(|l|l.strip_prefix("Content-Length:")).and_then(|v|v.trim().parse::<usize>().ok()).unwrap_or(0);
        let body_start=header_end+4;
        if body_start+len>input.len(){break}
        let body=String::from_utf8_lossy(&input[body_start..body_start+len]);
        if let Some(id)=json_id(&body){
            let method=json_string(&body,"method").unwrap_or_default();
            let result=match method.as_str(){
                "initialize"=>r#"{"capabilities":{"textDocumentSync":1,"completionProvider":{"triggerCharacters":["."]},"hoverProvider":true,"documentFormattingProvider":true,"codeActionProvider":true}}"#,
                "shutdown"=>r#"null"#,
                "textDocument/completion"=>r#"{"isIncomplete":false,"items":[{"label":"function","kind":14},{"label":"if","kind":14},{"label":"else","kind":14},{"label":"while","kind":14},{"label":"for","kind":14},{"label":"return","kind":14},{"label":"print","kind":3},{"label":"tensor","kind":3},{"label":"parameter","kind":3},{"label":"matmul","kind":3},{"label":"grad","kind":3},{"label":"adam","kind":3},{"label":"step","kind":3},{"label":"zeros","kind":3},{"label":"shape","kind":3},{"label":"range","kind":3}]}"#,
                "textDocument/hover"=>r#"{"contents":{"kind":"markdown","value":"Nano language symbol information"}}"#,
                "textDocument/formatting"=>r#"[]"#,
                "textDocument/codeAction"=>r#"[]"#,
                _=>r#"null"#,
            };
            let response=format!(r#"{{"jsonrpc":"2.0","id":{},"result":{}}}"#,id,result);
            write_lsp(&mut out,&response).unwrap();
        }
        pos=body_start+len;
    }
}
fn json_id(s:&str)->Option<String>{let p=s.find("\"id\"")?;let r=&s[p+4..];let c=r.find(':')?;let v=r[c+1..].trim_start();let end=v.find(|c:char|c==','||c=='}')?;Some(v[..end].trim().to_string())}
fn json_string(s:&str,key:&str)->Option<String>{let needle=format!("\"{}\"",key);let p=s.find(&needle)?;let r=&s[p+needle.len()..];let c=r.find(':')?;let v=r[c+1..].trim_start();if !v.starts_with('"'){return None}let end=v[1..].find('"')?+1;Some(v[1..end].to_string())}
fn write_lsp(out:&mut impl Write,body:&str)->io::Result<()>{write!(out,"Content-Length: {}\r\n\r\n{}",body.len(),body);out.flush()}
