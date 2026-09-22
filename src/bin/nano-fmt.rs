use std::{env,fs,process};

fn format_source(src:&str)->String {
    let mut out=String::new();
    let mut indent=0usize;
    for raw in src.lines() {
        let line=raw.trim();
        if line.is_empty(){ if !out.ends_with("\n\n"){out.push('\n');} continue; }
        if line.starts_with('}') && indent>0 { indent-=1; }
        for _ in 0..indent { out.push_str("    "); }
        out.push_str(line);
        out.push('\n');
        let opens=line.chars().filter(|&c|c=='{').count();
        let closes=line.chars().filter(|&c|c=='}').count();
        indent=indent.saturating_add(opens).saturating_sub(closes);
    }
    out.trim_end().to_string()+"\n"
}
fn main(){
    let args:Vec<String>=env::args().collect();
    if args.len()!=2 { eprintln!("uso: nano-fmt <arquivo.nano>"); process::exit(2); }
    let path=&args[1];
    let src=fs::read_to_string(path).unwrap_or_else(|e|{eprintln!("Nano fmt: {e}");process::exit(1)});
    if let Err(e)=fs::write(path,format_source(&src)){eprintln!("Nano fmt: {e}");process::exit(1)}
}
#[cfg(test)]
mod tests{
 use super::format_source;
 #[test] fn formats_blocks(){assert_eq!(format_source("if true {\nprint 1\n}\n"),"if true {\n    print 1\n}\n");}
}
