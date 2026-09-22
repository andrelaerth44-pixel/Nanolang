# Nano Self-Hosting

Stage 0: Rust bootstrap compiler.
Stage 1: lexer, parser, validador semântico e compilador textual de IR já funcionam em Nano, cobrindo funções, expressões, objetos, indexação, loops e controle básico.
Stage 2: Rust e Nano passam a expor uma representação IR textual determinística para comparação e o bootstrap valida a AST antes da emissão.
Stage 3: bootstrap passa a produzir artefatos de compilador nativo.
Stage 4: Nano compila o próprio compilador.
Stage 5: formatter, LSP, package manager, linter e debugger tornam-se ferramentas Nano-native.

A regra de autoridade continua: o compilador self-hosted só substitui o bootstrap Rust quando a suíte de conformidade comprovar semântica equivalente.
