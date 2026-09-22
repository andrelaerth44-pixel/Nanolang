# Nano Self-Hosting

Stage 0: Rust bootstrap compiler.
Stage 1: compiler/lexer.nano, parser.nano, ast.nano and ir.nano.
Stage 2: Rust and Nano compilers emit normalized equivalent IR.
Stage 3: bootstrap emits native compiler artifacts.
Stage 4: Nano compiles its own compiler.
Stage 5: formatter, LSP, package manager, linter and debugger become Nano-native tools.

The self-hosted compiler becomes authoritative only after the conformance suite proves equivalent semantics.
