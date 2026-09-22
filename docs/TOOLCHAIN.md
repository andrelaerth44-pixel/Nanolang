# Nano Toolchain

Commands:
- nano run file.nano
- nano check file.nano
- nano-fmt file.nano
- nano-lsp
- nano build
- nano test
- nano package

## IDE intelligence

Syntax highlighting colors tokens.

LSP supplies completion/autocomplete, hover, diagnostics, formatting and code actions.

The familiar editor lightbulb is normally a Code Action / Quick Fix indicator. Nano exposes textDocument/codeAction so compatible editors can show that interface.

Formatter supplies consistent layout.

These layers together provide the modern IDE experience expected from Kotlin-like development environments.
