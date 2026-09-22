# Nano syntax highlighting

This directory contains the editor-independent Nano syntax definition.

- Grammar: `nano.tmLanguage.json`
- Language configuration: `nano-language-configuration.json`
- File extension: `.nano`
- TextMate scope: `source.nano`

The grammar highlights comments, strings, escape sequences, numbers, control-flow keywords, function definitions and calls, booleans, `null`, Nano built-ins, operators, properties and identifiers.

The grammar assigns semantic scopes; the editor/theme chooses the actual colors. This is the same separation used by mature language tooling: Nano defines what a token is, while the editor defines how it looks.

Core keywords:
`use`, `function`, `print`, `if`, `else`, `return`, `while`, `for`, `in`, `true`, `false`.

Tensor/runtime built-ins:
`len`, `range`, `tensor`, `parameter`, `zeros`, `shape`, `matmul`, `sum`, `mean`, `grad`, `step`, `adam`, `backend`, `device`, `dtype`, `memory_bytes`, `cast`.
