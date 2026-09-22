# Nano language completion contract

Nano is being finished in layers. A language is not considered complete merely because the parser accepts more keywords; the compiler, runtime, tooling and standard library must agree on the same language contract.

## Completed foundation

- `.nano` source files and `main.nano` entry point
- lexer, parser and semantic checking
- inferred types, variables, expressions and blocks
- functions, returns, conditionals and loops
- lists, objects, indexing and field access
- modules with `use`
- Nano IR and optimizer
- CPU runtime
- Tensor, autograd, SGD and Adam
- CPU/GPU backend abstraction
- real wgpu compute backend
- resident GPU tensor execution
- resident GPU backward/gradient operations
- resident GPU optimizer updates
- explicit GPU readback
- compact f32/f16/bf16 storage
- CLI backend/dtype selection
- CI with cargo check and cargo test
- editor-independent TextMate syntax highlighting

## Remaining engineering contract

1. Toolchain: formatter, linter, source locations/diagnostics, package metadata, reproducible builds and LSP.
2. Language surface: richer assignment targets, unary/logical operators, richer strings, first-class functions/closures, explicit error/result handling and a stable standard library.
3. Runtime: filesystem, processes, networking, concurrency/tasks, timers/events and application/UI primitives.
4. Native execution: native CPU code generation, broader GPU kernels, NPU backend, graph-level memory planning, command batching, true low-precision GPU arithmetic and multi-device execution.
5. Self-host: Nano lexer/parser/AST/IR, double bootstrap and compiler self-compilation.
6. Application framework: UI, input/events, rendering, assets and packaging/deployment.

This checklist is finite and testable. It is the contract that separates the current bootstrap compiler from a mature application platform.
