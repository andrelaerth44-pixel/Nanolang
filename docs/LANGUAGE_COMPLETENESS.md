# Nano language completion contract

Este documento acompanha a implementação real. Um recurso só é considerado concluído quando frontend, IR, runtime, tooling, testes e backend compatível concordam no mesmo contrato.

## Implementado

- .nano + main.nano como entrypoint convencional.
- Lexer, parser, semantic checker e IR.
- Variáveis, inferência de tipos, funções, retornos, condicionais, loops e break.
- Listas, objetos, indexação, atribuição por índice/campo.
- Operadores unários, comparação, %, && e || com short-circuit.
- Function values e chamadas indiretas.
- use e namespaces std.*.
- Result helpers: ok, err, is_ok, unwrap, error.
- Tensor, autograd, SGD e Adam.
- CPU e GPU wgpu, incluindo caminhos residentes.
- NPU provider ABI v1 + provider de referência.
- Filesystem, ambiente, processos, TCP, HTTP, tempo, threads, tasks e canais.
- UI desktop nativa com eventos básicos.
- Formatter, linter, package manager, test runner e CLI.
- LSP interativo com completion, hover, definition, references, rename, signature help, document symbols, formatting, diagnostics e Quick Fixes.
- Self-host bootstrap: lexer, parser, compiler e bootstrap em Nano.
- CI end-to-end para native CPU, NPU, self-host, package e integration tests.

## Hardening que continua

1. Diagnósticos LSP ainda precisam convergir para o mesmo parser/semantic engine do compilador em vez de manter uma análise lexical independente.
2. O formatter pode ganhar preservação de mais trivia e regras de estilo adicionais.
3. O native CPU precisa ampliar o subconjunto para coleções, objetos, Tensor e APIs de runtime, além do atual caminho escalar.
4. GPU ainda precisa de validação numérica end-to-end de treino, scheduler/liveness global, batching de command encoders, low precision aritmética nativa e multi-device.
5. NPU depende de providers de hardware/SDK; a ABI e o provider de referência já existem.
6. Self-host precisa sair de IR textual para bootstrap duplo e auto-compilação do próprio compilador.
7. UI ainda precisa de rendering/input/assets além da janela e eventos básicos.
8. Package manager precisa evoluir de dependências locais para resolução e distribuição reproduzíveis.
9. Debugger, source maps e diagnósticos com spans ainda são trabalho de toolchain.