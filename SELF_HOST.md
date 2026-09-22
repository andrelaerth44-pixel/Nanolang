# Nano Self-Host Roadmap

O objetivo da Nano é chegar a um ponto em que o compilador da própria linguagem seja escrito e compilado pela Nano.

## Bootstrap

A primeira geração do compilador/runtime pode ser escrita em Rust.

Isso é bootstrap, não uma dependência sem fim.

Rust -> Nano compiler -> Nano program

## Stage 1 — linguagem suficiente

A Nano precisa ganhar:

- módulos;
- listas e mapas;
- estruturas/objetos;
- tipos e inferência;
- funções de primeira classe;
- arquivos e diretórios;
- processos/tarefas;
- testes;
- erros;
- biblioteca padrão.

## Stage 2 — Nano IR

O compilador precisa separar:

Source -> AST -> Nano IR -> Backend

O IR deve ser uma representação própria da Nano.

## Stage 3 — compiler em Nano

Partes do compilador são reescritas em Nano:

lexer.nano
parser.nano
ast.nano
ir.nano
compiler.nano

O compilador antigo, escrito em Rust, continua servindo como bootstrap.

## Stage 4 — self compilation

Uma versão estável do compilador Rust compila o compilador escrito em Nano.

Nano compiler A
      |
      +-- compila --> Nano compiler B

A saída precisa produzir uma nova versão funcional do compilador Nano.

## Stage 5 — independência

A partir daqui, a evolução principal do compilador pode acontecer em Nano.

Rust permanece como ferramenta de bootstrap histórica, não como parte obrigatória da linguagem.

## Regra

Nunca definir a linguagem de forma que self-host se torne impossível.

Cada recurso novo deve ser pensado também em termos de:

1. como Nano consegue expressá-lo;
2. como o compilador Nano consegue implementá-lo;
3. como o próprio compilador poderá migrar para Nano.

## Estado executável do bootstrap

A etapa de self-host não será considerada concluída por uma reescrita textual. Cada estágio precisa produzir uma compilação reproduzível.

### Stage 0 — contrato atual
`nano check` deve analisar os exemplos sem executar o programa. O CI executa `cargo check` e `cargo test` para manter o bootstrap estável.

### Stage 1 — front-end em Nano
Criar os módulos `compiler/lexer.nano`, `compiler/parser.nano`, `compiler/ast.nano` e `compiler/ir.nano`. Primeiro eles precisam tokenizar, analisar e emitir uma IR textual de um subconjunto mínimo.

### Stage 2 — bootstrap duplo
O compilador Rust e o compilador Nano devem aceitar o mesmo subconjunto e gerar uma IR normalizada equivalente.

### Stage 3 — compilação do compilador
O bootstrap Rust deve compilar o compilador Nano e gerar um executável funcional. Esse executável passa a ser usado para compilar novas versões do compilador Nano.

### Stage 4 — independência
Quando o compilador Nano conseguir compilar seu próprio código-fonte sem intervenção do Rust, Rust deixa de ser a implementação principal e fica apenas como bootstrap histórico.

### Primeiro alvo de subconjunto
`use`, funções, listas, objetos, condicionais, loops, expressões, arquivos, erros e a emissão de IR precisam funcionar antes de migrar Tensor/GPU para o compilador escrito em Nano.
