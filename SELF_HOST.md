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