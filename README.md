# NanoLang

Nano é uma linguagem de programação independente, nativa e simples de aprender.

A extensão oficial dos programas é `.nano`.

## Filosofia

**Pouco código. Poucos conceitos. Enorme capacidade.**

Nano não é uma linguagem ponte para Python, Kotlin, Java, JavaScript ou outra linguagem.

A arquitetura pretendida é:

`.nano -> Nano Frontend -> Nano IR -> Nano Runtime -> Native CPU/GPU/NPU`

## main.nano

O arquivo principal de um projeto Nano é simplesmente:

```
main.nano
```

Ele **não é um menu** e não é uma cópia de `MainActivity`. É o ponto de entrada do programa.

Um projeto pode ter:

```
MeuProjeto/
├── main.nano
├── ui.nano
├── data.nano
└── model.nano
```

O programa começa em `main.nano`. Outros arquivos existem para organizar sistemas maiores.

Primeiro programa:

```nano
print "Olá, mundo!"
```

Salve como `main.nano`.

## CLI

Bootstrap atual:

```bash
cargo run -- run examples/main.nano
```

Direção da CLI:

```bash
nano run
nano build
nano build --native
```

Quando `run` for executado dentro de um projeto, a convenção será procurar automaticamente por `main.nano`.

## Nano 0.3

O núcleo inicial contém:

- números, texto e booleanos;
- variáveis;
- expressões;
- `print`;
- `if` / `else`;
- blocos;
- funções;
- arquivos `.nano`;
- listas;
- objetos/mapas;
- indexação;
- `len()`;
- módulos com `use`;
- runtime próprio.

Depois entram módulos, coleções, eventos, aplicações, dados, paralelismo, IA, compressão e backends nativos.

## Self-host

O objetivo de longo prazo é o **self-host**.

A evolução planejada é:

```
Stage 0
Nano compiler inicial escrito em Rust
        ↓
Stage 1
Nano compiler escrito em Nano
        ↓
Stage 2
Nano compila o próprio Nano compiler
        ↓
Stage 3
Nano torna-se a principal linguagem de implementação de sua própria ferramentachain
```

O compilador inicial em Rust é apenas o bootstrap. Ele não define a linguagem como uma linguagem dependente de Rust.

## Objetivo

Fazer com que uma pessoa aprenda Nano rapidamente e ainda consiga construir aplicações grandes, sistemas, jogos, ferramentas de dados e IA usando a mesma linguagem.
