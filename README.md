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

Ou, dentro de um projeto:

```bash
nano run
```

A direção da CLI também inclui:

```bash
nano build
nano build --native
```

Quando `run` for executado dentro de um projeto sem arquivo explícito, a convenção será procurar automaticamente por `main.nano`.

## Nano 0.5

O núcleo atual contém:

- números, texto e booleanos;
- variáveis;
- inferência automática de tipos;
- verificação semântica básica antes da execução;
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
- tipo nativo `Tensor`;
- `tensor()`, `zeros()`, `shape()` e `matmul()`;
- IR otimizado com constant folding;
- runtime próprio.

A regra é: capacidades novas não devem transformar Nano numa linguagem cheia de declarações obrigatórias.

O próximo foco é transformar o Tensor em uma infraestrutura de computação real: memória contígua, operações vetorizadas, tipos de precisão, autograd, otimização de grafos e backends GPU.

A meta de engenharia é permitir código Nano muito curto para dados e IA. Suporte a treinamento de modelos muito grandes, incluindo uma classe de 5 bilhões de parâmetros, será tratado como uma meta de backend e memória — não como uma promessa de que a VM atual já consegue fazer isso em qualquer GPU.

Depois entram interface, eventos, aplicações, paralelismo, IA avançada, compressão e execução nativa.

## Exemplo de tipos inferidos

O programador escreve:

```nano
name = "André"
age = 25
active = true
numbers = [10, 20, 30]

print name
print age
print active
print len(numbers)
```

Nano infere internamente `Text`, `Number`, `Boolean` e `List`. Nenhum tipo precisa ser escrito manualmente.

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


## Nano IR

A primeira implementação de IR está em `src/ir.rs`.

O fluxo agora é:

```
.nano
 ↓
Lexer
 ↓
Parser
 ↓
Semantic
 ↓
AST
 ↓
Nano IR
 ↓
IrRuntime
```

Isso cria uma fronteira real entre a linguagem e o backend. A VM atual serve como etapa inicial; a mesma representação poderá futuramente alimentar backends nativos para CPU, GPU e NPU.

