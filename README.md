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

Selecione o backend explicitamente:

```bash
cargo run -- run --backend cpu examples/tensor.nano
cargo run -- run --backend gpu examples/tensor.nano
```

O backend `gpu` já é reconhecido, mas ainda falha explicitamente porque o kernel GPU real ainda não foi implementado.

A mesma seleção pode ser feita com:

```bash
NANO_BACKEND=cpu cargo run -- run examples/tensor.nano
```

A opção `--backend` tem prioridade sobre `NANO_BACKEND`.

Ou, dentro de um projeto:

```bash
nano run
```

A CLI bootstrap agora inclui:

```bash
nano run
nano check examples/tensor.nano
nano run --backend cpu examples/tensor.nano
```

A seleção de `gpu` já existe na CLI, mas o backend GPU real ainda não está implementado.

Quando `run` for executado dentro de um projeto sem arquivo explícito, a convenção será procurar automaticamente por `main.nano`.

## Nano 0.8

O núcleo atual contém:

- números, texto e booleanos;
- variáveis;
- inferência automática de tipos;
- verificação semântica básica antes da execução;
- expressões;
- `print`;
- `if` / `else`;
- `while`;
- `for ... in ...`;
- `range()`;
- blocos;
- funções;
- arquivos `.nano`;
- listas;
- objetos/mapas;
- indexação;
- `len()`;
- módulos com `use`;
- tipo nativo `Tensor`;
- `tensor()`, `parameter()`, `zeros()`, `shape()`, `matmul()`, `sum()` e `mean()`;
- introspecção com `backend()` e `device()`;
- seleção explícita de backend no CLI;
- autograd com `grad()`;
- atualização de parâmetros com `step()`;
- otimizador Adam com `adam()`;
- módulos de modelo usando `use`;
- atualizações de parâmetros in-place;
- IR otimizado com constant folding;
- runtime próprio.

A regra é: capacidades novas não devem transformar Nano numa linguagem cheia de declarações obrigatórias.

O próximo foco é transformar o Tensor em uma infraestrutura de computação real: operações vetorizadas, tipos de precisão, otimização do grafo, memória planejada, transferência entre dispositivos e backends GPU reais.

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

A primeira implementação de IR está em `src/ir.rs`. O runtime também protege carregamento repetido de módulos e detecta ciclos de `use`.

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

