# NanoLang

Nano é uma linguagem de programação independente, nativa e simples de aprender.

A extensão oficial dos programas é .nano.

## Filosofia

Pouco código. Poucos conceitos. Enorme capacidade.

Nano não é uma linguagem ponte para Python, Kotlin, Java, JavaScript ou outra linguagem.

Arquitetura pretendida:

.nano -> Nano Frontend -> Nano IR -> Nano Runtime / Native Backends

A versão 0.1 usa um runtime próprio para executar a linguagem diretamente. A evolução prevista é adicionar geração de código nativo para CPU e, depois, GPU/NPU.

## Primeiro programa

app MainActivity {
    print "Olá, mundo!"
}

Salve como MainActivity.nano.

## CLI

cargo run -- run examples/MainActivity.nano

Na evolução do projeto:

nano run MainActivity.nano
nano build MainActivity.nano
nano build MainActivity.nano --native

## Nano 0.1

O núcleo inicial contém:

- valores numéricos, texto e booleanos;
- variáveis;
- expressões;
- print;
- if / else;
- blocos;
- funções simples;
- arquivos .nano;
- runtime próprio.

Depois entram dados, aplicações, paralelismo, IA, compressão e backends nativos.

## Objetivo

Fazer com que uma pessoa consiga aprender Nano rapidamente e ainda assim construir aplicações grandes, sistemas, jogos, ferramentas de dados e IA usando a mesma linguagem.
