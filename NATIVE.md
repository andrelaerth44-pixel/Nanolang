# Nano Native

Nano não transpila para Kotlin, Java ou Python.

## CPU

nano build --native compila:

Nao source -> lexer/parser/semantic -> Nano IR -> optimizer -> x86-64 assembly -> linker do sistema -> executável nativo.

O backend atual gera código x86-64 para Linux e cobre o subconjunto numérico necessário para aritmética, comparação, controle de fluxo simples, chamadas de funções numéricas, FMA e impressão.

O backend rejeita instruções fora do subconjunto com erro explícito. Isso evita produzir um executável que tenha semântica diferente do programa Nano.

## GPU

O backend GPU usa wgpu e possui execução residente para operações de tensor e caminhos de autograd/otimizador.

## NPU

O backend NPU usa um provider externo com ABI v1. A interface está em NPU.md.

O Nano não apresenta NPU como disponível quando não há provider configurado.

## Function values e Text

O backend x86-64 também materializa valores Function como ponteiros de função e Text como ponteiros para strings estáticas/retornos escalares. Chamadas indiretas escalares são emitidas como chamadas nativas via ponteiro.
