# Nano IR 0.1

O Nano IR é a representação intermediária entre a AST e a execução.

## Instruções iniciais

`Const`, `Load`, `Store`, `Binary`, `MakeList`, `MakeObject`, `Index`, `Field`, `Call`, `Print`, `Pop`, `Jump`, `JumpIfFalse`, `Return` e `Use`.

## Objetivo

A primeira implementação usa uma VM simples para validar a arquitetura:

```
Nano IR
 ├── VM / desenvolvimento
 ├── CPU nativa
 ├── GPU
 └── NPU
```

Nano continua sendo uma linguagem independente. O IR existe justamente para separar a sintaxe do backend.

