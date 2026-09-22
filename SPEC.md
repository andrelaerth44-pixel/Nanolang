# Nano 0.1 — Especificação inicial

## 1. Arquivo

Todo código-fonte Nano usa a extensão .nano.

Exemplos:
- MainActivity.nano
- Server.nano
- Game.nano
- Model.nano

## 2. Sintaxe

Variáveis:

name = "André"
age = 25
active = true

Impressão:

print "Olá"
print age

Expressões:

total = 10 + 5 * 2
message = "Olá " + name

Condições:

if age >= 18 {
    print "adulto"
} else {
    print "menor"
}

Funções:

function add(a, b) {
    return a + b
}

print add(2, 3)

## 3. Tipos

- Number
- Text
- Boolean
- Null

A linguagem favorece inferência automática.

## 4. Execução

Nano 0.1 possui um runtime próprio para validar a semântica da linguagem.

O runtime executa a AST diretamente. O programa Nano não é convertido para Python, Kotlin, Java, JavaScript ou outra linguagem.

Evolução planejada:

Nano source
  -> Lexer
  -> Parser
  -> AST
  -> Nano IR
  -> Runtime Nano
  -> Native CPU backend
  -> GPU backend
  -> NPU backend

## 5. Simplicidade

A sintaxe deve continuar pequena mesmo quando a plataforma ganhar capacidades avançadas.

O poder deve vir principalmente da biblioteca padrão, runtime, IR e compilador, não de centenas de palavras-chave.
