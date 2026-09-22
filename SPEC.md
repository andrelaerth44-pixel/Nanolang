# Nano 0.2 — Especificação inicial

## 1. Arquivos

Todo código-fonte Nano usa a extensão `.nano`.

O ponto de entrada convencional de um projeto é:

```
main.nano
```

`main.nano` significa **código principal**, não menu, tela ou Activity.

Não existe obrigação de usar nomes como `MainActivity`.

Exemplo:

```
MeuProjeto/
├── main.nano
├── ui.nano
├── data.nano
└── model.nano
```

O compilador/runtime deverá tratar `main.nano` como entrada quando o projeto for executado sem indicar um arquivo específico.

## 2. Sintaxe

A Nano deve evitar palavras obrigatórias que apenas descrevem o nome do arquivo.

Portanto, isto é válido:

```nano
print "Olá"

name = "André"

if name {
    print "Olá " + name
}
```

Não é necessário escrever:

```nano
app MainActivity {
    ...
}
```

### Variáveis

```nano
name = "André"
age = 25
active = true
```

### Impressão

```nano
print "Olá"
print age
```

### Expressões

```nano
total = 10 + 5 * 2
message = "Olá " + name
```

### Condições

```nano
if age >= 18 {
    print "adulto"
} else {
    print "menor"
}
```

### Funções

```nano
function add(a, b) {
    return a + b
}

print add(2, 3)
```

## 3. Coleções e objetos

Listas:

```nano
numbers = [10, 20, 30]
print numbers[0]
print len(numbers)
```

Objetos:

```nano
user = {
    name: "André",
    age: 25
}

print user.name
print user["age"]
```

Listas podem ser combinadas:

```nano
all = [1, 2] + [3, 4]
```

O mesmo valor de objeto serve como estrutura de dados simples. Não existe uma palavra-chave obrigatória como `class` para criar um objeto.

## 4. Aplicações e interface

A interface não define o arquivo principal.

Uma aplicação grande poderá separar responsabilidades:

```
main.nano     -> inicialização e entrada
ui.nano       -> interface
data.nano     -> dados
network.nano  -> rede
model.nano    -> lógica/IA
```

A sintaxe específica de interface será definida quando o núcleo da linguagem estiver estável.

## 5. Tipos

Nano 0.1 começa com:

- Number
- Text
- Boolean
- Null

A linguagem favorece inferência automática.

## 6. Execução

Nano 0.1 possui um runtime próprio.

O programa Nano não precisa ser convertido para Python, Kotlin, Java, JavaScript ou outra linguagem.

Arquitetura:

```
Nano source
    ↓
Lexer
    ↓
Parser
    ↓
AST
    ↓
Nano IR
    ├── Runtime Nano
    ├── Native CPU
    ├── Native GPU
    └── Native NPU
```

## 7. Self-host

O bootstrap inicial pode ser escrito em Rust apenas para dar nascimento à ferramentachain.

O objetivo é migrar o compilador para Nano:

```
Rust bootstrap
      ↓
Nano compiler 1
      ↓
Nano compiler 2
      ↓
Nano compila Nano
```

Quando o compilador Nano conseguir compilar o seu próprio código-fonte usando uma versão anterior funcional do compilador, teremos atingido o núcleo do self-host.

## 8. Regra de simplicidade

A sintaxe deve continuar pequena mesmo quando a plataforma ganhar capacidades avançadas.

O poder deve vir principalmente da composição da linguagem, biblioteca padrão, runtime, IR, compilador e ferramentas, não de centenas de palavras-chave.
