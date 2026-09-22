# Nano 0.4 — Especificação inicial

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

Os tipos são inferidos automaticamente. Não é necessário escrever o tipo da variável.

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

## 5. Módulos

Um arquivo Nano pode importar outro arquivo com uma única palavra:

```nano
use "math.nano"
```

O módulo é executado no mesmo runtime Nano, permitindo definir funções e dados reutilizáveis.

Exemplo:

```nano
# math.nano
function double(x) {
    return x * 2
}
```

```nano
# main.nano
use "math.nano"

print double(21)
```

A forma inicial é deliberadamente simples. Sistema de módulos com nomes, pacotes e namespaces será adicionado depois sem quebrar esta sintaxe básica.

## 6. Tipos e inferência

Nano 0.4 mantém os tipos fora da sintaxe do dia a dia.

Tipos-base atuais:

- Number
- Text
- Boolean
- List
- Object
- Null
- Any interno

Exemplo:

```nano
name = "André"
age = 25
active = true
numbers = [10, 20, 30]
```

O compilador/runtime infere automaticamente:

```
name    -> Text
age     -> Number
active  -> Boolean
numbers -> List
```

Não é necessário escrever:

```
Text name
Number age
Boolean active
```

### Verificação antecipada

Antes do runtime executar o programa, o compilador faz uma verificação semântica simples.

Por exemplo, isto é rejeitado:

```nano
age = 25
age = "vinte e cinco"
```

Erro esperado:

```
Nano: tipo incompatível em variável 'age': Number e Text
```

Operações também são verificadas quando o tipo já é conhecido:

```nano
total = 10 + 5
message = "idade: " + total
```

Funções e módulos que ainda não permitem descobrir um tipo com segurança usam `Any` internamente. Isso mantém a sintaxe pequena e evita exigir anotações de tipo.

## 7. Execução

Nano 0.4 possui um runtime próprio e uma camada de verificação semântica.

O programa Nano não precisa ser convertido para Python, Kotlin, Java, JavaScript ou outra linguagem.

Arquitetura:

```
Nano source
    ↓
Lexer
    ↓
Parser
    ↓
Semantic Types / AST
    ↓
Nano IR
    ├── Runtime Nano
    ├── Native CPU
    ├── Native GPU
    └── Native NPU
```

A camada de tipos é uma etapa do compilador; ela não adiciona sintaxe obrigatória ao programador.

## 8. Self-host

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

## 9. Regra de simplicidade

A sintaxe deve continuar pequena mesmo quando a plataforma ganhar capacidades avançadas.

O poder deve vir principalmente da composição da linguagem, biblioteca padrão, runtime, IR, compilador e ferramentas, não de centenas de palavras-chave.
