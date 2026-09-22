# Nano 0.7 — Especificação inicial

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

### Loops

Loop condicional:

```nano
while age < 30 {
    age = age + 1
}
```

Loop de coleção:

```nano
for item in numbers {
    print item
}
```

Contagem simples:

```nano
for epoch in range(10) {
    print epoch
}
```

Nano mantém apenas duas formas principais de repetição: `while` e `for ... in ...`.

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
- Tensor
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

## 7. Dados e Tensor

Tensor é um tipo nativo para dados numéricos multidimensionais.

Exemplo:

```nano
a = tensor([1, 2, 3, 4], [2, 2])
b = tensor([5, 6, 7, 8], [2, 2])

c = matmul(a, b)

print shape(c)
```

Primitivas iniciais:

- `tensor(dados, shape)`
- `zeros(shape)`
- `shape(tensor)`
- `matmul(a, b)`
- `sum(tensor)`
- `mean(tensor)`
- `parameter(dados, shape)`
- `grad(loss, parâmetro)`
- `step(parâmetro, gradiente, taxa)`
- `len(tensor)`

Nano 0.6 adiciona um grafo de operações de Tensor e diferenciação automática para operações iniciais.

Exemplo de um passo de treino:

```nano
x = tensor([1, 2, 3, 4], [2, 2])
w = parameter([1, 0, 0, 1], [2, 2])

y = matmul(x, w)
loss = mean(y)

g = grad(loss, w)
w = step(w, g, 0.01)
```

A implementação inicial usa dados `f32`, operações 2D em CPU e um grafo de autograd em memória. Isso já permite prototipar cálculo de gradientes, mas ainda não representa treinamento GPU de modelos de bilhões de parâmetros.


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

## 9. Self-host

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

## 10. Regra de simplicidade

A sintaxe deve continuar pequena mesmo quando a plataforma ganhar capacidades avançadas.

O poder deve vir principalmente da composição da linguagem, biblioteca padrão, runtime, IR, compilador e ferramentas, não de centenas de palavras-chave.


## 8. Nano IR

Nano 0.4 já possui uma primeira implementação de IR.

Pipeline:

```
Nano source
    ↓
Lexer
    ↓
Parser
    ↓
Semantic Types
    ↓
AST
    ↓
Nano IR
    ↓
Nano IR Runtime
    ├── Native CPU
    ├── Native GPU
    └── Native NPU
```

O compilador em `src/ir.rs` transforma a AST em instruções intermediárias como `Const`, `Load`, `Store`, `Binary`, `Call`, `Jump`, `JumpIfFalse`, `Return`, `Index`, `Field` e `Print`.

A VM Nano executa esse IR diretamente. A sintaxe da linguagem continua independente do backend.



## 9. Módulos de modelo

Um modelo pode ser separado em um módulo Nano comum.

```nano
# model.nano
function forward(x, w) {
    return matmul(x, w)
}
```

E usado pelo programa:

```nano
# train.nano
use "model.nano"

y = forward(x, w)
```

Não existe uma palavra-chave obrigatória como `model`. O módulo de modelo usa as mesmas funções, Tensor e módulos da linguagem.

## 10. Otimizadores

Nano 0.7 possui dois caminhos de atualização:

```nano
w = step(w, grad(loss, w), 0.001)
w = adam(w, grad(loss, w), 0.001)
```

`step()` executa SGD simples.

`adam()` mantém o estado do otimizador por parâmetro e usa atualização Adam com momentos e correção de viés.

As atualizações atuais são feitas **in-place** no armazenamento do Tensor para evitar criar um novo buffer do parâmetro a cada passo.

## 11. Memória

Tensor usa armazenamento contíguo `f32` e referências compartilhadas para o grafo de autograd.

Operações derivadas mantêm referências aos tensores de entrada em vez de copiar seus buffers inteiros.

`matmul()` também evita copiar os buffers de entrada para realizar o cálculo.

Isso é uma otimização inicial de memória. Para modelos muito grandes ainda serão necessárias alocação por arena, planejamento de memória, tipos de precisão menores, checkpointing e offload.

