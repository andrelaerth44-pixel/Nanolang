# Nano para IA — direção de engenharia

Nano deve continuar simples para o programador mesmo quando a infraestrutura de IA ficar grande.

## Objetivo

Permitir que operações comuns de dados e modelos sejam expressas em poucas linhas:

```nano
x = tensor(data, [batch, features])
w = tensor(weights, [features, hidden])
y = matmul(x, w)
```

A meta de longo prazo é chegar a treinamento de modelos da classe de bilhões de parâmetros com o mínimo de código Nano possível.

## Arquitetura planejada

```
Nano
  ↓
Tensor IR
  ↓
Graph / Fusion
  ↓
Memory Planner
  ↓
CPU backend / GPU backend
  ↓
CUDA-capable GPU e outros aceleradores
```

## Etapas

1. Tensor nativo em CPU.
2. Grafo de operações e autograd inicial.
3. Operações vetorizadas e kernels.
4. Tipos de precisão e armazenamento compacto.
5. Otimizadores: SGD e Adam na linguagem.
6. Data loader e datasets grandes.
7. Execução GPU.
8. Kernel fusion e planejamento de memória.
9. Treino distribuído/offload quando necessário.
10. Ferramentas para modelos de grande porte.

O suporte atual ainda está no início: Tensor usa `f32`, `matmul` 2D em CPU, autograd inicial e Adam. Ainda não há backend GPU. Portanto, a meta de 5B parâmetros ainda não é uma capacidade disponível do Nano atual.

### Memória para modelos grandes

A primeira regra é não duplicar buffers desnecessariamente. Nano já usa referências compartilhadas no grafo de Tensor e atualizações de parâmetros in-place.

As próximas camadas de memória serão:

```
contiguous buffers
      ↓
memory planner
      ↓
reuse de buffers
      ↓
mixed precision
      ↓
checkpoint / offload
      ↓
GPU memory manager
```

## Princípio

O programador não deve precisar escrever centenas de chamadas de baixo nível para obter um pipeline de IA. A complexidade deve ficar no compilador, IR, runtime e backends.
