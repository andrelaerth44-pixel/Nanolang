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
5. Otimizadores.
6. Data loader e datasets grandes.
7. Execução GPU.
8. Kernel fusion e planejamento de memória.
9. Treino distribuído/offload quando necessário.
10. Ferramentas para modelos de grande porte.

O suporte atual ainda está no início: Tensor usa `f32` e `matmul` 2D em CPU. Portanto, a meta de 5B parâmetros ainda não é uma capacidade disponível do Nano atual.

## Princípio

O programador não deve precisar escrever centenas de chamadas de baixo nível para obter um pipeline de IA. A complexidade deve ficar no compilador, IR, runtime e backends.
