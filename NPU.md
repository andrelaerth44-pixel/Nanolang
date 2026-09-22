# Nano NPU Provider

O backend `npu` do Nano usa um provider externo para falar com hardware NPU sem transformar Nano em outra linguagem.

## Configuração

Defina:

    NANO_NPU_PROVIDER=/caminho/para/nano-npu-provider

E execute:

    nano run --backend npu main.nano

O executável do provider recebe pedidos JSON por `stdin`, uma linha por pedido, e devolve uma resposta JSON por `stdout`.

## ABI v1

Handshake obrigatório:

    {"id":0,"op":"handshake","abi":1}

Resposta:

    {"id":0,"ok":true,"abi":1}

Operações do provider:

- `upload`: recebe `tensor_id` e `data`.
- `read`: recebe `tensor_id` e devolve `data`.
- `release`: libera um tensor residente.
- `matmul`: `left`, `left_shape`, `right`, `right_shape`.
- `elementwise`: `left`, `right`, `shape`, `operator` (`add`, `sub`, `mul`, `div`).
- `fused_mul_add`: `left`, `right`, `bias`, `shape`.
- `reduce`: `data` e `mean`, devolvendo `value`.
- `transfer`: recebe `data` e devolve `data`.

Sucesso:

    {"id":7,"ok":true,"data":[1,2,3]}

Erro:

    {"id":7,"ok":false,"error":"mensagem"}

## Regra de execução

O Nano não finge que NPU é GPU ou CPU. Sem `NANO_NPU_PROVIDER`, o backend retorna erro explícito. Um provider pode usar a SDK do fabricante por baixo, mantendo o contrato da linguagem independente do fabricante.

A camada atual é síncrona e baseada em buffers host; operações residentes assíncronas específicas de GPU continuam exclusivas do backend GPU.
