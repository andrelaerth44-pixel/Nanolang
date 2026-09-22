# Nano Standard Library

## Core

len, range, print, to_text, to_number, upper, lower, trim, contains, starts_with, ends_with, replace, substring, char_at, split, join, append.

As operações de listas e objetos também podem usar indexação (lista[n], objeto["chave"]) e campos (objeto.campo).

## Tensor / AI

tensor, parameter, zeros, shape, matmul, sum, mean, grad, step, adam, cast, dtype, device, memory_bytes, backend.

## Filesystem

fs_read_text, fs_write_text, fs_append_text, fs_exists, fs_list, fs_mkdir, fs_remove.

## Processos e ambiente

env_get, env_set, process_spawn, process_wait.

## Tempo e concorrência

time_now_ms, time_sleep_ms, thread_sleep_ms, thread_spawn, thread_join.

thread_spawn executa um comando externo em uma thread nativa e retorna um handle que pode ser aguardado com thread_join.

## Rede

net_tcp_connect, net_tcp_listen, net_tcp_accept, net_tcp_send, net_tcp_recv, net_tcp_close.

A camada atual fornece TCP direto. HTTP de alto nível pode ser construído sobre essa API.

## UI

ui_window, ui_set_title, ui_close, ui_poll_event.

A implementação atual cria uma janela desktop nativa e entrega eventos básicos (created, resized, close_requested, closed).

## Módulos

Estes imports são reconhecidos pelo runtime:

- std.fs
- std.net
- std.time
- std.async
- std.ui
- std.gfx
- std.process
- std.math

Os nomes dos módulos formam o contrato de organização; as funções acima são a superfície runtime atualmente implementada.

## Backend e hardware

O backend Tensor aceita cpu, gpu e npu.

- cpu: referência sem hardware externo.
- gpu: backend residente real baseado em wgpu.
- npu: provider externo via NANO_NPU_PROVIDER e ABI v1 documentada em NPU.md.

Nenhum backend é mascarado como outro: quando um provider ou capability não existe, o runtime retorna erro explícito.
