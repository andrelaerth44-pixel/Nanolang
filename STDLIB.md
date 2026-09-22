# Nano Standard Library

## Core

len, range, print, to_text, to_number, upper, lower, trim, contains, starts_with, ends_with, replace, substring, char_at, split, join, append, abs, sqrt, floor, ceil, round, sin, cos, tan, exp, log, pow, min, max.

As operações de listas e objetos também podem usar indexação (lista[n], objeto["chave"]) e campos (objeto.campo).

## Tensor / AI

tensor, parameter, zeros, shape, matmul, sum, mean, grad, step, adam, cast, dtype, device, memory_bytes, backend.

## Filesystem

fs_read_text, fs_write_text, fs_append_text, fs_exists, fs_is_file, fs_is_dir, fs_cwd, fs_list, fs_mkdir, fs_remove.

path_join, path_basename, path_dirname, path_extension.

## Processos e ambiente

env_get, env_set, os_cwd, os_args, process_spawn, process_wait, process_output.

## Tempo e concorrência

time_now_ms, time_sleep_ms, thread_sleep_ms, thread_yield, thread_spawn, thread_join, task_spawn, task_join, channel, send, recv, close_channel.

thread_spawn executa um comando externo em uma thread nativa e retorna um handle que pode ser aguardado com thread_join.

## Rede

net_http_get, net_tcp_connect, net_tcp_listen, net_tcp_accept, net_tcp_send, net_tcp_recv, net_tcp_close.

A camada atual fornece TCP direto. HTTP de alto nível pode ser construído sobre essa API.

## HTTP

http_request, http_post e http_get_structured fornecem respostas estruturadas com status, reason, headers e body. https_get usa curl quando disponível.

## Crypto

std.crypto.sha256 e std.crypto.hmac_sha256 fornecem hashing e autenticação HMAC-SHA256.

## Sync

std.sync fornece mutexes compartilhados, semáforos e canais globais seguros entre threads, incluindo recv_timeout e try_recv.

## JSON

std.json.encode e std.json.decode fazem conversão entre Text JSON e os tipos Nano compatíveis: Number, Boolean, Text, Null, List e Object.

## UI

ui_window, ui_set_title, ui_clear, ui_rect, ui_button, ui_close, ui_poll_event, ui_wait_event.

A camada 2D usa wgpu para renderizar retângulos e botões e faz hit-testing de clique no runtime da janela.

A implementação atual cria uma janela desktop nativa e entrega eventos de criação, resize, teclado, mouse, modificadores e fechamento. `ui_poll_event` é não bloqueante; `ui_wait_event` aguarda o próximo evento.

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
- std.path
- std.json
- std.http
- std.crypto
- std.sync
- std.os

Os nomes dos módulos formam o contrato de organização; as funções acima são a superfície runtime atualmente implementada.

## Self-host nativo

nano build --selfhost arquivo.nano executa o lexer/parser/validador/compiler escritos em Nano para produzir IR textual, reimporta esse IR e envia o programa ao backend nativo x86_64 Linux.

## Backend e hardware

O backend Tensor aceita cpu, gpu e npu.

- cpu: referência sem hardware externo.
- gpu: backend residente real baseado em wgpu.
- npu: provider externo via NANO_NPU_PROVIDER e ABI v1 documentada em NPU.md.

Nenhum backend é mascarado como outro: quando um provider ou capability não existe, o runtime retorna erro explícito.
