# Nano Standard Library

## Core
len, range, print, string/list/object operations, numeric operations and conversions.

## Tensor / AI
tensor, parameter, zeros, shape, matmul, sum, mean, grad, step, adam, cast, dtype, device, memory_bytes.

## Platform modules
std.fs: files, directories and paths.
std.net: TCP and HTTP primitives.
std.time: clocks, timers and durations.
std.async: tasks, channels and cancellation.
std.ui: windows, widgets, events and rendering.
std.gfx: portable drawing primitives.
std.process: subprocesses and environment.
std.math: deterministic math utilities.

Each module must have CPU reference behavior and backend-specific implementations where required.
