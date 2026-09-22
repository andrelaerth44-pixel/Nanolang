# Nano Native Platform

Nano is a native language. It must not depend on transpiling Nano into Kotlin, Java or Python.

## CPU native compiler
Nano IR is lowered through a dedicated native code-generation layer to the target ABI and machine code. The Rust bootstrap remains the semantic reference until self-hosting is proven.

## GPU
The current runtime provides a real wgpu backend and resident tensor execution.

## NPU
NPU support is a real backend contract: device discovery, buffer allocation, kernel submission and synchronization. Unsupported targets report a capability error instead of silently pretending to be an NPU.

## ABI
Native modules use an explicit Nano ABI for values, tensors, dtypes, ownership and errors.

## Conformance
Every backend is checked against the CPU semantic reference.
