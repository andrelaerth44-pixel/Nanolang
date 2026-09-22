use std::borrow::Cow;
use std::sync::mpsc;

use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::backend::{BackendError, BackendKind, ElementwiseOp, TensorBackend};
use crate::dtype::DType;
use crate::memory::MemoryPlanner;

const MATMUL_SHADER: &str = r#"
struct Params { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; let row = gid.y;
    if (row >= params.m || col >= params.n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < params.k; i = i + 1u) {
        acc = acc + a[row * params.k + i] * b[i * params.n + col];
    }
    out[row * params.n + col] = acc;
}
"#;

const FMA_SHADER: &str = r#"
struct Params { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    out[i] = left[i] * right[i] + bias[i];
}
"#;

const ELEMENTWISE_SHADER: &str = r#"
struct Params { len: u32, op: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    let a = left[i]; let b = right[i];
    if (params.op == 0u) { out[i] = a + b; }
    else if (params.op == 1u) { out[i] = a - b; }
    else if (params.op == 2u) { out[i] = a * b; }
    else { out[i] = a / b; }
}
"#;

const REDUCE_SHADER: &str = r#"
struct Params { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> scratch: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = lid.x; var index = gid.x; var value = 0.0;
    while (index < params.len) { value = value + input[index]; index = index + 256u; }
    scratch[lane] = value; workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lane < stride) { scratch[lane] = scratch[lane] + scratch[lane + stride]; }
        workgroupBarrier(); stride = stride / 2u;
    }
    if (lane == 0u) { out[0] = scratch[0]; }
}
"#;

pub(crate) struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul: wgpu::ComputePipeline,
    elementwise: wgpu::ComputePipeline,
    fma: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    planner: MemoryPlanner,
}

impl GpuBackend {
    pub(crate) fn new() -> Result<Self, BackendError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|e| BackendError(format!("não foi possível encontrar uma GPU: {e}")))?;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("nano-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| BackendError(format!("não foi possível abrir a GPU: {e}")))?;

        let make_shader = |label, source| device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        let make_pipeline = |label: &'static str, module: &wgpu::ShaderModule| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: None, module, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None,
            })
        };
        let matmul_module = make_shader("nano-matmul", MATMUL_SHADER);
        let elementwise_module = make_shader("nano-elementwise", ELEMENTWISE_SHADER);
        let fma_module = make_shader("nano-fma", FMA_SHADER);
        let reduce_module = make_shader("nano-reduce", REDUCE_SHADER);

        Ok(Self {
            device,
            queue,
            matmul: make_pipeline("nano-matmul-pipeline", &matmul_module),
            elementwise: make_pipeline("nano-elementwise-pipeline", &elementwise_module),
            fma: make_pipeline("nano-fma-pipeline", &fma_module),
            reduce: make_pipeline("nano-reduce-pipeline", &reduce_module),
            planner: MemoryPlanner::new(),
        })
    }

    fn bytes_f32(data: &[f32]) -> Vec<u8> { data.iter().flat_map(|v| v.to_ne_bytes()).collect() }
    fn bytes_u32(data: &[u32]) -> Vec<u8> { data.iter().flat_map(|v| v.to_ne_bytes()).collect() }

    fn create_buffer(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nano-gpu-buffer"), contents: data, usage,
        })
    }

    fn readback(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, BackendError> {
        let slice = buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result.is_ok()); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| BackendError(format!("falha ao esperar pela GPU: {e}")))?;
        if !rx.recv().map_err(|_| BackendError("callback da GPU não respondeu".into()))? {
            return Err(BackendError("não foi possível mapear o buffer GPU".into()));
        }
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| BackendError(format!("não foi possível acessar o readback da GPU: {e:?}")))?;
        let mut out = Vec::with_capacity(count);
        for chunk in mapped.chunks_exact(4).take(count) {
            out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        drop(mapped); buffer.unmap();
        Ok(out)
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, inputs: &[&wgpu::Buffer], uniform: &wgpu::Buffer, groups: (u32,u32,u32), output_len: usize) -> Result<Vec<f32>, BackendError> {
        if output_len == 0 { return Ok(Vec::new()); }
        let output_bytes = self.planner.bytes_for(output_len, DType::F32);
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nano-gpu-output"), size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nano-gpu-readback"), size: output_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let mut entries = Vec::with_capacity(inputs.len() + 2);
        for (i, buffer) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: i as u32, resource: buffer.as_entire_binding() });
        }
        entries.push(wgpu::BindGroupEntry { binding: inputs.len() as u32, resource: output.as_entire_binding() });
        entries.push(wgpu::BindGroupEntry { binding: (inputs.len()+1) as u32, resource: uniform.as_entire_binding() });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nano-gpu-bind-group"), layout: &pipeline.get_bind_group_layout(0), entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nano-gpu-command") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("nano-gpu-compute"), timestamp_writes: None });
            pass.set_pipeline(pipeline); pass.set_bind_group(0, &bind_group, &[]); pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (output_len*4) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.readback(&readback, output_len)
    }

    fn transfer_f32(&self, data: &[f32]) -> Result<Vec<f32>, BackendError> {
        if data.is_empty() { return Ok(Vec::new()); }
        let input = self.create_buffer(&Self::bytes_f32(data), wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE);
        let bytes = self.planner.bytes_for(data.len(), DType::F32);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nano-gpu-transfer"), size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nano-gpu-transfer") });
        encoder.copy_buffer_to_buffer(&input, 0, &readback, 0, (data.len()*4) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.readback(&readback, data.len())
    }
}

impl TensorBackend for GpuBackend {
    fn kind(&self) -> BackendKind { BackendKind::Gpu }

    fn matmul(&self, left: &[f32], left_shape: &[usize], right: &[f32], right_shape: &[usize]) -> Result<Vec<f32>, BackendError> {
        if left_shape.len()!=2 || right_shape.len()!=2 { return Err(BackendError("matmul GPU requer tensores 2D".into())); }
        let (m,k)=(left_shape[0],left_shape[1]); let (k2,n)=(right_shape[0],right_shape[1]);
        if k!=k2 || left.len()!=m*k || right.len()!=k2*n { return Err(BackendError("matmul GPU recebeu shapes incompatíveis".into())); }
        let a=self.create_buffer(&Self::bytes_f32(left),wgpu::BufferUsages::STORAGE);
        let b=self.create_buffer(&Self::bytes_f32(right),wgpu::BufferUsages::STORAGE);
        let params=self.create_buffer(&Self::bytes_u32(&[m as u32,k as u32,n as u32,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch(&self.matmul,&[&a,&b],&params,(((n as u32)+7)/8,((m as u32)+7)/8,1),m*n)
    }

    fn elementwise(&self,left:&[f32],right:&[f32],shape:&[usize],op:ElementwiseOp)->Result<Vec<f32>,BackendError>{
        let expected=shape.iter().copied().product::<usize>();
        if left.len()!=expected||right.len()!=expected||left.len()!=right.len(){return Err(BackendError("elementwise GPU recebeu buffers incompatíveis".into()));}
        let a=self.create_buffer(&Self::bytes_f32(left),wgpu::BufferUsages::STORAGE);
        let b=self.create_buffer(&Self::bytes_f32(right),wgpu::BufferUsages::STORAGE);
        let code=match op{ElementwiseOp::Add=>0,ElementwiseOp::Sub=>1,ElementwiseOp::Mul=>2,ElementwiseOp::Div=>3};
        let params=self.create_buffer(&Self::bytes_u32(&[left.len() as u32,code,0,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch(&self.elementwise,&[&a,&b],&params,(((left.len() as u32)+255)/256,1,1),left.len())
    }

    fn fused_mul_add(
        &self,
        left: &[f32],
        right: &[f32],
        bias: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        let expected = shape.iter().copied().product::<usize>();
        if left.len() != expected || right.len() != expected || bias.len() != expected {
            return Err(BackendError("fused_mul_add GPU recebeu buffers incompatíveis".into()));
        }

        let a = self.create_buffer(&Self::bytes_f32(left), wgpu::BufferUsages::STORAGE);
        let b = self.create_buffer(&Self::bytes_f32(right), wgpu::BufferUsages::STORAGE);
        let c = self.create_buffer(&Self::bytes_f32(bias), wgpu::BufferUsages::STORAGE);
        let params = self.create_buffer(
            &Self::bytes_u32(&[left.len() as u32, 0, 0, 0]),
            wgpu::BufferUsages::UNIFORM,
        );

        self.dispatch(
            &self.fma,
            &[&a, &b, &c],
            &params,
            (((left.len() as u32) + 255) / 256, 1, 1),
            left.len(),
        )
    }

    fn reduce(&self,data:&[f32],mean:bool)->Result<f32,BackendError>{
        if data.is_empty(){return Ok(0.0);}
        let input=self.create_buffer(&Self::bytes_f32(data),wgpu::BufferUsages::STORAGE);
        let params=self.create_buffer(&Self::bytes_u32(&[data.len() as u32,0,0,0]),wgpu::BufferUsages::UNIFORM);
        let sum=self.dispatch(&self.reduce,&[&input],&params,(1,1,1),1)?[0];
        Ok(if mean{sum/data.len() as f32}else{sum})
    }

    fn transfer(&self,data:&[f32])->Result<Vec<f32>,BackendError>{self.transfer_f32(data)}
}

#[cfg(test)]
mod tests {
    #[test]
    fn gpu_backend_contains_real_compute_shaders() {
        assert!(!super::MATMUL_SHADER.is_empty());
        assert!(!super::ELEMENTWISE_SHADER.is_empty());
        assert!(!super::REDUCE_SHADER.is_empty());
    }
}
