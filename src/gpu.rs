use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
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

const TRANSPOSED_MATMUL_SHADER: &str = r#"
struct Params {
    m: u32, k: u32, n: u32,
    a_transpose: u32, b_transpose: u32, _pad0: u32, _pad1: u32, _pad2: u32
};
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    if (row >= params.m || col >= params.n) { return; }
    var acc = 0.0;
    for (var i = 0u; i < params.k; i = i + 1u) {
        let ai = if (params.a_transpose == 1u) { i * params.m + row } else { row * params.k + i };
        let bi = if (params.b_transpose == 1u) { col * params.k + i } else { i * params.n + col };
        acc = acc + a[ai] * b[bi];
    }
    out[row * params.n + col] = acc;
}
"#;

const SCALE_SHADER: &str = r#"
struct Params { len: u32, scale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    out[i] = input[i] * params.scale;
}
"#;

const BROADCAST_SHADER: &str = r#"
struct Params { len: u32, scale: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> scalar: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    out[i] = scalar[0] * params.scale;
}
"#;

const FILL_SHADER: &str = r#"
struct Params { len: u32, value: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    out[i] = params.value;
}
"#;

const STEP_SHADER: &str = r#"
struct Params { len: u32, lr: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    param[i] = param[i] - params.lr * grad[i];
}
"#;

const ADAM_SHADER: &str = r#"
struct Params { len: u32, step: u32, lr: f32, _pad0: u32 };
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }

    let g = grad[i];
    let m_new = 0.9 * m[i] + 0.1 * g;
    let v_new = 0.999 * v[i] + 0.001 * g * g;
    m[i] = m_new;
    v[i] = v_new;

    let step_f = f32(params.step);
    let m_hat = m_new / (1.0 - pow(0.9, step_f));
    let v_hat = v_new / (1.0 - pow(0.999, step_f));
    param[i] = param[i] - params.lr * m_hat / (sqrt(v_hat) + 1e-8);
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
struct Params { len: u32, mean: u32, _pad1: u32, _pad2: u32 };
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
    if (lane == 0u) {
        if (params.mean == 1u) { out[0] = scratch[0] / f32(params.len); }
        else { out[0] = scratch[0]; }
    }
}
"#;

struct ResidentBuffer {
    buffer: wgpu::Buffer,
    logical_bytes: usize,
    block: crate::memory::MemoryBlock,
}

pub(crate) struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul: wgpu::ComputePipeline,
    elementwise: wgpu::ComputePipeline,
    fma: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    transposed_matmul: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    broadcast: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    step: wgpu::ComputePipeline,
    adam: wgpu::ComputePipeline,
    planner: RefCell<MemoryPlanner>,
    resident: RefCell<HashMap<u64, ResidentBuffer>>,
}

impl GpuBackend {
    pub(crate) fn new() -> Result<Self, BackendError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let force_fallback = std::env::var_os("NANO_GPU_FALLBACK").is_some();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: force_fallback,
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
        let transposed_matmul_module = make_shader("nano-transposed-matmul", TRANSPOSED_MATMUL_SHADER);
        let scale_module = make_shader("nano-scale", SCALE_SHADER);
        let broadcast_module = make_shader("nano-broadcast", BROADCAST_SHADER);
        let fill_module = make_shader("nano-fill", FILL_SHADER);
        let step_module = make_shader("nano-step", STEP_SHADER);
        let adam_module = make_shader("nano-adam", ADAM_SHADER);

        let matmul = make_pipeline("nano-matmul-pipeline", &matmul_module);
        let elementwise = make_pipeline("nano-elementwise-pipeline", &elementwise_module);
        let fma = make_pipeline("nano-fma-pipeline", &fma_module);
        let reduce = make_pipeline("nano-reduce-pipeline", &reduce_module);
        let transposed_matmul = make_pipeline("nano-transposed-matmul-pipeline", &transposed_matmul_module);
        let scale = make_pipeline("nano-scale-pipeline", &scale_module);
        let broadcast = make_pipeline("nano-broadcast-pipeline", &broadcast_module);
        let fill = make_pipeline("nano-fill-pipeline", &fill_module);
        let step = make_pipeline("nano-step-pipeline", &step_module);
        let adam = make_pipeline("nano-adam-pipeline", &adam_module);

        Ok(Self {
            device,
            queue,
            matmul,
            elementwise,
            fma,
            reduce,
            transposed_matmul,
            scale,
            broadcast,
            fill,
            step,
            adam,
            planner: RefCell::new(MemoryPlanner::new()),
            resident: RefCell::new(HashMap::new()),
        })
    }


    fn resident_buffer(&self,id:u64,elements:usize)->Result<wgpu::Buffer,BackendError>{
        if elements==0{return Err(BackendError("tensor vazio não requer buffer GPU".into()));}
        let logical_bytes=elements.saturating_mul(DType::F32.bytes());
        if let Some(entry)=self.resident.borrow().get(&id){
            if entry.logical_bytes>=logical_bytes{return Ok(entry.buffer.clone());}
        }
        if let Some(old)=self.resident.borrow_mut().remove(&id){
            self.planner.borrow_mut().release(old.block);
        }
        let block=self.planner.borrow_mut().allocate(elements,DType::F32);
        let buffer=self.device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("nano-gpu-resident"),size:block.size as u64,
            usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:false,
        });
        self.resident.borrow_mut().insert(id,ResidentBuffer{buffer:buffer.clone(),logical_bytes,block});
        Ok(buffer)
    }

    fn ensure_resident(&self,id:u64,data:&[f32])->Result<wgpu::Buffer,BackendError>{
        let logical_bytes=data.len().saturating_mul(4);
        let had_sized=self.resident.borrow().get(&id).map(|e|e.logical_bytes>=logical_bytes).unwrap_or(false);
        let buffer=self.resident_buffer(id,data.len())?;
        if !had_sized && !data.is_empty(){self.queue.write_buffer(&buffer,0,&Self::bytes_f32(data));}
        Ok(buffer)
    }

    fn encode_resident_dispatch(
        &self,
        pipeline:&wgpu::ComputePipeline,
        inputs:&[&wgpu::Buffer],
        uniform:&wgpu::Buffer,
        groups:(u32,u32,u32),
        output:&wgpu::Buffer,
    ) -> wgpu::CommandBuffer {
        let mut entries=Vec::with_capacity(inputs.len()+2);
        for(i,buffer)in inputs.iter().enumerate(){entries.push(wgpu::BindGroupEntry{binding:i as u32,resource:buffer.as_entire_binding()});}
        entries.push(wgpu::BindGroupEntry{binding:inputs.len() as u32,resource:output.as_entire_binding()});
        entries.push(wgpu::BindGroupEntry{binding:(inputs.len()+1) as u32,resource:uniform.as_entire_binding()});
        let bind_group=self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("nano-gpu-resident-bind-group"),
            layout:&pipeline.get_bind_group_layout(0),
            entries:&entries
        });
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("nano-gpu-resident-command")});
        {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("nano-gpu-resident-compute"),timestamp_writes:None});
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0,&bind_group,&[]);
            pass.dispatch_workgroups(groups.0,groups.1,groups.2);
        }
        encoder.finish()
    }

    fn dispatch_resident(&self,pipeline:&wgpu::ComputePipeline,inputs:&[&wgpu::Buffer],uniform:&wgpu::Buffer,groups:(u32,u32,u32),output:&wgpu::Buffer){
        let command=self.encode_resident_dispatch(pipeline,inputs,uniform,groups,output);
        self.queue.submit(Some(command));
    }

    fn dispatch_into(&self,pipeline:&wgpu::ComputePipeline,inputs:&[&wgpu::Buffer],uniform:&wgpu::Buffer,groups:(u32,u32,u32),output:&wgpu::Buffer,output_len:usize)->Result<Vec<f32>,BackendError>{
        if output_len==0{return Ok(Vec::new());}
        let output_bytes=self.planner.borrow().bytes_for(output_len,DType::F32);
        let readback=self.device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("nano-gpu-readback"),size:output_bytes as u64,
            usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
        });
        let command=self.encode_resident_dispatch(pipeline,inputs,uniform,groups,output);
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("nano-gpu-readback-command")});
        encoder.copy_buffer_to_buffer(output,0,&readback,0,(output_len*4) as u64);
        self.queue.submit(Some(command));
        self.queue.submit(Some(encoder.finish()));
        self.readback(&readback,output_len)
    }

    fn dispatch_resident_1(&self,pipeline:&wgpu::ComputePipeline,inputs:&[&wgpu::Buffer],uniform:&wgpu::Buffer,groups:u32,output:&wgpu::Buffer){
        self.dispatch_resident(pipeline,inputs,uniform,(groups,1,1),output);
    }

    fn dispatch_inplace(&self,pipeline:&wgpu::ComputePipeline,inputs:&[&wgpu::Buffer],uniform:&wgpu::Buffer,groups:u32){
        let mut entries=Vec::with_capacity(inputs.len()+1);
        for(i,buffer)in inputs.iter().enumerate(){
            entries.push(wgpu::BindGroupEntry{binding:i as u32,resource:buffer.as_entire_binding()});
        }
        entries.push(wgpu::BindGroupEntry{binding:inputs.len() as u32,resource:uniform.as_entire_binding()});
        let bind_group=self.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("nano-gpu-inplace-bind-group"),
            layout:&pipeline.get_bind_group_layout(0),
            entries:&entries
        });
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("nano-gpu-inplace-command")});
        {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("nano-gpu-inplace"),timestamp_writes:None});
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0,&bind_group,&[]);
            pass.dispatch_workgroups(groups,1,1);
        }
        self.queue.submit(Some(encoder.finish()));
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
        let output_bytes = self.planner.borrow().bytes_for(output_len, DType::F32);
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
        let bytes = self.planner.borrow().bytes_for(data.len(), DType::F32);
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

    fn matmul_resident_async(&self,left_id:u64,left_shape:&[usize],right_id:u64,right_shape:&[usize],output_id:u64)->Result<(),BackendError>{
        if left_shape.len()!=2||right_shape.len()!=2{return Err(BackendError("matmul GPU requer tensores 2D".into()));}
        let(m,k)=(left_shape[0],left_shape[1]);let(k2,n)=(right_shape[0],right_shape[1]);
        if k!=k2{return Err(BackendError("matmul GPU recebeu shapes incompatíveis".into()));}
        let a=self.resident.borrow().get(&left_id).ok_or_else(||BackendError("tensor esquerdo não está residente na GPU".into()))?.buffer.clone();
        let b=self.resident.borrow().get(&right_id).ok_or_else(||BackendError("tensor direito não está residente na GPU".into()))?.buffer.clone();
        let out=self.resident_buffer(output_id,m*n)?;
        let params=self.create_buffer(&Self::bytes_u32(&[m as u32,k as u32,n as u32,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident(&self.matmul,&[&a,&b],&params,(((n as u32)+7)/8,((m as u32)+7)/8,1),&out);
        Ok(())
    }

    fn matmul_resident(&self,left_id:u64,left:&[f32],left_shape:&[usize],right_id:u64,right:&[f32],right_shape:&[usize],output_id:u64)->Result<Vec<f32>,BackendError>{
        if left_shape.len()!=2||right_shape.len()!=2{return Err(BackendError("matmul GPU requer tensores 2D".into()));}
        let(m,k)=(left_shape[0],left_shape[1]);let(k2,n)=(right_shape[0],right_shape[1]);
        if k!=k2||left.len()!=m*k||right.len()!=k2*n{return Err(BackendError("matmul GPU recebeu shapes incompatíveis".into()));}
        let a=self.ensure_resident(left_id,left)?;let b=self.ensure_resident(right_id,right)?;let out=self.resident_buffer(output_id,m*n)?;
        let params=self.create_buffer(&Self::bytes_u32(&[m as u32,k as u32,n as u32,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_into(&self.matmul,&[&a,&b],&params,(((n as u32)+7)/8,((m as u32)+7)/8,1),&out,m*n)
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

    fn elementwise_resident_async(&self,left_id:u64,right_id:u64,shape:&[usize],op:ElementwiseOp,output_id:u64)->Result<(),BackendError>{
        let expected=shape.iter().copied().product::<usize>();
        let a=self.resident.borrow().get(&left_id).ok_or_else(||BackendError("tensor esquerdo não está residente na GPU".into()))?.buffer.clone();
        let b=self.resident.borrow().get(&right_id).ok_or_else(||BackendError("tensor direito não está residente na GPU".into()))?.buffer.clone();
        let out=self.resident_buffer(output_id,expected)?;
        let code=match op{ElementwiseOp::Add=>0,ElementwiseOp::Sub=>1,ElementwiseOp::Mul=>2,ElementwiseOp::Div=>3};
        let params=self.create_buffer(&Self::bytes_u32(&[expected as u32,code,0,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident(&self.elementwise,&[&a,&b],&params,(((expected as u32)+255)/256,1,1),&out);
        Ok(())
    }

    fn elementwise_resident(&self,left_id:u64,left:&[f32],right_id:u64,right:&[f32],shape:&[usize],op:ElementwiseOp,output_id:u64)->Result<Vec<f32>,BackendError>{
        let expected=shape.iter().copied().product::<usize>();
        if left.len()!=expected||right.len()!=expected||left.len()!=right.len(){return Err(BackendError("elementwise GPU recebeu buffers incompatíveis".into()));}
        let a=self.ensure_resident(left_id,left)?;let b=self.ensure_resident(right_id,right)?;let out=self.resident_buffer(output_id,left.len())?;
        let code=match op{ElementwiseOp::Add=>0,ElementwiseOp::Sub=>1,ElementwiseOp::Mul=>2,ElementwiseOp::Div=>3};
        let params=self.create_buffer(&Self::bytes_u32(&[left.len() as u32,code,0,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_into(&self.elementwise,&[&a,&b],&params,(((left.len() as u32)+255)/256,1,1),&out,left.len())
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

    fn fused_mul_add_resident_async(&self,left_id:u64,right_id:u64,bias_id:u64,shape:&[usize],output_id:u64)->Result<(),BackendError>{
        let expected=shape.iter().copied().product::<usize>();
        let a=self.resident.borrow().get(&left_id).ok_or_else(||BackendError("tensor esquerdo não está residente na GPU".into()))?.buffer.clone();
        let b=self.resident.borrow().get(&right_id).ok_or_else(||BackendError("tensor direito não está residente na GPU".into()))?.buffer.clone();
        let c=self.resident.borrow().get(&bias_id).ok_or_else(||BackendError("bias não está residente na GPU".into()))?.buffer.clone();
        let out=self.resident_buffer(output_id,expected)?;
        let params=self.create_buffer(&Self::bytes_u32(&[expected as u32,0,0,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident(&self.fma,&[&a,&b,&c],&params,(((expected as u32)+255)/256,1,1),&out);
        Ok(())
    }

    fn fused_mul_add_resident(&self,left_id:u64,left:&[f32],right_id:u64,right:&[f32],bias_id:u64,bias:&[f32],shape:&[usize],output_id:u64)->Result<Vec<f32>,BackendError>{
        let expected=shape.iter().copied().product::<usize>();
        if left.len()!=expected||right.len()!=expected||bias.len()!=expected{return Err(BackendError("fused_mul_add GPU recebeu buffers incompatíveis".into()));}
        let a=self.ensure_resident(left_id,left)?;let b=self.ensure_resident(right_id,right)?;let c=self.ensure_resident(bias_id,bias)?;let out=self.resident_buffer(output_id,left.len())?;
        let params=self.create_buffer(&Self::bytes_u32(&[left.len() as u32,0,0,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_into(&self.fma,&[&a,&b,&c],&params,(((left.len() as u32)+255)/256,1,1),&out,left.len())
    }

    fn sync_tensor(&self,id:u64,data:&[f32])->Result<(),BackendError>{
        if data.is_empty(){return Ok(());}
        let buffer=self.resident_buffer(id,data.len())?;
        self.queue.write_buffer(&buffer,0,&Self::bytes_f32(data));
        Ok(())
    }

    fn release_tensor(&self,id:u64)->Result<(),BackendError>{
        if let Some(entry)=self.resident.borrow_mut().remove(&id){self.planner.borrow_mut().release(entry.block);}
        Ok(())
    }

    fn fill_resident_async(&self,elements:usize,value:f32,output_id:u64)->Result<(),BackendError>{
        let out=self.resident_buffer(output_id,elements)?;
        let params2=self.create_buffer(&[
            (elements as u32).to_ne_bytes().as_slice(),
            value.to_ne_bytes().as_slice(),
            &[0;4],
            &[0;4],
        ].concat(),wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident_1(&self.fill,&[],&params2,((elements as u32)+255)/256,&out);

        Ok(())
    }

    fn scale_resident_async(&self,input_id:u64,elements:usize,scale:f32,output_id:u64)->Result<(),BackendError>{
        let input=self.resident.borrow().get(&input_id).ok_or_else(||BackendError("tensor não está residente na GPU".into()))?.buffer.clone();
        let out=self.resident_buffer(output_id,elements)?;
        let bytes=[
            (elements as u32).to_ne_bytes().as_slice(),
            scale.to_ne_bytes().as_slice(),
            &[0;4],&[0;4],
        ].concat();
        let params=self.create_buffer(&bytes,wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident_1(&self.scale,&[&input],&params,((elements as u32)+255)/256,&out);
        Ok(())
    }

    fn broadcast_resident_async(&self,scalar_id:u64,elements:usize,scale:f32,output_id:u64)->Result<(),BackendError>{
        let scalar=self.resident.borrow().get(&scalar_id).ok_or_else(||BackendError("escalar não está residente na GPU".into()))?.buffer.clone();
        let out=self.resident_buffer(output_id,elements)?;
        let bytes=[
            (elements as u32).to_ne_bytes().as_slice(),
            scale.to_ne_bytes().as_slice(),
            &[0;4],&[0;4],
        ].concat();
        let params=self.create_buffer(&bytes,wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident_1(&self.broadcast,&[&scalar],&params,((elements as u32)+255)/256,&out);
        Ok(())
    }

    fn matmul_transposed_resident_async(&self,a_id:u64,a_shape:&[usize],b_id:u64,b_shape:&[usize],a_transpose:bool,b_transpose:bool,output_id:u64,output_shape:&[usize])->Result<(),BackendError>{
        if a_shape.len()!=2||b_shape.len()!=2||output_shape.len()!=2{return Err(BackendError("matmul transposto requer shapes 2D".into()));}
        let a=self.resident.borrow().get(&a_id).ok_or_else(||BackendError("tensor A não está residente na GPU".into()))?.buffer.clone();
        let b=self.resident.borrow().get(&b_id).ok_or_else(||BackendError("tensor B não está residente na GPU".into()))?.buffer.clone();
        let out_elems=output_shape.iter().copied().product::<usize>();
        let out=self.resident_buffer(output_id,out_elems)?;
        let a_rows=a_shape[0]; let a_cols=a_shape[1];
        let b_rows=b_shape[0]; let b_cols=b_shape[1];
        let m=output_shape[0]; let n=output_shape[1];
        let k=if a_transpose{a_rows}else{a_cols};
        let k_b=if b_transpose{b_cols}else{b_rows};
        if k!=k_b{return Err(BackendError("matmul transposto recebeu shapes incompatíveis".into()));}
        let bytes=[
            (m as u32).to_ne_bytes().as_slice(),
            (k as u32).to_ne_bytes().as_slice(),
            (n as u32).to_ne_bytes().as_slice(),
            (if a_transpose{1u32}else{0}).to_ne_bytes().as_slice(),
            (if b_transpose{1u32}else{0}).to_ne_bytes().as_slice(),
            &[0;4],&[0;4],&[0;4],
        ].concat();
        let params=self.create_buffer(&bytes,wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident(&self.transposed_matmul,&[&a,&b],&params,(((n as u32)+7)/8,((m as u32)+7)/8,1),&out);
        Ok(())
    }

    fn step_resident_async(&self,param_id:u64,grad_id:u64,elements:usize,lr:f32)->Result<(),BackendError>{
        let param=self.resident.borrow().get(&param_id).ok_or_else(||BackendError("parâmetro não está residente na GPU".into()))?.buffer.clone();
        let grad=self.resident.borrow().get(&grad_id).ok_or_else(||BackendError("gradiente não está residente na GPU".into()))?.buffer.clone();
        let bytes=[(elements as u32).to_ne_bytes().as_slice(),lr.to_ne_bytes().as_slice(),&[0;4],&[0;4]].concat();
        let params=self.create_buffer(&bytes,wgpu::BufferUsages::UNIFORM);
        self.dispatch_inplace(&self.step,&[&param,&grad],&params,((elements as u32)+255)/256);
        Ok(())
    }

    fn adam_resident_async(&self,param_id:u64,grad_id:u64,elements:usize,lr:f32,step:u32)->Result<(),BackendError>{
        let param=self.resident.borrow().get(&param_id).ok_or_else(||BackendError("parâmetro não está residente na GPU".into()))?.buffer.clone();
        let grad=self.resident.borrow().get(&grad_id).ok_or_else(||BackendError("gradiente não está residente na GPU".into()))?.buffer.clone();
        let m_id=(1u64<<63)|param_id.wrapping_mul(2);
        let v_id=m_id.wrapping_add(1);
        let m_exists=self.resident.borrow().contains_key(&m_id);
        let v_exists=self.resident.borrow().contains_key(&v_id);
        let m=self.resident_buffer(m_id,elements)?;
        let v=self.resident_buffer(v_id,elements)?;
        if !m_exists {
            self.fill_resident_async(elements,0.0,m_id)?;
        }
        if !v_exists {
            self.fill_resident_async(elements,0.0,v_id)?;
        }
        let bytes=[
            (elements as u32).to_ne_bytes().as_slice(),
            step.to_ne_bytes().as_slice(),
            lr.to_ne_bytes().as_slice(),
            &[0;4],
        ].concat();
        let params=self.create_buffer(&bytes,wgpu::BufferUsages::UNIFORM);
        self.dispatch_inplace(&self.adam,&[&param,&grad,&m,&v],&params,((elements as u32)+255)/256);
        Ok(())
    }

    fn reduce_resident_async(&self,input_id:u64,elements:usize,mean:bool,output_id:u64)->Result<(),BackendError>{
        let input=self.resident.borrow().get(&input_id).ok_or_else(||BackendError("tensor de redução não está residente na GPU".into()))?.buffer.clone();
        let output=self.resident_buffer(output_id,1)?;
        let params=self.create_buffer(&Self::bytes_u32(&[elements as u32,if mean{1}else{0},0,0]),wgpu::BufferUsages::UNIFORM);
        self.dispatch_resident(&self.reduce,&[&input],&params,(1,1,1),&output);
        Ok(())
    }

    fn read_tensor(&self,id:u64,elements:usize)->Result<Vec<f32>,BackendError>{
        let buffer=self.resident.borrow().get(&id).ok_or_else(||BackendError("tensor não está residente na GPU".into()))?.buffer.clone();
        let bytes=self.planner.borrow().bytes_for(elements,DType::F32);
        let staging=self.device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("nano-gpu-explicit-readback"),size:bytes as u64,
            usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false,
        });
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("nano-gpu-explicit-readback-command")});
        encoder.copy_buffer_to_buffer(&buffer,0,&staging,0,(elements*4) as u64);
        self.queue.submit(Some(encoder.finish()));
        self.readback(&staging,elements)
    }

    fn reduce(&self,data:&[f32],mean:bool)->Result<f32,BackendError>{
        if data.is_empty(){return Ok(0.0);}
        let input=self.create_buffer(&Self::bytes_f32(data),wgpu::BufferUsages::STORAGE);
        let params=self.create_buffer(&Self::bytes_u32(&[data.len() as u32,if mean{1}else{0},0,0]),wgpu::BufferUsages::UNIFORM);
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

    #[test]
    fn gpu_matmul_can_execute_with_fallback_adapter() {
        if std::env::var_os("NANO_GPU_E2E").is_none() {
            return;
        }

        std::env::set_var("NANO_GPU_FALLBACK", "1");
        let backend = super::GpuBackend::new().expect("wgpu fallback adapter");
        let output = backend
            .matmul(
                &[1.0, 2.0, 3.0, 4.0],
                &[2, 2],
                &[5.0, 6.0, 7.0, 8.0],
                &[2, 2],
            )
            .expect("GPU matmul");

        assert_eq!(output.len(), 4);
        assert!((output[0] - 19.0).abs() < 1e-4);
        assert!((output[1] - 22.0).abs() < 1e-4);
        assert!((output[2] - 43.0).abs() < 1e-4);
        assert!((output[3] - 50.0).abs() < 1e-4);
    }
}
