use std::sync::{mpsc::{self, Receiver, Sender}, Arc};
use std::thread;

use pollster::block_on;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

pub(crate) enum UiWidgetSpec {
    Text {
        text: String,
        size: f32,
        color: [f32; 4],
    },
    Button {
        id: String,
        label: String,
        color: [f32; 4],
    },
    Rect {
        color: [f32; 4],
    },
}

pub(crate) enum UiCommand {
    SetTitle(String),
    Close,
    Clear([f32; 4]),
    Rect { x: f32, y: f32, width: f32, height: f32, color: [f32; 4] },
    Button { id: String, x: f32, y: f32, width: f32, height: f32, color: [f32; 4] },
    Text { text: String, x: f32, y: f32, size: f32, color: [f32; 4] },
    VBox { x: f32, y: f32, width: f32, row_height: f32, gap: f32, children: Vec<UiWidgetSpec> },
}

pub(crate) struct UiHandle {
    pub(crate) command: Sender<UiCommand>,
    pub(crate) events: Receiver<String>,
}

#[derive(Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    color: [f32; 4],
}

struct Button {
    id: String,
    rect: Rect,
}

#[derive(Clone)]
struct TextItem {
    text: String,
    x: f32,
    y: f32,
    size: f32,
    color: [f32; 4],
}

fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
        'B' => [0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110],
        'C' => [0b01111,0b10000,0b10000,0b10000,0b10000,0b10000,0b01111],
        'D' => [0b11110,0b10001,0b10001,0b10001,0b10001,0b10001,0b11110],
        'E' => [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111],
        'F' => [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000],
        'G' => [0b01111,0b10000,0b10000,0b10111,0b10001,0b10001,0b01111],
        'H' => [0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
        'I' => [0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b11111],
        'J' => [0b00111,0b00010,0b00010,0b00010,0b00010,0b10010,0b01100],
        'K' => [0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001],
        'L' => [0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111],
        'M' => [0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001],
        'N' => [0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001],
        'O' => [0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        'P' => [0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000],
        'Q' => [0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101],
        'R' => [0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001],
        'S' => [0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110],
        'T' => [0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100],
        'U' => [0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
        'V' => [0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100],
        'W' => [0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001],
        'X' => [0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001],
        'Y' => [0b10001,0b10001,0b01010,0b00100,0b00100,0b00100,0b00100],
        'Z' => [0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111],
        '0' => [0b01110,0b10011,0b10101,0b10101,0b10101,0b11001,0b01110],
        '1' => [0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110],
        '2' => [0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111],
        '3' => [0b11110,0b00001,0b00001,0b01110,0b00001,0b00001,0b11110],
        '4' => [0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010],
        '5' => [0b11111,0b10000,0b10000,0b11110,0b00001,0b00001,0b11110],
        '6' => [0b01110,0b10000,0b10000,0b11110,0b10001,0b10001,0b01110],
        '7' => [0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000],
        '8' => [0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110],
        '9' => [0b01110,0b10001,0b10001,0b01111,0b00001,0b00001,0b01110],
        '!' => [0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100],
        '.' => [0b00000,0b00000,0b00000,0b00000,0b00000,0b00110,0b00110],
        ':' => [0b00000,0b00110,0b00110,0b00000,0b00110,0b00110,0b00000],
        '-' => [0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000],
        '_' => [0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b11111],
        '/' => [0b00001,0b00010,0b00010,0b00100,0b01000,0b01000,0b10000],
        _ => [0,0,0,0,0,0,0],
    }
}


struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    config: wgpu::SurfaceConfiguration,
    vertex_buffer: wgpu::Buffer,
}

impl Renderer {
    fn new(window: Arc<Window>) -> Result<Self, String> {
        block_on(Self::new_async(window))
    }

    async fn new_async(window: Arc<Window>) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone())
            .map_err(|e| format!("Nano UI: surface: {e}"))?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: std::env::var_os("NANO_GPU_FALLBACK").is_some(),
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }).await.map_err(|e| format!("Nano UI: adapter: {e}"))?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("nano-ui-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }).await.map_err(|e| format!("Nano UI: device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = *caps.formats.first().ok_or_else(|| "Nano UI: surface sem formatos".to_string())?;
        let width = 1u32;
        let height = 1u32;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            color_space: wgpu::SurfaceColorSpace::Auto,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nano-ui-shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
"#)),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nano-ui-pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 6 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                        wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nano-ui-vertices"),
            size: 1024 * 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self { surface, device, queue, pipeline, config, vertex_buffer })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&self, clear: [f32; 4], rects: &[Rect], texts: &[TextItem]) -> Result<(), String> {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) |
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Outdated |
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Err("Nano UI: surface desatualizada; tente novamente".into());
            }
            wgpu::CurrentSurfaceTexture::Timeout => return Err("Nano UI: timeout ao adquirir surface".into()),
            wgpu::CurrentSurfaceTexture::Occluded => return Err("Nano UI: janela oculta".into()),
            wgpu::CurrentSurfaceTexture::Validation => return Err("Nano UI: erro de validação da surface".into()),
        };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let width = self.config.width.max(1) as f32;
        let height = self.config.height.max(1) as f32;
        let mut vertices: Vec<f32> = Vec::with_capacity((rects.len() + texts.len() * 64) * 6 * 6);

        let mut emit_rect = |rect: Rect| {
            let x0 = rect.x / width * 2.0 - 1.0;
            let x1 = (rect.x + rect.width) / width * 2.0 - 1.0;
            let y0 = 1.0 - rect.y / height * 2.0;
            let y1 = 1.0 - (rect.y + rect.height) / height * 2.0;
            let c = rect.color;
            for (x, y) in [(x0,y0),(x1,y0),(x1,y1),(x0,y0),(x1,y1),(x0,y1)] {
                vertices.extend([x, y, c[0], c[1], c[2], c[3]]);
            }
        };

        for &rect in rects {
            emit_rect(rect);
        }

        for text in texts {
            let pixel = (text.size / 7.0).max(1.0);
            let advance = pixel * 6.0;
            let mut cursor_x = text.x;
            for ch in text.text.chars() {
                if ch == ' ' {
                    cursor_x += advance;
                    continue;
                }
                let rows = glyph(ch);
                for (row, bits) in rows.iter().enumerate() {
                    for col in 0..5 {
                        if bits & (1 << (4 - col)) != 0 {
                            emit_rect(Rect {
                                x: cursor_x + col as f32 * pixel,
                                y: text.y + row as f32 * pixel,
                                width: pixel,
                                height: pixel,
                                color: text.color,
                            });
                        }
                    }
                }
                cursor_x += advance;
            }
        }

        if !vertices.is_empty() {
            let bytes: Vec<u8> = vertices.iter().flat_map(|value| value.to_ne_bytes()).collect();
            if bytes.len() > self.vertex_buffer.size() as usize {
                return Err("Nano UI: buffer de vértices excedido".into());
            }
            self.queue.write_buffer(&self.vertex_buffer, 0, &bytes);
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nano-ui-frame") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("nano-ui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0] as f64, g: clear[1] as f64, b: clear[2] as f64, a: clear[3] as f64
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            if !vertices.is_empty() {
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..(vertices.len() / 6) as u32, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}

struct UiApp {
    title: String,
    width: f64,
    height: f64,
    commands: Receiver<UiCommand>,
    events: Sender<String>,
    window: Option<Arc<Window>>,
    window_id: Option<WindowId>,
    renderer: Option<Renderer>,
    clear: [f32; 4],
    rects: Vec<Rect>,
    buttons: Vec<Button>,
    texts: Vec<TextItem>,
    cursor: (f32, f32),
}

impl ApplicationHandler for UiApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attributes = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(LogicalSize::new(self.width, self.height));

        match event_loop.create_window(attributes) {
            Ok(window) => {
                let window = Arc::new(window);
                self.window_id = Some(window.id());
                self.renderer = Renderer::new(window.clone()).ok();
                let _ = self.events.send(format!("created:{:?}", window.id()));
                self.window = Some(window);
            }
            Err(error) => {
                let _ = self.events.send(format!("error:{error}"));
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        if Some(window_id) != self.window_id { return; }

        match event {
            WindowEvent::CloseRequested => {
                let _ = self.events.send("close_requested".into());
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() { renderer.resize(size.width, size.height); }
                let _ = self.events.send(format!("resized:{}:{}", size.width, size.height));
            }
            WindowEvent::Focused(focused) => {
                let _ = self.events.send(format!("focused:{focused}"));
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
                let _ = self.events.send(format!("mouse_move:{:.3}:{:.3}", position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed && button == MouseButton::Left {
                    for widget in &self.buttons {
                        if self.cursor.0 >= widget.rect.x
                            && self.cursor.0 <= widget.rect.x + widget.rect.width
                            && self.cursor.1 >= widget.rect.y
                            && self.cursor.1 <= widget.rect.y + widget.rect.height
                        {
                            let _ = self.events.send(format!("button:{}", widget.id));
                        }
                    }
                }
                let _ = self.events.send(format!("mouse_button:{button:?}:{state:?}"));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let _ = self.events.send(format!("key:{:?}:{:?}", event.logical_key, event.state));
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &self.renderer {
                    if let Err(error) = renderer.render(self.clear, &self.rects, &self.texts) {
                        let _ = self.events.send(format!("error:{error}"));
                    }
                }
                let _ = self.events.send("redraw".into());
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        while let Ok(command) = self.commands.try_recv() {
            match command {
                UiCommand::SetTitle(title) => {
                    self.title = title.clone();
                    if let Some(window) = &self.window { window.set_title(&title); }
                }
                UiCommand::Clear(color) => {
                    self.clear = color;
                    self.rects.clear();
                    self.buttons.clear();
                    self.texts.clear();
                }
                UiCommand::Rect { x, y, width, height, color } => {
                    self.rects.push(Rect { x, y, width, height, color });
                }
                UiCommand::Button { id, x, y, width, height, color } => {
                    let rect = Rect { x, y, width, height, color };
                    self.rects.push(rect);
                    let label = id.clone();
                    self.buttons.push(Button { id, rect });
                    self.texts.push(TextItem {
                        text: label,
                        x: x + 8.0,
                        y: y + ((height - 14.0).max(0.0) * 0.5),
                        size: 14.0,
                        color: [1.0, 1.0, 1.0, 1.0],
                    });
                }
                UiCommand::Text { text, x, y, size, color } => {
                    self.texts.push(TextItem { text, x, y, size, color });
                }
                UiCommand::VBox { x, y, width, row_height, gap, children } => {
                    let mut current_y = y;
                    for child in children {
                        match child {
                            UiWidgetSpec::Text { text, size, color } => {
                                self.texts.push(TextItem {
                                    text,
                                    x: x + 8.0,
                                    y: current_y + 4.0,
                                    size,
                                    color,
                                });
                            }
                            UiWidgetSpec::Button { id, label, color } => {
                                let rect = Rect {
                                    x,
                                    y: current_y,
                                    width,
                                    height: row_height,
                                    color,
                                };
                                self.rects.push(rect);
                                self.buttons.push(Button { id, rect });
                                self.texts.push(TextItem {
                                    text: label,
                                    x: x + 8.0,
                                    y: current_y + ((row_height - 14.0).max(0.0) * 0.5),
                                    size: 14.0,
                                    color: [1.0, 1.0, 1.0, 1.0],
                                });
                            }
                            UiWidgetSpec::Rect { color } => {
                                self.rects.push(Rect { x, y: current_y, width, height: row_height, color });
                            }
                        }
                        current_y += row_height + gap;
                    }
                }
                UiCommand::Close => {
                    let _ = self.events.send("closed".into());
                    event_loop.exit();
                }
            }
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

pub(crate) fn spawn(title: String, width: f64, height: f64) -> Result<UiHandle, String> {
    if width <= 0.0 || height <= 0.0 {
        return Err("Nano UI: largura e altura devem ser positivas".into());
    }

    let (command_tx, command_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();

    thread::Builder::new().name("nano-ui".into()).spawn(move || {
        let Ok(event_loop) = EventLoop::new() else {
            let _ = event_tx.send("error:event-loop".into());
            return;
        };

        let mut app = UiApp {
            title,
            width,
            height,
            commands: command_rx,
            events: event_tx,
            window: None,
            window_id: None,
            renderer: None,
            clear: [0.08, 0.08, 0.10, 1.0],
            rects: Vec::new(),
            buttons: Vec::new(),
            texts: Vec::new(),
            cursor: (0.0, 0.0),
        };

        let _ = event_loop.run_app(&mut app);
    }).map_err(|e| format!("Nano UI: não foi possível criar thread: {e}"))?;

    Ok(UiHandle { command: command_tx, events: event_rx })
}
